import gym
import os
import numpy as np
import matplotlib.pyplot as plt

from copy import copy
from collections import deque
from multiprocessing import Lock

import notify

from time_buffer import *
from score_buffer import *
from colors import *
from schedulers import *

# Global variables
bench_episodes = 50
memlock = Lock()
memthresh = 0.3

# Sampling from replay buffers
def sample(rbf, size):
    indices = np.random.randint(len(rbf), size = size)
    batch = [rbf[index] for index in indices]

    states, actions, rewards, nstates, dones = [
            np.array(
                    [experience[field_index] for experience in batch]
            )
    for field_index in range(5)]

    return states, actions, rewards, nstates, dones

def train(tf, model, rbf, batch_size, loss_ftn, optimizer, gamma, nouts):
    states, actions, rewards, nstates, dones = sample(rbf, batch_size)

    next_Qvs = model(nstates)
    max_next_Qvs = np.max(next_Qvs, axis = 1)

    tgt_Qvs = (rewards + (1 - dones) * gamma * max_next_Qvs)
    tgt_Qvs = tgt_Qvs.reshape(-1, 1)

    mask = tf.one_hot(actions, nouts)
    with tf.GradientTape() as tape:
        all_Qvs = model(states)
        Qvs = tf.reduce_sum(all_Qvs * mask, axis = 1, keepdims = True)
        loss = tf.reduce_mean(loss_ftn(tgt_Qvs, Qvs))

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

def run_policy(ename, skeleton, heurestic, schedref, trial, episodes, steps, dirn):
    # Assume only using GPU 0 (for my system)
    import nvidia_smi

    # Named process id
    scheduler = copy(schedref)
    id = ename + ': ' + heurestic.name + ' and ' + scheduler.name + ': ' + \
        str(trial + 1)

    # TODO: put in function
    i = 0
    while True:
        memlock.acquire()

        nvidia_smi.nvmlInit()

        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

        # info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        # print(YELLOW + '[CHECK]' + RESET + f' {id}, L{i}: Available GPU memory {info.free/info.total:%}')
        if info.free/info.total < memthresh:
            # print(RED + 'Reached 50% threshold, waiting (5s)...' + RESET)
            memlock.release()
            time.sleep(5)
        else:
            break

        i += 1

    # Now import tensorflow
    import tensorflow as tf

    from tensorflow import keras

    # Config GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # First two layers
    first_layer = keras.layers.Dense(skeleton[2][0], activation = skeleton[2][1], input_shape = skeleton[0])
    last_layer = keras.layers.Dense(skeleton[1])

    # Populating the layers
    layers = [first_layer]
    for sl in skeleton[3:]:
        layers.append(keras.layers.Dense(sl[0], activation = sl[1]))
    layers.append(last_layer)

    # Creating the models
    model = keras.models.Sequential(layers)

    # Other variables
    env = gym.make(ename)
    eps = 1
    rbf = deque(maxlen = 2000)  # Should be customizable
    batch_size = 32             # Should be customizable

    gamma = 0.95                # Should be customizable
    loss_ftn = keras.losses.mean_squared_error
    optimizer = keras.optimizers.Adam(learning_rate = 1e-2)
    nouts = env.action_space.n  # Should be customizable

    start = time.time()

    scores = []
    epsilons = []

    # Training loop
    tb = TimeBuffer(episodes)
    proj = 0

    memlock.release()
    # print(YELLOW + '[CHECK]' + RESET + ' MEMLOCK MUTEX RELEASED!')
    for e in range(episodes):
        # Get the first observation
        state = env.reset()
        score = 0

        tb.start()
        for s in range(steps):
            # Get the action
            if np.random.rand() < eps:
                action = heurestic(state)
            else:
                Qvs = model(state[np.newaxis])
                action = np.argmax(Qvs[0])

            # Apply the action and update the state
            nstate, reward, done, info = env.step(action)
            rbf.append((state, action, reward, nstate, done))
            state = nstate
            score += reward

            if done:
                break

        # Post episode routines
        tb.split()
        proj, fmt1, fmt2 = cmp_and_fmt(proj, tb.projected())

        # Logging
        msg = f'{id:<50} Episode {e:<5} Score = {score:<5} Epsilon {eps:.2f}' \
                f' Time = [{str(tb):<20}], Projected'

        print(msg + f' [{fmt1}]')
        if e % notify.interval == 1:
            notify.log(msg + f' [{fmt2}]')

        # Post processing
        scores.append(score)
        epsilons.append(eps)
        eps = scheduler()

        # Train the model if the rbf is full enough
        if len(rbf) >= batch_size:
            train(tf, model, rbf, batch_size, loss_ftn, optimizer, gamma, nouts)

    # Training loop
    finals = []
    for e in range(bench_episodes):
        # Get the first observation
        state = env.reset()
        score = 0

        for s in range(steps):
            # Get the action
            Qvs = model(state[np.newaxis])
            action = np.argmax(Qvs[0])

            # Apply the action and update the state
            nstate, reward, done, info = env.step(action)
            state = nstate
            score += reward

            if done:
                break

        finals.append(score)

    # Log completion
    msg = f'{id} finished in {fmt_time(time.time() - start)}'
    print(GREEN + msg + RESET)
    notify.notify(msg)

    # Record the results
    pdir = dirn + '/' + ename + '/' + (heurestic.name + '_and_' + \
        scheduler.name).replace(' ', '_')
    os.system(f'mkdir -p {pdir}')
    fname = pdir + f'/Trial_{trial + 1}.csv'
    fout = open(fname, 'w')
    fout.writelines([
        'Epsilons, ' + ','.join([str(e) for e in epsilons]) + '\n',
        'Scores, ' + ','.join([str(s) for s in scores]) + '\n',
        'Finals, ' + ','.join([str(f) for f in finals])
    ])
    fout.close()

def run_policy_ret(ename, skeleton, heurestic, schedref, trial, episodes, steps):
    # Now import tensorflow
    import tensorflow as tf

    from tensorflow import keras

    # Config GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # Initialize the scheduler and id
    scheduler = copy(schedref)
    id = ename + ': ' + heurestic.name + ' and ' + scheduler.name + ': ' + \
        str(trial + 1)

    # First two layers
    first_layer = keras.layers.Dense(skeleton[2][0], activation = skeleton[2][1], input_shape = skeleton[0])
    last_layer = keras.layers.Dense(skeleton[1])

    # Populating the layers
    layers = [first_layer]
    for sl in skeleton[3:]:
        layers.append(keras.layers.Dense(sl[0], activation = sl[1]))
    layers.append(last_layer)

    # Creating the models
    model = keras.models.Sequential(layers)

    # Other variables
    env = gym.make(ename)
    eps = 1
    rbf = deque(maxlen = 2000)  # Should be customizable
    batch_size = 32             # Should be customizable

    gamma = 0.95                # Should be customizable
    loss_ftn = keras.losses.mean_squared_error
    optimizer = keras.optimizers.Adam(learning_rate = 1e-2)
    nouts = env.action_space.n  # Should be customizable

    start = time.time()

    scores = []
    epsilons = []

    # Training loop
    tb = TimeBuffer(episodes)
    proj = 0

    for e in range(episodes):
        # Get the first observation
        state = env.reset()
        score = 0

        tb.start()
        for s in range(steps):
            # Get the action
            if np.random.rand() < eps:
                action = heurestic(state)
            else:
                Qvs = model(state[np.newaxis])
                action = np.argmax(Qvs[0])

            # Apply the action and update the state
            nstate, reward, done, info = env.step(action)
            rbf.append((state, action, reward, nstate, done))
            state = nstate
            score += reward

            if done:
                break

        # Post episode routines
        tb.split()
        proj, fmt1, fmt2 = cmp_and_fmt(proj, tb.projected())

        # Logging
        msg = f'{id:<50} Episode {e:<5} Score = {score:<5} Epsilon {eps:.2f}' \
                f' Time = [{str(tb):<20}], Projected'

        print(msg + f' [{fmt1}]')
        if e % notify.interval == 1:
            notify.log(msg + f' [{fmt2}]')

        # Post processing
        scores.append(score)
        epsilons.append(eps)
        eps = scheduler()

        # Train the model if the rbf is full enough
        if len(rbf) >= batch_size:
            train(tf, model, rbf, batch_size, loss_ftn, optimizer, gamma, nouts)

    # Record the results
    return scores

def run_tutoring(ename, skeleton, heurestic, schedref, trial, episodes, steps, dirn):
    # Assume only using GPU 0 (for my system)
    import nvidia_smi

    # Named process id
    scheduler = copy(schedref)
    id = ename + ': TS: ' + heurestic.name + ' and ' + \
            scheduler.name + ': ' + str(trial + 1)

    # TODO: put in function
    i = 0
    while True:
        memlock.acquire()

        nvidia_smi.nvmlInit()

        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

        # info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        # print(YELLOW + '[CHECK]' + RESET + f' {id}, L{i}: Available GPU memory {info.free/info.total:%}')
        if info.free/info.total < memthresh:
            # print(RED + 'Reached 50% threshold, waiting (5s)...' + RESET)
            memlock.release()
            time.sleep(5)
        else:
            break

        i += 1

    import tensorflow as tf

    from tensorflow import keras

    # Config GPUs (consume only half the memory)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # First two layers
    first_layer = keras.layers.Dense(skeleton[2][0], activation = skeleton[2][1], input_shape = skeleton[0])
    last_layer = keras.layers.Dense(skeleton[1])

    # Populating the layers
    layers = [first_layer]
    for sl in skeleton[3:]:
        layers.append(keras.layers.Dense(sl[0], activation = sl[1]))
    layers.append(last_layer)

    # Creating the models
    m1 = keras.models.Sequential(layers)
    m2 = keras.models.Sequential(layers)

    # Other variables
    env1 = gym.make(ename)
    env2 = gym.make(ename)

    eps = 1
    keps = (1 - eps)/2

    rbf1 = deque(maxlen = 2000)  # Should be customizable
    rbf2 = deque(maxlen = 2000)  # Should be customizable

    batch_size = 32             # Should be customizable
    scheduler = copy(schedref)

    gamma = 0.95                # Should be customizable
    loss_ftn = keras.losses.mean_squared_error
    optimizer = keras.optimizers.Adam(learning_rate = 1e-2)
    nouts = env1.action_space.n

    start = time.time()

    scores1 = []
    scores2 = []

    buf1 = ScoreBuffer()
    buf2 = ScoreBuffer()

    epsilons = []
    kepsilons = []

    # Training loop
    tb = TimeBuffer(episodes)
    proj = 0

    memlock.release()
    # print(YELLOW + '[CHECK]' + RESET + ' MEMLOCK MUTEX RELEASED!')
    for e in range(episodes):
        # Get the first observation
        state1 = env1.reset()
        state2 = env2.reset()

        score1 = 0
        score2 = 0

        done1 = False
        done2 = False

        tb.start()

        for s in range(steps):
            # TODO: function
            r = np.random.rand()
            if r < eps:
                action1 = heurestic(state1)
                action2 = heurestic(state2)
                pass
            elif r < eps + keps:    # Teacher-student or peer-peer here
                if buf1.average() > buf2.average():
                    model = m1
                else:
                    model = m2

                Qvs1 = model(state1[np.newaxis])
                Qvs2 = model(state2[np.newaxis])

                action1 = np.argmax(Qvs1[0])
                action2 = np.argmax(Qvs2[0])
            else:
                Qvs1 = m1(state1[np.newaxis])
                Qvs2 = m2(state2[np.newaxis])

                action1 = np.argmax(Qvs1[0])
                action2 = np.argmax(Qvs2[0])

            # Apply the action and update the state (TODO: another function)
            if not done1:
                nstate1, reward1, done1, info = env1.step(action1)
                rbf1.append((state1, action1, reward1, nstate1, done1))
                state1 = nstate1
                score1 += reward1

            if not done2:
                nstate2, reward2, done2, info = env2.step(action2)
                rbf2.append((state2, action2, reward2, nstate2, done2))
                state2 = nstate2
                score2 += reward2

            if done1 and done2:
                break

        # Post episode routines
        tb.split()
        proj, fmt1, fmt2 = cmp_and_fmt(proj, tb.projected())

        # Post processing
        scores1.append(score1)
        scores2.append(score2)

        buf1.append(score1)
        buf2.append(score2)

        # Logging (TODO: remove all score logging after debugging)
        msg = f'{id:<50} Episode {e:<5} Agent #1: Score = {score1:<5} [{buf1.average():.2f}]' \
                f' Agent #2: Score {score2:<5} [{buf2.average():.2f}] Epsilon {eps:.2f}' \
                f' Time = [{str(tb):<20}], Projected'

        print(msg + f' [{fmt1}]')
        # if e % notify.interval == 1:
        #    notify.log(msg + f' [{fmt2}]')

        epsilons.append(eps)
        kepsilons.append(keps)

        eps = scheduler()
        keps = (1 - eps)/2

        # Train the model if the rbf is full enough
        if len(rbf1) >= batch_size:
            train(tf, m1, rbf1, batch_size, loss_ftn, optimizer, gamma, nouts)

        if len(rbf2) >= batch_size:
            train(tf, m2, rbf2, batch_size, loss_ftn, optimizer, gamma, nouts)

    # Bench loop (TODO: another function)
    finals1 = []
    finals2 = []

    for e in range(bench_episodes):
        # Get the first observation
        state1 = env1.reset()
        state2 = env2.reset()

        score1 = 0
        score2 = 0

        done1 = False
        done2  = False

        for s in range(steps):
            # Get the action
            Qvs1 = m1(state1[np.newaxis])
            Qvs2 = m2(state2[np.newaxis])

            action1 = np.argmax(Qvs1[0])
            action2 = np.argmax(Qvs2[0])

            # Apply the action and update the state
            if not done1:
                nstate1, reward1, done1, info = env1.step(action1)
                state1 = nstate1
                score1 += reward1

            if not done2:
                nstate2, reward2, done2, info = env2.step(action2)
                state2 = nstate2
                score2 += reward2

            if done1 and done2:
                break

        finals1.append(score1)
        finals2.append(score2)

    # Log completion
    msg = f'{id} finished in {fmt_time(time.time() - start)}'
    print(GREEN + msg + RESET)
    notify.notify(msg)

    # Record the results
    pdir = dirn + '/' + ename + '/TS_' + (heurestic.name + \
            '_and_' + scheduler.name).replace(' ', '_')
    os.system(f'mkdir -p {pdir}')
    fname = pdir + f'/Trial_{trial + 1}.csv'
    fout = open(fname, 'w')
    fout.write('Testing file...')
    fout.writelines([
        'Epsilons, ' + ','.join([str(e) for e in epsilons]) + '\n',
        'Kepsilons, ' + ','.join([str(e) for e in kepsilons]) + '\n',
        'Scores A1, ' + ','.join([str(s) for s in scores1]) + '\n',
        'Scores A2, ' + ','.join([str(s) for s in scores2]) + '\n',
        'Finals A1, ' + ','.join([str(f) for f in finals1]) + '\n',
        'Finals A2, ' + ','.join([str(f) for f in finals2])
    ])
    fout.close()

def run_heuristic_bench(ename, heurestic, episodes, steps, render = False, plot = False):
    env = gym.make(ename)

    finals = []
    for e in range(episodes):
        # Get the first observation
        state = env.reset()
        score = 0

        if render:
            env.render()

        for s in range(steps):
            # Get the action
            action = heurestic(state)

            # Apply the action and update the state
            nstate, reward, done, info = env.step(action)
            state = nstate
            score += reward

            if render:
                    env.render()

            if done:
                break

        finals.append(score)

        if render:
            input('Enter for next episode: ')

    print(f'Average score for {heurestic.name} out of {episodes}' \
            + f' episodes was {np.average(finals)}')

    if plot:
        plt.plot(range(1, episodes + 1), finals)
        plt.show()

    return finals

# Hard coded for now
class ActorCritic():
    # TODO: pass hidden layers only next, or better yet just manually program the network architectures
    def __init__(self, tf, nins, nouts):
        # Critic
        self.critic = tf.keras.models.Sequential([
            tf.keras.layers.Dense(32, activation='elu', input_shape=nins),
            tf.keras.layers.Dense(32, activation='elu'),
            tf.keras.layers.Dense(1)
        ])
        self.c_opt = tf.keras.optimizers.Adam(learning_rate=7e-3)

        # Actor
        self.actor = tf.keras.models.Sequential([
            tf.keras.layers.Dense(32, activation='elu', input_shape=nins),
            tf.keras.layers.Dense(32, activation='elu'),
            tf.keras.layers.Dense(nouts, activation='softmax')
        ])
        self.a_opt = tf.keras.optimizers.Adam(learning_rate=7e-3)

        # Confidence
        self.conf = tf.keras.models.Sequential([
            tf.keras.layers.Dense(32, activation='elu', input_shape=nins),
            tf.keras.layers.Dense(32, activation='elu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.conf_opt = tf.keras.optimizers.Adam(learning_rate=7e-3)

        # TODO: should go as part of the environment (config)
        self.gamma = 0.97

    def __call__(self, tf, state):
        prob = self.actor(np.array([state]))
        prob = prob.numpy()
        dist = tf.compat.v1.distributions.Categorical(logits=tf.math.log([prob]), dtype=tf.float32)
        action = dist.sample()
        return self.conf(np.array([state])), int(action.numpy()[0])

    def actor_loss(self, tf, probs, actions, td):
        probability = []
        log_probability = []
        for pb, a in zip(probs, actions):
          dist = tf.compat.v1.distributions.Categorical(logits=tf.math.log([pb]), dtype=tf.float32)
          log_prob = dist.log_prob(a)
          prob = dist.prob(a)
          probability.append(prob)
          log_probability.append(log_prob)

        p_loss = []
        e_loss = []
        td = td.numpy()
        for pb, t, lpb in zip(probability, td, log_probability):
            t =  tf.constant(t)
            policy_loss = tf.math.multiply(lpb,t)
            entropy_loss = tf.math.negative(tf.math.multiply(pb,lpb))
            p_loss.append(policy_loss)
            e_loss.append(entropy_loss)

        p_loss = tf.stack(p_loss)
        e_loss = tf.stack(e_loss)
        p_loss = tf.reduce_mean(p_loss)
        e_loss = tf.reduce_mean(e_loss)

        loss = -p_loss - 0.0001 * e_loss

        return loss

    def learn(self, tf, states, actions, discounted):
        discounted = tf.reshape(discounted, (len(discounted),))

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2, tf.GradientTape() as tape3:
            p = self.actor(states, training=True)
            v = self.critic(states, training=True)
            c = self.conf(states, training=True)

            v = tf.reshape(v, (len(v),))
            td = tf.math.subtract(discounted, v)

            kfactor = 2
            rc = tf.ones(c.shape) - tf.math.abs(tf.math.tanh(td/kfactor))

            a_loss = self.actor_loss(tf, p, actions, td)
            c_loss = 0.5 * tf.keras.losses.mean_squared_error(discounted, v)
            conf_loss = 0.5 * tf.keras.losses.mean_squared_error(rc, c)

        grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)
        grads3 = tape3.gradient(conf_loss, self.critic.trainable_variables)

        self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))
        self.conf_opt.apply_gradients(zip(grads3, self.conf.trainable_variables))

        return tf.math.reduce_mean(td), tf.math.reduce_mean(rc)

def preprocess(states, actions, rewards, gamma):
    discounted = []
    sum_reward = 0
    rewards.reverse()
    for r in rewards:
      sum_reward = r + gamma*sum_reward
      discounted.append(sum_reward)
    discounted.reverse()
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.int32)
    discounted = np.array(discounted, dtype=np.float32)

    return states, actions, discounted

# NOTE: we are not copying the scheduler
def run_policy_ret_modded(ename, heurestic, episodes, steps):
    # Import statements
    import tensorflow as tf

    from tensorflow import keras

    # Config GPUs (consume only half the memory -> put in a function)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # Global variables
    scores = []
    aconfs = []
    atds = []
    env = gym.make(ename)
    ac = ActorCritic(tf,
            env.observation_space.shape,
            env.action_space.n
    )

    for e in range(episodes):
        # Episode local variables
        done = False
        state = env.reset()
        score = 0
        rewards = []
        states = []
        actions = []
        confs = []

        for s in range(steps):
            # if np.random.rand() < eps:
            #     action = heurestic(state)
            # else:
            #    action = ac(tf, state)

            conf, action = ac(tf, state)
            # print('Confidence = ', conf)
            if conf < 0.5:
                action = heurestic(state)

            nstate, reward, done, _ = env.step(action)

            rewards.append(reward)
            states.append(state)
            actions.append(action)
            confs.append(conf)

            state = nstate
            score += reward

            if done:
                break

        states, actions, discounted = preprocess(states, actions, rewards, 1)
        tdm, rcm = ac.learn(tf, states, actions, discounted)

        scores.append(score)
        aconfs.append(np.mean(confs))
        atds.append(tdm)
        print(f'total reward after {e} episodes is {score}; average'
                f' confidence was {np.mean(confs)}, td was {tdm} rc was {rcm}')

    return scores, aconfs, atds
