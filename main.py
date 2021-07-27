import gym

import numpy as np
import matplotlib.pyplot as plt
# import nvidia_smi

from copy import copy
from collections import deque
# from pathos.multiprocessing import ProcessPool # Do we even need this
from multiprocessing import Lock, Process, Queue
# from threading import Lock

import notify

from scheduler import *
from heurestics import *
from upload import *
from time_buffer import *
from score_buffer import *

# Global variables
bench_episodes = 50
rets = Queue()
memlock = Lock()

# Setup of environments: use YAML maybe
environments = {
    'CartPole-v1': {
        'skeleton': [[4], 2, [32, 'elu'], [32, 'elu']],
        'heurestics': [
            Heurestic('Egreedy', egreedy),
            Heurestic('Great HR', theta_omega),
            Heurestic('Bad HR', badhr)
        ],
        'schedulers': [
            LinearDecay(800, 50),
            DampedOscillator(800, 50)
        ],
        'trials': 10,
        'episodes': 15,
        'steps': 500,
        'ts-tutoring': False
    }
}

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

def run_policy(ename, skeleton, heurestic, schedref, trial, episodes, steps):
    # Assume only using GPU 0 (for my system)
    import nvidia_smi

    # Named process id
    scheduler = copy(schedref)
    id = ename + ': ' + heurestic.name + ' and ' + scheduler.name + ': ' + str(trial)

    # TODO: put in function
    i = 0
    while True:
        memlock.acquire()

        nvidia_smi.nvmlInit()

        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

        # info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print(YELLOW + '[CHECK]' + RESET + f' {id}, L{i}: Available GPU memory {info.free/info.total:%}')
        if info.free/info.total < 0.5:
            print(RED + 'Reached 50% threshold, waiting (5s)...' + RESET)
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
    print(YELLOW + '[CHECK]' + RESET + ' MEMLOCK MUTEX RELEASED!')
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
    print(msg)
    notify.notify(msg)

    # Record the results
    fname = ename + '/' + (heurestic.name + '_and_' + scheduler.name).replace(' ', '_') + '.csv'
    rets.put((fname, (scores, epsilons, np.average(finals))))

def run_tutoring(ename, skeleton, heurestic, schedref, trial, episodes, steps):
    # Assume only using GPU 0 (for my system)
    import nvidia_smi

    # Named process id
    scheduler = copy(schedref)
    id = ename + ': Tutoring (TS): ' + str(trial)

    # TODO: put in function
    i = 0
    while True:
        memlock.acquire()

        nvidia_smi.nvmlInit()

        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

        # info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print(YELLOW + '[CHECK]' + RESET + f' {id}, L{i}: Available GPU memory {info.free/info.total:%}')
        if info.free/info.total < 0.5:
            print(RED + 'Reached 50% threshold, waiting (5s)...' + RESET)
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

    id = ename + ': Tutoring with ' + scheduler.name + ': ' + str(trial)

    # Training loop
    tb = TimeBuffer(episodes)
    proj = 0

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
    print(msg)
    # notify.notify(msg)

    # Record the results
    fname = env + '/' + (heurestic.name + '_and_' + scheduler.name).replace(' ', '_') + '.csv'
    rets.put((fname, (scores1, scores2, epsilons, kepsilons, np.average(finals1),
                np.average(finals2))))

# TODO: change tutoring to an integer (to diff between teacher-student and peer-peer)
def run(ename, skeleton, heurestic, schedref, trial, episodes, steps, tutoring):
    if tutoring:
        return run_tutoring(ename, skeleton, heurestic, schedref, trial,
                episodes, steps)
    else:
        return run_policy(ename, skeleton, heurestic, schedref, trial, episodes,
                steps)

def write_data(fpath, rets, episodes):
    fout = open(fpath, 'w')

    fout.write(', '.join(['Episodes'] + [str(i) for i in range(1, episodes + 1)]) + '\n')
    fout.write(', '.join(['Epsilon'] + [str(i) for i in rets[0][1]]) + '\n')

    i = 1
    for scores, epsilon, finals in rets:
        fout.write(', '.join([f'Trial #{i}'] + [str(s) for s in scores]) + '\n')
        i += 1

    i = 1
    for scores, epsilon, finals in rets:
        fout.write(f'Bench Trial #{i}, {finals}\n')
        i += 1

def write_tutoring_data(fpath, rets, episodes):
    fout = open(fpath, 'w')

    fout.write(', '.join(['Episodes'] + [str(i) for i in range(1, episodes + 1)]) + '\n')
    fout.write(', '.join(['Epsilon'] + [str(i) for i in rets[0][1]]) + '\n')

    i = 1
    for scores1, scores2, epsilon, keps, finals1, finals2 in rets:
        fout.write(', '.join([f'Agent (1) #{i}'] + [str(s) for s in scores1]) + '\n')
        fout.write(', '.join([f'Agent (2) #{i}'] + [str(s) for s in scores2]) + '\n')
        i += 1

    i = 1
    for scores1, scores2, epsilon, keps, finals1, finals2 in rets:
        fout.write(f'Bench (1) #{i}, {finals1}\n')
        fout.write(f'Bench (1) #{i}, {finals2}\n')
        i += 1

# Load up the processes
pool = []

for env in environments:
    ecp = environments[env]
    heurestics = ecp['heurestics']
    schedulers = ecp['schedulers']
    trials = ecp['trials']

    for hr in heurestics:
        for sc in schedulers:
            for i in range(trials):
                pool.append(Process(target = run,
                    args = (env, ecp['skeleton'], hr,
                    sc, i + 1, ecp['episodes'], ecp['steps'],
                    False, )))

# Launch the processes
start = time.time()
notify.su_off = True

for proc in pool:
    proc.start()

while len(pool) > 0:
    for i in range(len(pool)):
        if not pool[i].is_alive():
            # TODO: still need to retrieve results
            pool[i].join()

            print(GREEN + 'Another join!' + RESET)

            del pool[i]
            break

# Collect and sort the results
files = dict()
print('results (rets):')
while not rets.empty():
    fname, vs = rets.get()

    if fname in files:
        files[fname].append(vs)
    else:
        files[fname] = [vs]
    print('\t' + str(rets.get()))

# Write data
dir = setup(environments)
for file in files:
    trials = files[file]
    i = file.find('/')
    ename = file[:i]

    # Regular policy trials
    if len(trials[0]) == 3:
        write_data(dir + '/' + file, trials, environments[ename]['episodes'])
    # Tutoring policy trials
    else:
        write_tutoring_data(dir + '/' + file, trials, environments[ename]['episodes'])

# Upload data
upload(dir)

# Log completion
msg = f'Completed all simulations in {fmt_time(time.time() - start)}, see `{dir}`'
print(msg)
notify.notify(msg)
