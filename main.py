import gym

import numpy as np
import matplotlib.pyplot as plt

from copy import copy
from collections import deque
from pathos.multiprocessing import ProcessPool

import notify

from scheduler import *
from heurestics import *
from upload import *
from time_buffer import *

# Global variables
bench_episodes = 10

# Setup of environments: use YAML maybe
environments = {
    'CartPole-v1': {
        'skeleton': [[4], 2, [32, 'elu'], [32, 'elu']],
        'heurestics': [
            Heurestic('Egreedy', egreedy),
            Heurestic('Great HR', theta_omega)
        ],
        'schedulers': [
            LinearDecay(400)
        ],
        'trials': 2,
        'episodes': 10,
        'steps': 500
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

# If a skeleton becomes necessary use a list of some named tuples, or smthing
def run(ename, skeleton, heurestic, schedref, trial, episodes, steps):
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
    scheduler = copy(schedref)

    gamma = 0.95                # Should be customizable
    loss_ftn = keras.losses.mean_squared_error
    optimizer = keras.optimizers.Adam(learning_rate = 1e-2)
    nouts = env.action_space.n  # Should be customizable

    start = time.time()

    scores = []
    epsilons = []
    
    id = ename + ': ' + heurestic.name + ' and ' + scheduler.name + ': ' + str(trial)

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
        msg = f'{id}, Episode {e}, Score = {score}, Epsilon {eps},' \
                f' Time = [{tb}], Projected'
        
        print(msg + f' [{fmt1}]')
        if e % notify.interval == 1:
            notify.log(msg + f' [{fmt2}]')

        # Post processing
        scores.append(score)
        epsilons.append(eps)
        eps = scheduler()

        # Train the model if the rbf is full enough
        if len(rbf) >= batch_size:
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

    return scores, epsilons, finals

def write_data(fpath, rets, episodes):
    fout = open(fpath, 'w')

    fout.write(', '.join(['Episodes'] + [str(i) for i in range(1, episodes + 1)]) + '\n')
    fout.write(', '.join(['Epsilon'] + [str(i) for i in rets[0][1]]) + '\n')

    i = 1
    for scores, epsilon, finals in rets:
        fout.write(', '.join([f'Trial #{i}'] + [str(s) for s in scores]) + '\n')
        i += 1

    fout.write(', '.join(['Bench Episodes'] + [str(i) for i in range(1, bench_episodes + 1)]) + '\n')

    i = 1
    for scores, epsilon, finals in rets:
        fout.write(', '.join([f'Bench Trial #{i}'] + [str(s) for s in finals]) + '\n')
        i += 1

enames = []
skeletons = []
hrs = []
scs = []
trialns = []
episodes = []
steps = []

for env in environments:
    ecp = environments[env]
    heurestics = ecp['heurestics']
    schedulers = ecp['schedulers']
    trials = ecp['trials']

    for hr in heurestics:
        for sc in schedulers:
            for i in range(trials):
                enames.append(env)
                skeletons.append(ecp['skeleton'])
                hrs.append(hr)
                scs.append(sc)
                trialns.append(i + 1)
                episodes.append(ecp['episodes'])
                steps.append(ecp['steps'])

# Launch the processes
start = time.time()

pool = ProcessPool(len(enames))
rets = pool.map(run, enames, skeletons, hrs, scs, trialns, episodes, steps)

# Write the data
dir = setup(environments)
index = 0
for env in environments:
    trials = environments[env]['trials']
    for hr in heurestics:
        for sc in schedulers:
            fpath = dir + '/' + env + '/' + (hr.name + '_and_' + sc.name).replace(' ', '_') + '.csv'
            write_data(fpath, rets[index : index + trials], environments[env]['episodes'])
            index += trials

# Upload
upload(dir, sudo = False)

# Log completion
msg = f'Completed all simulations in {fmt_time(time.time() - start)}, see `{dir}`'
print(msg)
notify.notify(msg)