import gym

import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from pathos.multiprocessing import ProcessPool

from scheduler import *
from heurestics import *

'''
'''

'''
# Model skeletons
models = {
    # Taken from hands-on-machine learning
    'CartPole-v1': keras.models.Sequential([
        keras.layers.Dense(32, activation="elu", input_shape=[4]),
        keras.layers.Dense(32, activation="elu"),
        keras.layers.Dense(2)
    ])
}'''

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
def run(ename, heurestic, scheduler, episodes, steps):
    import tensorflow as tf
    from tensorflow import keras

    # Config GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    model = keras.models.Sequential([
        keras.layers.Dense(32, activation="elu", input_shape=[4]),
        keras.layers.Dense(32, activation="elu"),
        keras.layers.Dense(2)
    ])

    # Setup the model and environment
    # model = keras.models.load_model(path)
    # model = keras.models.clone_model(skeleton)
    # model.set_weights(skeleton.get_weights())

    # Other variables
    env = gym.make(ename)
    eps = 1
    rbf = deque(maxlen = 2000)  # Should be customizable
    batch_size = 32             # Should be customizable

    gamma = 0.95                # Should be customizable
    loss_ftn = keras.losses.mean_squared_error
    optimizer = keras.optimizers.Adam(learning_rate = 1e-2)
    nouts = env.action_space.n  # Should be customizable

    scores = []

    # Training loop
    for e in range(episodes):
        # Get the first observation
        state = env.reset()
        score = 0

        for s in range(steps):
            # Get the action
            if np.random.rand() < eps:
                action = heurestic(state)
            else:
                Qvs = model(state[np.newaxis])
                # print the Q_values
                action = np.argmax(Qvs[0])

            # Apply the action and update the state
            nstate, reward, done, info = env.step(action)
            rbf.append((state, action, reward, nstate, done))
            state = nstate
            score += reward

            if done:
                break

        # Progress the scheduler
        print(f'Episode {e}, Score = {score}, Epsilon {eps}')
        scores.append(score)
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

    return scores

# run(models['CartPole-v1'], 'CartPole-v1', Heurestic('Egreedy', theta_omega),
#        DampedOscillator(400, 50), 1000, 500)
# models['CartPole-v1'].save('cv1.h5')

envs = ['CartPole-v1', 'CartPole-v1']
hrs = [Heurestic('Great HR', theta_omega), Heurestic('Egreedy', egreedy)]
schs = [DampedOscillator(400, 50), DampedOscillator(400, 50)]
episodes = [1000, 1000]
steps = [500, 500]

pool = ProcessPool(2)
results = pool.map(run, envs, hrs, schs, episodes, steps)

print('results: ', results)
