import gym

import numpy as np
import tensorflow as tf

from tensorflow import keras
from collections import deque

from scheduler import *
from heurestics import *

models = {
    # Taken from hands-on-machine learning
    'CartPole-v1': keras.models.Sequential([
        keras.layers.Dense(32, activation="elu", input_shape=[4]),
        keras.layers.Dense(32, activation="elu"),
        keras.layers.Dense(2)
    ])
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

def run(skeleton, ename, heurestic, scheduler, episodes, steps):
    # Setup the model and environment
    model = keras.models.clone_model(skeleton)
    model.set_weights(skeleton.get_weights())

    # Other variables
    env = gym.make(ename)
    eps = 1
    rbf = deque(maxlen = 2000)  # Should be customizable
    batch_size = 32             # Should be customizable

    gamma = 0.95                # Should be customizable
    loss = keras.losses.mean_squared_error
    optimizer = keras.optimizers.Adam(learning_rate = 1e-2)
    nouts = 1                   # Should be customizable

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
        eps = scheduler()
        print('Score = ', score)

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
                loss = tf.reduce_mean(loss(tgt_Qvs, Qvs))

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

run(models['CartPole-v1'], 'CartPole-v1', Heurestic('Great HR', theta_omega), LinearDecay(400), 10, 500)