import gym
import os
import sys
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

def heuristic_bench(ename, heurestic, episodes, steps, render = False, plot = False):
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
    # TODO: steps -> max steps should grow
    def __init__(self, tf, nins, nouts, steps):
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

        # TODO: should go as part of the environment (config)
        self.gamma = 0.97

        # Pseudo rewards
        self.rho = tf.keras.models.Sequential([
            tf.keras.layers.Dense(3, activation='elu', input_shape=nins),
            # tf.keras.layers.Dense(128, activation='elu'),
            tf.keras.layers.Dense(3, activation='elu'),
            tf.keras.layers.Dense(1, activation=tf.keras.layers.LeakyReLU(alpha=0.01))
        ])
        self.rho_opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

        # Maximal trajectory
        self.max_trajectory = None
        self.max_reward = -np.Inf

    def __call__(self, tf, state):
        prob = self.actor(np.array([state]))
        prob = prob.numpy()
        dist = tf.compat.v1.distributions.Categorical(logits=tf.math.log([prob]), dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])

    def reward(self, state):
        #raw = self.rho(state[np.newaxis]).numpy()[0, 0]
        #return raw
        return -math.dist(state, (2.5, 2.5))

    def learn_maximal_rho(self, tf):
        # print('learning maximal, with reward = ', self.max_reward)
        with tf.GradientTape() as tape:
            rhos = self.rho(self.max_trajectory, training=True)
            new_rhos = tf.divide(tf.fill(rhos.shape, 1.1), rhos)
            #scales = tf.pow(tf.constant(0.985),
            #        tf.range(len(self.max_trajectory[0]), 0.0, -1.0))
            #new_rhos = tf.multiply(new_rhos, scales)
            rho_loss = 0.5 * tf.keras.losses.mean_squared_error(rhos, new_rhos)

        grads = tape.gradient(rho_loss, self.rho.trainable_variables)
        self.rho_opt.apply_gradients(zip(grads, self.rho.trainable_variables))

        # again_rhos = self.rho(self.max_trajectory)
        # print('old, expected, new = ', tf.concat([rhos, new_rhos, again_rhos], axis=1))

    def learn_conf(self, tf, states, calibrate=False):
        with tf.GradientTape() as tape:
            confs = self.conf(states, training=True)
            if calibrate:
                act_confs = tf.zeros(confs.shape)
            else:
                act_confs = tf.ones(confs.shape)
            conf_loss = 0.5 * tf.keras.losses.mean_squared_error(act_confs,
                    confs) * 1.0 # some factor to reduce growth

        grads = tape.gradient(conf_loss, self.conf.trainable_variables)
        self.conf_opt.apply_gradients(zip(grads, self.conf.trainable_variables))

    def actor_loss(self, tf, probs, actions, discounted, td):
        probability = []
        log_probability = []

        for pb, a, v in zip(probs, actions, discounted):
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

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.actor(states, training=True)
            v = self.critic(states, training=True)

            v = tf.reshape(v, (len(v),))
            td = tf.math.subtract(discounted, v)

            a_loss = self.actor_loss(tf, p, actions, discounted, td)
            c_loss = 0.5 * tf.keras.losses.mean_squared_error(discounted, v)

        grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)

        self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))

        return tf.math.reduce_mean(td)

def preprocess(states, actions, rewards, gamma):
    discounted = []
    sum_reward = 0
    rewards.reverse()
    for r in rewards:
      sum_reward = r + gamma * sum_reward
      discounted.append(sum_reward)
    discounted.reverse()
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.int32)
    discounted = np.array(discounted, dtype=np.float32)

    return states, actions, discounted

# NOTE: we are not copying the scheduler
def a2c(env, heurestic, episodes, steps, rho, gamma=0.99):
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
    tscores = []
    # env = gym.make(ename)
    ac = ActorCritic(tf,
            env.observation_space.shape,
            env.action_space.n,
            steps
    )
    sched = LinearDecay(800, 50)
    eps = sched()

    # Initial rho rewards
    x = np.linspace(0.0, env.dim)
    y = np.linspace(0.0, env.dim)

    inits = [[ac.reward(np.array([xi, yi])) for xi in x] for yi in y]

    # Initialize dummy true score
    true = -1
    for e in range(episodes):
        # Episode local variables
        done = False
        state = env.reset()
        score = 0
        tscore = 0
        rewards = []
        trues = []
        states = []
        actions = []

        for s in range(steps):
            action = ac(tf, state)
            nstate, true, done, _ = env.step(action)

            if rho:
                reward = ac.reward(nstate)
            else:
                reward = true

            rewards.append(reward)
            states.append(copy(state))
            actions.append(action)
            trues.append(true)

            state = nstate
            score += reward
            tscore += true

            if done:
                break

        states, actions, discounted = preprocess(states, actions, rewards, gamma)
        ac.learn(tf, states, actions, discounted)

        eps = sched()

        # Change after the preprocessing
        if tscore > ac.max_reward:
            ac.max_trajectory = states
            ac.max_reward = tscore

        if rho:
            ac.learn_maximal_rho(tf)
            print('Learning optimal trajectory...')

        scores.append(score)
        tscores.append(tscore)
        print(f'episode {e + 1}, score {score}, true score {tscore}')

    # Summary
    score = 0
    for state in ac.max_trajectory:
        score += ac.reward(state)
    print('Summary: score for max trajectory is', score)

    if rho:
        return tscores, scores, ac, inits
    else:
        return scores
