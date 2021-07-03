import gym
import dqnagent
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from collections import deque

# CSV fields (for each field)
fields = [
    'Episode', 'Random HR Reward',
    'Bad HR Reward', 'Great HR Reward'
]

'''
Policy is a combination of function (hereustic/random)
and the scheduler
'''

# Epsilon greedy policy
def egreedy_policy(outputs, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(outputs)
    else:
        return -1

# Bad heurestic
def badhr_policy(outputs, state, epsilon):
    if np.random.rand() < epsilon:
        return 1
    else:
        return -1

# Best heurestic
def theta_omega_policy(outputs, state, epsilon):
    if np.random.rand() < epsilon:
        theta, w = state[2:4]
        if abs(theta) < 0.03:
            return 0 if w < 0 else 1
        else:
            return 0 if theta < 0 else 1
    else:
        return -1

# Filled with episode numbers at first
cartpole_rewards = [[i for i in range(1, 601)]]

def run_experiment(env_name, policy):
    env = gym.make(env_name)
    skeleton = models[env_name]

    agent = DQNAgent(skeleton, policy, env.action_space.n, 0.95)

    rewards = []
    epsilon = 1

    for episode in range(600):
        obs = env.reset()

        trew = 0
        for step in range(500):
            # TODO: use a scheduler later
            epsilon = max(1 - episode/500, 0.01)

            obs, reward, done, info = agent.step(env, obs, epsilon)
            trew += reward
            if done:
                break

        rewards.append(trew)
        if episode > 50:
            agent.train(50)

        print(f"\repisode: {episode}, reward: {trew}, eps: {epsilon:.3f}")

    # Display the graph of the results
    plt.figure(figsize = (8, 4))
    plt.plot(rewards)
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Sum of rewards", fontsize=14)
    plt.show()

    cartpole_rewards.append(rewards)

for pair in models:
    print("pair =", pair)
    run_experiment(pair, badhr_policy)
    run_experiment(pair, theta_omega_policy)
    run_experiment(pair, egreedy_policy)

    csv_file = open(pair, 'w')
    csv_writer = csv.writer(csv_file)

    csv_writer.writerow(fields)
    csv_writer.writerows(cartpole_rewards)
