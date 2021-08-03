import gym
import numpy as np
import tensorflow as tf

from tensorflow import keras

# Models
actor = keras.models.Sequential([
	keras.layers.Dense(64, activation='elu', input_shape=4),
	keras.layers.Dense(64, activation='elu'),
	keras.layers.Dense(2, activation='softmax')
])

critic = keras.models.Sequential([
	keras.layers.Dense(64, activation='elu', input_shape=4),
	keras.layers.Dense(32, activation='elu'),
	keras.layers.Dense(1, activation='linear')
])

# Environment
env = gym.make_env('CartPole-v1')

env.reset()
