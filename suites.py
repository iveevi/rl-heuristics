import tensorflow as tf
import numpy as np
import gym

from tensorflow import keras
from collections import deque

models = {
    'CartPole-v1': keras.models.Sequential([
        keras.layers.Dense(32, activation="elu", input_shape=[4]),
        keras.layers.Dense(32, activation="elu"),
        keras.layers.Dense(2)
    ])
}
