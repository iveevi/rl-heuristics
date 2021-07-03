import tensorflow as tf

from tensorflow import keras

# Models for each environment
models = {
    'CartPole-v1': keras.models.Sequential([
        keras.layers.Dense(32, activation="elu", input_shape=[4]),
        keras.layers.Dense(32, activation="elu"),
        keras.layers.Dense(2)
    ])
}