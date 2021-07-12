import numpy as np

class Policy:
    def __init__(self, name, function):
        self.name = name
        self.function = function
    def __call__(self, state, epsilon):
        if np.random.rand() < epsilon:
            return self.function(state)

        return -1
