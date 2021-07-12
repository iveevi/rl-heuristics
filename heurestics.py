import numpy as np

class Heurestic:
    def __init__(self, name, function):
        self.name = name
        self.function = function
    def __call__(self, x):
        return self.function(x)

# Epsilon greedy policy
def egreedy(state):
    return np.random.randint(1)

# Bad heurestic
def badhr(state):
    return 1

# Best heurestic
def theta_omega(state):
    theta, w = state[2:4]
    if abs(theta) < 0.03:
        return 0 if w < 0 else 1
    else:
        return 0 if theta < 0 else 1
