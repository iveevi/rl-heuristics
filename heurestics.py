import random
import numpy as np

class Heurestic:
    def __init__(self, name, function):
        self.name = name
        self.function = function
    def __call__(self, x):
        return self.function(x)

# Epsilon greedy policy
def egreedy_ftn(nouts):
    return lambda s: np.random.randint(nouts)

egreedy_cp = egreedy_ftn(2)
egreedy_mc = egreedy_ftn(3)

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

# Mixed
def hybrid(state):
    if random.uniform(0, 1) > 0.5:
        return egreedy(state)
    else:
        return theta_omega(state)
