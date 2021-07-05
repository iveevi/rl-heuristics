import numpy as np

from envtest import EnvTest
from policy import Heurestic
from models import models
from scheduler import *

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

etest = EnvTest(
        'CartPole-v1',
        [
            Heurestic('Epsilon Greedy', egreedy),
            Heurestic('Bad HR', badhr),
            Heurestic('Great HR', theta_omega)
        ],
        [
            LinearDecay(400),
            Scheduler('Zero', lambda e: 0)
        ],
        600,
        500,
        32
)

etest.run()
