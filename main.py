import numpy as np

from environment_simulation import EnvironmentSimulation
from policy import Heurestic
from scheduler import *
from upload import setup, upload

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

# Use for testing only
etest = EnvironmentSimulation(
        'CartPole-v1',
        [
            Heurestic('Great HR', theta_omega)
        ],
        [
            LinearDecay(400, 50),
            DampedOscillator(400, 50)
        ],
        1,
        100,
        500,
        32  # TODO: add bench episodes as a parameter
)

'''etest = EnvironmentSimulation(
        'CartPole-v1',
        [
            Heurestic('Epsilon Greedy', egreedy),
            Heurestic('Bad HR', badhr),
            Heurestic('Great HR', theta_omega)
        ],
        [
            LinearDecay(400),
            DampedOscillator(400)
        ],
        2,
        2,
        50,
        32  # TODO: add bench episodes as a parameter
)'''

setup([etest])
etest.run()

# Prompt for upload
upload()
