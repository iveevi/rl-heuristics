import sys
import gym
import numpy as np
import matplotlib.pyplot as plt

from collections import namedtuple

import runs

from heurestics import *
from schedulers import *
from score_buffer import *

class BoxEnv:
    act_space = namedtuple('action_space', 'n')
    obs_space = namedtuple('observation_space', 'shape')

    def __init__(self, dim, tgt=np.random.rand(2)):
        self.pos = None
        self.vel = None

        self.dim = dim
        self.tgt = np.array(tgt)

        self.action_space = self.act_space(5)
        self.observation_space = self.obs_space(self.tgt.shape)

    def __goal(self):
        return True if (abs(self.pos[0] - self.tgt[0]) <= 1.0) and \
            (abs(self.pos[1] - self.tgt[1]) <= 1.0) else False

    def reset(self):
        self.pos = np.random.rand(2)
        self.vel = np.zeros(2)
        return self.pos

    def step(self, move):
        self.pos += self.vel

        # Check 4 moves (0 for nothing is ignored, naturally)
        if move == 1: # Right
            self.vel[0] += 0.01
        elif move == 2: # Left
            self.vel[0] -= 0.01
        elif move == 3: # Up
            self.vel[1] += 0.01
        elif move == 4: # Down
            self.vel[1] -= 0.01

        # Check bound conditions (for each axis)
        npos = self.pos + self.vel
        if (npos[0] < 0.0) or (npos[0] > self.dim):
            self.vel[0] = 0.0

        if (npos[1] < 0.0) or (npos[1] > self.dim):
            self.vel[1] = 0.0

        return self.pos, (0 if self.__goal() else -1), \
            self.__goal(), None

env = BoxEnv(5, (3, 4))

episodes = 1000
score1, rho1 = runs.a2c(
        env,
        Heurestic('Great', mc_great),
        episodes,
        200,
        True
)

score = runs.a2c(
        env,
        Heurestic('Great', mc_great),
        episodes,
        500,
        False
)

# Plotting
eps = range(1, episodes + 1)

plt.plot(eps, score1, label='Rho True')
plt.plot(eps, rho1, label='Rho Reward')

plt.plot(eps, score, label='Regular')

plt.legend()
plt.show()
