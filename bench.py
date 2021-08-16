import sys
import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from collections import namedtuple

import runs

from heurestics import *
from schedulers import *
from score_buffer import *

# Put in another file
class BoxEnv:
    act_space = namedtuple('action_space', 'n')
    obs_space = namedtuple('observation_space', 'shape')
    radius = 1.0

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

tgt = (2.5, 2.5)
env = BoxEnv(5, tgt)
episodes = 100
steps = 200

score1, rho1, ac, inits = runs.a2c(
        env,
        Heurestic('Great', mc_great),
        episodes,
        steps,
        True
)

'''
score = runs.a2c(
        env,
        Heurestic('Great', mc_great),
        episodes,
        steps,
        False
)
'''

# Plotting helper function
def smooth(x, w):
    xp = []
    sb = ScoreBuffer(w)
    for xi in x:
        sb.append(xi)
        xp.append(sb.average())
    return xp

# Plotting
eps = range(1, episodes + 1)
x = np.linspace(0.0, env.dim)
y = np.linspace(0.0, env.dim)

fins = [[ac.reward(np.array([xi, yi])) for xi in x] for yi in y]
fqvs = [[ac.critic(np.array([xi, yi])[None])[0, 0] for xi in x] for yi in y]

fig, (main, ax1, ax2, ax3) = plt.subplots(4)

main.plot(eps, smooth(score1, 10), label='Rho True')
main.plot(eps, smooth(rho1, 10), label='Rho Reward')
# main.plot(eps, smooth(score, 10), label='Regular')

# Plot the maximal (mu) trajectory
(mu_xs, mu_ys, xs, ys) = ([], [], [], [])
for state in ac.max_trajectory:
    mu_xs.append(state[0])
    mu_ys.append(state[1])

state = env.reset()
for _ in range(steps):
    act = ac(tf, state)
    nstate, true, done, _ = env.step(act)
    xs.append(state[0])
    ys.append(state[1])
    state = nstate

    if done:
        break

pt = (tgt[0] - 1, tgt[1] - 1)
rad = 2 * BoxEnv.radius

ax2.plot(xs, ys, color='grey')
ax2.plot(mu_xs, mu_ys, color='white')
ax2.add_patch(patches.Rectangle(pt, rad, rad, alpha=0.5, color='grey'))

cf1 = ax1.contourf(x, y, inits, 100)
cf2 = ax2.contourf(x, y, fins, 100)
cf3 = ax3.contourf(x, y, fqvs, 100)

fig.colorbar(cf1, ax=ax1)
fig.colorbar(cf2, ax=ax2)
fig.colorbar(cf3, ax=ax3)

fig.tight_layout()
fig.set_size_inches(8, 10)

main.legend()
plt.show()
