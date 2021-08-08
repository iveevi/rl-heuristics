import gym

import runs
import notify

from heurestics import *
from schedulers import *
from score_buffer import *

import matplotlib.pyplot as plt

notify.su_off = True

episodes = 1000
scores, confs, tds = runs.run_policy_ret_modded(
        'MountainCar-v0',
        Heurestic('Great', mc_great),
        episodes,
        200
)

'''
scores = runs.run_policy_ret(
        'MountainCar-v0',
        [[2], 3, [32, 'elu'], [32, 'elu']],
        Heurestic('Great', mc_great),
        DampedOscillator(800, 50),
        1,
        1000,
        200
)
'''

def smooth(x, w):
    xp = []
    sb = ScoreBuffer(w)
    for xi in x:
        sb.append(xi)
        xp.append(sb.average())
    return xp

# Plot
fig, (ax1, ax2, ax3) = plt.subplots(3)
eps = range(1, episodes + 1)
ax1.plot(eps, smooth(scores, 100))
ax2.plot(eps, smooth(confs, 5))
ax3.plot(eps, smooth(tds, 5))
plt.show()

'''
runs.run_heuristic_bench('CartPole-v1', Heurestic('Great', theta_omega), 100, 500)
runs.run_heuristic_bench('CartPole-v1', Heurestic('Random', egreedy_cp), 100, 500)
runs.run_heuristic_bench('MountainCar-v0', Heurestic('Random', egreedy_mc), 100, 200)
runs.run_heuristic_bench('MountainCar-v0', Heurestic('Great', mc_great), 100, 200)
'''
