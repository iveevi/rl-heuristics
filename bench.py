import runs
import notify

from heurestics import *
from schedulers import *

import matplotlib.pyplot as plt

notify.su_off = True

'''
scores = runs.run_policy_ret(
        'CartPole-v1',
        [[4], 2, [32, 'elu'], [32, 'elu']],
        Heurestic('Great', theta_omega),
        LinearDecay(800, 50),
        1,
        1000,
        500
)
'''

'''
scores = runs.run_policy_ret(
        'MountainCar-v0',
        [[2], 3, [32, 'elu'], [32, 'elu']],
        Heurestic('Random', egreedy_mc),
        LinearDecay(800, 50),
        1,
        1000,
        200
)

plt.plot(range(1, 1001), scores)
plt.show()
'''

def mc_great(state):
    pos = state[0] + 0.5
    vel = state[1]

    if pos > 0:
        if vel > 0:
            return 2
        else:
            return 0
    else:
        if vel < 0:
            return 0
        else:
            return 2

runs.run_heuristic_bench('CartPole-v1', Heurestic('Great', theta_omega), 100, 500)
runs.run_heuristic_bench('CartPole-v1', Heurestic('Random', egreedy_cp), 100, 500)
runs.run_heuristic_bench('MountainCar-v0', Heurestic('Random', egreedy_mc), 100, 200)
runs.run_heuristic_bench('MountainCar-v0', Heurestic('Great', mc_great), 100, 200)
