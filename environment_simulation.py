import csv
import numpy as np

from threading import Thread

from policy_simulation import PolicySimulation
from scheduler import *
from colors import *

class EnvironmentSimulation:
    # For average window
    max_time_length = 25

    def __init__(self, ename, heurestics, schedulers, trials, episodes, steps, size, gamma = 0.95):
        self.ename = ename

        # Trial properties
        self.episodes = episodes
        self.steps = steps

        # Array of policies
        self.policies = []

        nhrs = len(heurestics)
        schs = len(schedulers)

        for i, j in np.ndindex(nhrs, schs):
            self.policies.append(PolicySimulation(ename, heurestics[i],
                schedulers[j], trials, size, gamma))

    def run(self):
        pool = []
        for policy in self.policies:
            pool.append(Thread(
                target = policy.run,
                args = (self.episodes, 50, self.steps, )
            ))
        
        # Launch the threads
        for thread in pool:
            thread.start()
        
        # Collect all the threads
        done = [False for i in range(len(pool))]
        while True:
            if all(d == True for d in done):
                break
            
            for i in range(len(done)):
                if (not done[i]) and (not pool[i].is_alive()):
                    pool[i].join()
                    done[i] = True

        # Display the total time to complete
        print(YELLOW + f'{self.ename} finished in [time]' + RESET)
