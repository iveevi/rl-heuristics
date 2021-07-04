import gym
import csv
import time

import numpy as np

from copy import copy

from dqnagent import DQNAgent
from models import models
from policy import Heurestic, Policy
from scheduler import *

# For average window
max_time_length = 25
        
class MatEntry:
    '''
    @param ename the name of environment in the gym registry
    '''
    def __init__(self, ename, heurestic, scheduler, gamma):
        self.obs = None
        self.score = 0
        self.episode = 0
        self.epsilon = 1
        self.total_rewards = []
        self.done = False
        self.env = gym.make(ename)
        self.ename = ename
        self.pname = '(' + heurestic.name + ' & ' + scheduler.name + ')'
        self.scheduler = copy(scheduler)
        self.agent = DQNAgent(
                models[ename],
                Policy (self.pname, heurestic),
                self.env.action_space.n,
                gamma
        )

        # self.env.render()

    def reset(self):
        self.obs = self.env.reset()
        # TODO: dont need this here
        self.total_rewards.append((self.score, self.epsilon))
        self.score = 0
        self.done = False
        self.episode += 1
        self.epsilon = self.scheduler()

    def run(self):
        # Do nothing if already done
        if self.done:
            return True

        self.obs, reward, self.done = self.agent.step(
                self.env, self.obs, self.epsilon)
        self.score += reward

        return self.done

class EnvTest:
    def __init__(self, ename, heurestics, schedulers, episodes, steps, gamma = 0.95):
        self.heurestics = heurestics # TODO: no need to store
        self.schedulers = schedulers # TODO: also no need to store
        self.ename = ename

        # Timing
        self.time = time.time()
        self.times = []
        self.left = 0

        # Trial properties
        self.episodes = episodes
        self.steps = steps

        # Dimensions of the testing matrix
        self.nhrs = len(heurestics)
        self.schs = len(schedulers)

        # Actual matrix
        self.matrix = [
            [
                MatEntry(
                    ename,
                    heurestics[i],
                    schedulers[j],
                    gamma
                ) for j in range(self.schs)
            ] for i in range(self.nhrs)
        ]
    
    def run_step(self):
        done = True

        for i in range(self.nhrs):
            for j in range(self.schs):
                done &= self.matrix[i][j].run()
        
        return done

    def write_episode(self, csv, episode):
        fields = [episode]
        for i in range(self.nhrs):
            for j in range(self.schs):
                fields.append(self.matrix[i][j].score)
                fields.append(self.matrix[i][j].epsilon)
        csv.writerow(fields)

        # Get time
        end = time.time()

        self.times.append(end - self.time)
        if len(self.times) > max_time_length:
            del self.times[0]

        average = np.sum(self.times) / len(self.times)

        # In minutes
        left = (average * (self.episodes - episode + 1))/60

        diff = left - self.left
        out = f'+{diff:.3f}' if diff > 0 else f'{diff:.3f}'

        # TODO: use mutex here to clean output
        print(f'{self.ename} completed episode #{episode}'
                + f' in {end - self.time:.3f} secs ->'
                + f' average = {average:.3f} secs ->'
                + f' projected time left = {left:.3f} mins [{out}]')

        # Update times
        self.time = end
        self.left = left

    def run(self):
        # Setup csv
        csv_file = open(self.ename + '_results.csv', 'w')
        csv_writer = csv.writer(csv_file)

        # Gather the fields
        fields = ['Episode']
        for i in range(self.nhrs):
            for j in range(self.schs):
                name = self.matrix[i][j].pname
                fields.append(name + '.score')
                fields.append(name + '.epsilon')

        csv_writer.writerow(fields)

        # Set the times
        self.time = time.time()
        start = time.time()

        # Run the simulations
        for eps in range(self.episodes):

            # Reset the observations
            for i in range(self.nhrs):
                for j in range(self.schs):
                    self.matrix[i][j].reset()
            
            for step in range(self.steps):
                done = self.run_step()

                if done:
                    break

            self.write_episode(csv_writer, eps + 1)

        # Report end and total time
        end = time.time()
        total = (end - start)/60

        print(f'Simulation {self.ename} finished in '
            + f'{total:.3f} mins.')

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
            LinearDecay(400)
        ],
        600,
        500
)

etest.run()
