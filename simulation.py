import gym

from copy import copy

from dqnagent import DQNAgent
from models import models
from policy import Policy
from scheduler import *

class Simulation:
    '''
    @param ename the name of environment in the gym registry
    @param trial the trial number
    @param batch_size the batch sizes
    '''
    def __init__(self, ename, trial, heurestic, scheduler, batch_size, gamma):
        # Assigned attributes
        self.ename = ename
        self.trial = trial
        self.batch_size = batch_size

        # TODO: this is useless
        self.pname = '(' + heurestic.name + ' & ' + scheduler.name + ')'

        self.obs = None
        self.score = 0
        self.episode = 0
        self.epsilon = 1

        self.rewards = []
        
        self.done = False
        self.env = gym.make(ename)
        self.scheduler = copy(scheduler)
        self.agent = DQNAgent(
                models[ename],
                Policy (self.pname, heurestic),
                self.env.action_space.n,
                gamma
        )

    def reset(self):
        self.obs = self.env.reset()

        self.rewards.append((self.score, self.epsilon))
        self.score = 0
        self.done = False
        self.episode += 1
        self.epsilon = self.scheduler()

        # Train the model if enough elements
        if len(self.agent.rbf) >= self.batch_size:
            self.agent.train(self.batch_size)

    def run(self):
        # Do nothing if already done
        if self.done:
            return True

        self.obs, reward, self.done = self.agent.step(self.env, self.obs, self.epsilon)
        self.score += reward

        return self.done

    def run_trial(self, episodes, steps):
        for episode in range(episodes):
            self.reset()
            for step in range(steps):
                done = self.run()

                if done:
                    break
