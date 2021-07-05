import gym

from copy import copy

from dqnagent import DQNAgent
from models import models
from policy import Policy
from scheduler import *

class MatEntry:
    '''
    @param ename the name of environment in the gym registry
    @param size the batch sizes
    '''
    def __init__(self, ename, heurestic, scheduler, size, gamma):
        # TODO: organize these attributes
        self.obs = None
        self.score = 0
        self.episode = 0
        self.epsilon = 1
        self.size = size
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

        # Train the model if enough elements
        if len(self.agent.rbf) >= self.size:
            self.agent.train(self.size)

    def run(self):
        # Do nothing if already done
        if self.done:
            return True

        self.obs, reward, self.done = self.agent.step(self.env, self.obs, self.epsilon)
        self.score += reward

        return self.done

    def run_episode(self, steps):
        self.reset()
        for step in range(steps):
            done = self.run()

            if done:
                break
