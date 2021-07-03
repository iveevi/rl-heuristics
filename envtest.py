import gym

from collections import namedtuple

from dqnagent import DQNAgent
from models import models
from policy import Policy

class EnvTest:
    def __init__(self, ename, heurestics, schedulers, episodes, steps, gamma = 0.95):
        self.heurestics = heurestics
        self.schedulers = schedulers
        self.ename = ename

        # Trial properties
        self.episodes = episodes
        self.steps = steps

        # Dimensions of the testing matrix
        self.nhrs = len(heurestics)
        self.schs = len(schedulers)

        # Dummy environment
        env = gym.make(ename)

        # TODO: adjoin all of these into a single matrix

        # Environments are layed out heurestics x schedulers
        self.envs = [
            [
                [
                    gym.make(ename) for j in range(self.schs)
                ] for i in range(self.nhrs)
            ]
        ]

        # Agents are also layed out in a matrix
        self.agents = [
            [
                [
                    DQNAgent(models[ename], Policy(
                        heurestics[i].name + schedulers[j].name,
                        heurestics[i], schedulers[j]
                    ), env.action_space.n, gamma) for j in range(self.schs)
                ] for i in range(self.nhrs)
            ]
        ]

        # Matrix entry
        MatEntry = namedtuple('MatEntry', ['obs', 'rewards', 'done', 'total_rewards'])

        self.matrix = [
            [
                [
                    MatEntry(obs = None, reward = 0, done = False, total_rewards = [])
                ] for j in range(self.schs)
            ] for i in range(self.nhrs)
        ]
    
    def run_index(self, i, j):
        me = self.matrix[i][j]

        # Do nothing if already done
        if me.done:
            return
        
        env = self.envs[i][j]
        agent = self.agents[i][j]

        me.obs, reward, me.done, env = agent.step(env, me.obs)
        me.reward += reward

        # Updating
        self.agents[i][j]
        self.envs[i][j]

        return me.done
    
    def run_step(self):
        done = True

        for i in range(self.nhrs):
            for j in range(self.schs):
                done &= self.run_index(i, j)
        
        return True

    def run(self):
        for eps in range(self.episodes):
            # Reset the observations
            for i in range(self.nhrs):
                for j in range(self.schs):
                    env = self.envs[i][j]
                    me = self.matrix[i][j]
                    me.obs = env.reset()
            
            for steps in range(self.steps):
                done = self.run_step()

                if done:
                    break
            
            # Add all the rewards to the matrix entries
            for i in range(self.nhrs):
                for j in range(self.schs):
                    me = self.matrix[i][j]
                    me.total_rewards.append(me.reward)
                    me.reward = 0

            # TODO: need to print results to csv (after each episode, not whole thing)

etest = EnvTest('CartPole-v1', [1, 2], [3, 4, 7])
