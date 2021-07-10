import gym

from copy import copy

from dqnagent import DQNAgent
from models import models
from policy import Policy
from time_buffer import *
from scheduler import *
from colors import *
from notify import *

class Simulation:
    '''
    @param ename the name of environment in the gym registry
    @param trial the trial number
    @param batch_size the batch sizes
    '''
    def __init__(self, ename, pname, trial, heurestic, scheduler, batch_size, gamma):
        # Assigned attributes
        self.ename = ename
        self.pname = pname
        self.trial = trial
        self.batch_size = batch_size

        self.obs = None
        self.score = 0
        self.episode = 0
        self.epsilon = 1

        # Reward arrays
        self.rewards = []
        self.finals = []

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

        self.rewards.append(self.score)
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
        # Setting up time
        start = time.time()
        tb = TimeBuffer(episodes)
        proj = 0

        for episode in range(episodes):
            self.reset()
            tb.start()
            for step in range(steps):
                done = self.run()

                if done:
                    break
            
            t = tb.split()
            proj, fmt1, ftm2 = cmp_and_fmt(proj, tb.projected())

            # Log
            msg = f'{self.ename}, {self.pname}, #{self.trial}' \
                f' ep #{episode + 1} [{fmt_time(t)}], proj '
            print(msg + f'[{fmt1}]')

            # Log only a couple episodes (set as variable)
            if episode % interval == 0:
                log(msg + f'[{ftm2}]')
        
        # Notify
        t = time.time() - start
        msg = f'{self.ename}, {self.pname}, #{self.trial} finished in {fmt_time(t)}'
        print(msg)
        notify(msg)

    # Running without any training, to see final results
    def run_bench(self, episodes, steps):
        self.agent.set_policy(lambda x, y : -1)
        for episode in range(episodes):
            self.score = 0

            for step in range(steps):
                done = self.run()

                if done:
                    break

            self.finals.append(self.score)
