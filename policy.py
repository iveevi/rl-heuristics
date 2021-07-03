import numpy as np

class Policy:
    def __init__(self, name, function, scheduler):
        self.name = name
        self.function = function
        self.scheduler = scheduler
    def __call__(self, state):
          epsilon = self.scheduler()

           if np.random.rand() < epsilon:
                  return self.function(state)

            return -1

class Heurestic:
    def __init__(self, name, function):
        self.name = name
        self.function = function
    def __call__(x):
        return self.function(x)