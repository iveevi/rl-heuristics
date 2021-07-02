import numpy as np

class Policy:
        def __init__(self, name, function, scheduler):
                self.name = name
                self.function = function
                self.scheduler = scheduler
        def do(self, state):
                epsilon = self.scheduler.tick()

                if np.random.rand() < epsilon:
                        return self.function(state)
                
                return -1