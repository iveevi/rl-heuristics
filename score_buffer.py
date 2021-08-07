import numpy as np

class ScoreBuffer:
    def __init__(self, length = 25):
        self.array = []
        self.length = length

    def append(self, score):
        self.array.append(score)

        if len(self.array) > self.length:
            del self.array[0]

    def average(self):
        return np.average(self.array)
    
    def max(self):
        return np.max(self.array)
    
    def min(self):
        return np.min(self.array)
    
    def range(self):
        rng = self.max() - self.min()

        if rng == 0:
            return 1e-10
        
        return rng
