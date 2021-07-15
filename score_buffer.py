import numpy as np

class ScoreBuffer:
    max_buf_len = 25

    def __init__(self):
        self.array = []

    def append(self, score):
        self.array.append(score)

        if len(self.array) > self.max_buf_len:
            del self.array[0]

    def average(self):
        return np.average(self.array)
