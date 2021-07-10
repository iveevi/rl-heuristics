import time
import numpy as np

class TimeBuffer:
    max_buffer_len = 25

    def __init__(self, iterations):
        self.end = time.time()
        self.buffer = []
        self.iterations = iterations
    
    def start(self):
        self.end = time.time()
    
    # Automatically progresses the iterations
    def split(self):
        end = time.time()
        self.buffer.append(end - self.end)
        self.end = end
        self.iterations -= 1

        if len(self.buffer) > self.max_buffer_len:
            del self.buffer[0]
    
    def __str__(self):
        average = np.sum(self.buffer)/len(self.buffer)
        average *= self.iterations

        hours = average // (60 * 60)
        average %= 60 * 60

        minutes = average // 60
        average %= 60

        out = ''

        if hours > 0:
            out += f'{hours}h '

        if minutes > 0:
            out += f'{minutes}m '
        
        out += f'{average}s'