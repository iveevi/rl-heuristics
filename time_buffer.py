import time
import numpy as np

from colors import *

def fmt_time(t):
    hours = t // (60 * 60)
    t %= 60 * 60

    minutes = t // 60
    t %= 60

    out = ''

    if hours > 0:
        out += f'{hours:.2f}h '

    if minutes > 0:
        out += f'{minutes:.2f}m '

    return out + f'{t:.2f}s'

def cmp_and_fmt(old, new):
    out1 = ''
    out2 = ''
    if new > old:
        out1 = GREEN + '+'
        out2 = '+'
    else:
        out1 = RED + '-'
        out2 = '-'
    return new, (out1 + fmt_time(new) + RESET), (out2 + fmt_time(new))

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
        t = end - self.end
        self.buffer.append(t)
        self.end = end
        self.iterations -= 1

        if len(self.buffer) > self.max_buffer_len:
            del self.buffer[0]

        return t

    def projected(self):
        average = np.sum(self.buffer)/len(self.buffer)
        average *= self.iterations

        return average

    def __str__(self):
        return fmt_time(self.projected())
