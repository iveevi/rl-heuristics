# General scheduler class
class Scheduler:
    def __init__(self, name, function):
        self.function = function
        self.iteration = 0
        self.name = name
    def __call__(self):
        self.iteration += 1
        return self.function(self.iteration)
    def reset(self):
        self.iteration = 0

class DelayedScheduler(Scheduler):
    '''
     @param lag the number of episodes to wait before starting the decay
    '''
    def __init__(self, name, lag, function):
        ftn = lambda e : 1 if e < lag else function(e)

        super().__init__(name, ftn)

class LinearDecay(DelayedScheduler):
    '''
    @param episodes the duration of the decay in episodes
    @param lag the number of episodes to wait before starting the decay
    @param min the smallest value of the decay (flat line value)
    '''
    def __init__(self, episodes, lag = 50, min = 0.1):
        # ftn = lambda e : 1 if e < lag else max(1 - (1 - min) * (e - lag)/episodes, min)
        ftn = lambda e : max(1 - (1 - min) * (e - lag)/episodes, min)

        super().__init__('Linear Decay', lag, ftn)

linear1 = Scheduler('linear', lambda x : max(1 - x/60, 0.1))
linear2 = LinearDecay(50, 10)

for i in range(100):
    print(f"#{i + 1: >4}: linear1 = {linear1(): .2f} linear2 = {linear2(): .2f}")