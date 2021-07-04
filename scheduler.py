# General scheduler class
class Scheduler:
    def __init__(self, name, function):
        self.function = function
        self.iteration = 0
        self.name = name
    def __call__(self):
        self.iteration += 1
        return self.function(self.iteration)
    def __copy__(self):
        return type(self)(self.name, self.function)
    def reset(self):
        self.iteration = 0

class DelayedScheduler(Scheduler):
    '''
     @param lag the number of episodes to wait before starting the decay
    '''
    def __init__(self, name, lag, function):
        ftn = lambda e : 1 if e < lag else function(e)
        super().__init__(name, ftn)

        # Additional params
        self.lag = lag
    def __copy__(self):
        return type(self)(self.name, self.lag, self.function)

class LinearDecay(DelayedScheduler):
    '''
    @param episodes the duration of the decay in episodes
    @param lag the number of episodes to wait before starting the decay
    @param min the smallest value of the decay (flat line value)
    '''
    def __init__(self, episodes, lag = 0, min = 0.1):
        ftn = lambda e : max(1 - (1 - min) * (e - lag)/episodes, min)
        super().__init__('Linear Decay', lag, ftn)

        # Additonal params
        self.episodes = episodes
        self.min = min
    def __copy__(self):
        return type(self)(self.episodes, self.lag, self.min)
