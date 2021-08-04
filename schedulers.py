import math

from scipy.optimize import root

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

class DampedOscillator(DelayedScheduler):
    epsilon = 1e-2

    '''
    @param m the minimum value of the scheduler
    @param p the period of the oscillation
    '''
    def __init__(self, episodes, lag = 0, p = 50, m = 0.1,
            name = 'DampedOscillator'):
        alpha = math.log((1 - m)/self.epsilon)/episodes

        base = lambda e : (1 - m) * math.exp(-alpha * e) + m
        osc = lambda e : math.exp(-alpha * e) * math.cos(2 * math.pi * (e + p/4)/p)

        ftn = lambda e : base(e) + osc(e)

        super().__init__(name, lag, ftn)

        # Additonal params
        self.episodes = episodes
        self.min = m
        self.period = p
        self.name = name

    def __copy__(self):
        return type(self)(self.episodes, self.lag, self.period, self.min,
                self.name)

class WaveDecay(DelayedScheduler):
    epsilon = 1e-2

    def __init__(self, episodes, lag = 0, p = 50, m = 0.1,
            name = 'DampedOscillator'):
        gamma = 0.9 # Bounce factor
        mp = 1 - m
        kgamma = lambda e : gamma ** ((e - 1) // p)

        modded = (episodes - 1) % p if (episodes - 1) % p != 0 else episodes % p
        slk = math.log(self.epsilon/(mp * kgamma(episodes)))/modded   # Decay factor

        ftn = lambda e : mp * math.exp(slk * ((e - 1) % p)) * kgamma(e) + m

        super().__init__(name, lag, ftn)

        # Additonal params
        self.episodes = episodes
        self.min = m
        self.period = p
        self.name = name

    def __copy__(self):
        return type(self)(self.episodes, self.lag, self.period, self.min,
                self.name)

class InverseWaveDecay(DelayedScheduler):
    epsilon = 1e-2

    def __init__(self, episodes, lag = 0, p = 50, m = 0.1,
            name = 'DampedOscillator'):
        gamma = 0.9 # Bounce factor
        mp = 1 - m
        kgamma = lambda e : gamma ** ((e - 1) // p)

        modded = (episodes - 1) % p if (episodes - 1) % p != 0 else episodes % p
        slk = math.log(self.epsilon/(mp * kgamma(episodes)))/modded   # Decay factor

        ftn = lambda e : mp * (1 - math.exp(slk * ((p - e) % p))) * kgamma(e) + m

        super().__init__(name, lag, ftn)

        # Additonal params
        self.episodes = episodes
        self.min = m
        self.period = p
        self.name = name

    def __copy__(self):
        return type(self)(self.episodes, self.lag, self.period, self.min,
                self.name)
