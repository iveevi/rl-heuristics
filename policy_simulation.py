import time

from copy import copy
from threading import Thread
from pathos.multiprocessing import ProcessPool

from time_buffer import *
from simulation import Simulation
from upload import directory
from colors import *
from notify import *

import tensorflow as tf

from tensorflow import keras

def run_trial(agent, env, policy, episodes, steps):
    pass

class PolicySimulation():
    '''
    @param trials the number of trials to conduct
    '''
    def __init__(self, ename, heurestic, scheduler, trials, batch_size, gamma):
        # Transfering attributes
        self.ename = ename
        self.pname = heurestic.name + ' and ' + scheduler.name

        self.scheduler = copy(scheduler)
        self.file = directory + '/' + ename.replace(' ', '_')   \
            + '/' + self.pname.replace(' ', '_') + '.csv'

        self.batch_size = batch_size

        # Load the simulations into a vector for processing
        self.sims = [
            Simulation(ename, self.pname, i + 1,
                heurestic, scheduler, batch_size, gamma)
                for i in range(trials)
        ]
    
    def run(self, episodes, benchs, steps):
        start = time.time()

        # Open the file
        # data_file = open(self.file, 'w')

        # Write the first line
        # data_file.write(
        #     ','.join(['Episodes'] + [str(i) for i in range(1, episodes + 1)])
        #         + '\n'
        # )
        # data_file.write(
        #     ','.join(['Epsilons','1'] + [str(self.scheduler()) for i in range(1, episodes)])
        #         + '\n'
        # )
        # data_file.flush()

        # Construct the pool
        self.main = keras.models.clone_model(skeleton)
        self.main.set_weights(skeleton.get_weights())

        eplist = [episodes for i in range(len(self.sims))]
        stlist = [steps for i in range(len(self.sims))]

        pool = ProcessPool(len(self.sims))
        pool.map(Simulation.run_trial, self.sims, eplist, stlist)
        # pool = []
        # for sim in self.sims:
        #     pool.append(Process(
        #         target = sim.run_trial,
        #         args = (episodes, steps, )
        #     ))
        
        # # Launch the pool
        # for thread in pool:
        #     thread.start()
        
        # Collect threads and write their results
        # done = [False for i in range(len(pool))]
        # while True:
        #     if all(d == True for d in done):
        #         break
        
        '''
            for i in range(len(done)):
                if (not done[i]) and (not pool[i].is_alive()):
                    pool[i].join()
                    done[i] = True

                    data_file.write(
                        ','.join(map(str, [f'Trial #{i + 1}']
                            + self.sims[i].rewards)) + '\n'
                    )
                    data_file.flush()
        '''
        
        # Run the final bench (no training)
        data_file.write(
            ','.join(['Bench Episodes'] + [str(i) for i in range(1, benchs + 1)])
                + '\n'
        )
        
        for i in range(len(self.sims)):
            self.sims[i].run_bench(benchs, steps)

            data_file.write(
                ','.join(map(str, [f'Final #{i + 1}']
                    + self.sims[i].finals)) + '\n'
            )

        # Notify
        t = time.time() - start
        msg = f'Finished {self.ename} : {self.pname} in {fmt_time(t)}.'
        print(msg)
        notify(msg)
