import csv
import time
import threading

import numpy as np

from scheduler import *
from matentry import MatEntry

# Colors
RED = '\033[1;91m'
GREEN = '\033[1;92m'
RESET = '\033[0m'

class EnvTest:
    # For average window
    max_time_length = 25

    def __init__(self, ename, heurestics, schedulers, episodes, steps, size, gamma = 0.95):
        self.heurestics = heurestics # TODO: no need to store
        self.schedulers = schedulers # TODO: also no need to store
        self.ename = ename

        # Timing
        self.time = time.time()
        self.times = []
        self.left = 0

        # Trial properties
        self.episodes = episodes
        self.steps = steps

        # Dimensions of the testing matrix
        self.nhrs = len(heurestics)
        self.schs = len(schedulers)

        # Actual matrix
        self.matrix = [
            [
                MatEntry(
                    ename,
                    heurestics[i],
                    schedulers[j],
                    size,
                    gamma
                ) for j in range(self.schs)
            ] for i in range(self.nhrs)
        ]
    
    '''
    TODO: redundant
    def run_step(self):
        done = True

        for i in range(self.nhrs):
            for j in range(self.schs):
                done &= self.matrix[i][j].run()
        
        return done
    '''

    def write_episode(self, csv, episode):
        fields = [episode]
        for i in range(self.nhrs):
            for j in range(self.schs):
                fields.append(self.matrix[i][j].score)
                fields.append(self.matrix[i][j].epsilon)
        csv.writerow(fields)

        # Get time
        end = time.time()

        self.times.append(end - self.time)
        if len(self.times) > self.max_time_length:
            del self.times[0]

        average = np.sum(self.times) / len(self.times)

        # In minutes
        left = (average * (self.episodes - episode + 1))/60

        diff = left - self.left
        out = (GREEN + f'+{diff:.3f}' + RESET)
            if diff > 0 else (RED + f'{diff:.3f}' + RESET)

        # TODO: use mutex here to clean output
        print(f'{self.ename} completed episode #{episode}'
                + f' in {end - self.time:.3f} secs ->'
                + f' average = {average:.3f} secs ->'
                + f' projected time left = {left:.3f} mins [{out}]')

        # Update times
        self.time = end
        self.left = left

    def run(self):
        # Setup csv
        csv_file = open(self.ename + '_results.csv', 'w')
        csv_writer = csv.writer(csv_file)

        # Gather the fields
        fields = ['Episode']
        for i in range(self.nhrs):
            for j in range(self.schs):
                name = self.matrix[i][j].pname
                fields.append(name + '.score')
                fields.append(name + '.epsilon')

        csv_writer.writerow(fields)

        # Set the times
        self.time = time.time()
        start = time.time()

        # Run the simulations
        for eps in range(self.episodes):

            ''' Reset the observations
            for i in range(self.nhrs):
                for j in range(self.schs):
                    self.matrix[i][j].reset()
            
            for step in range(self.steps):
                done = self.run_step()

                if done:
                    break
            '''

            # Create the thread matrix
            thread_matrix = []

            # TODO: use numpy's ndindex and layout in linear fassion
            for i in range(self.nhrs):
                thread_row = []
                for j in range(self.schs):
                    thread = threading.Thread(
                            target = self.matrix[i][j].run_episode,
                            args = (self.steps, )
                    )

                    thread.start()
                    thread_row.append(thread)

                thread_matrix.append(thread_row)

            # Collect all the threads
            for i in range(self.nhrs):
                for j in range(self.schs):
                    thread_matrix[i][j].join()

            # Log episode information
            self.write_episode(csv_writer, eps + 1)

        # Report end and total time
        end = time.time()
        total = (end - start)/60

        print(f'Simulation {self.ename} finished in '
            + f'{total:.3f} mins.')
