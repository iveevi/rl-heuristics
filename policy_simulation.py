from threading import Thread

from simulation import Simulation

# TODO: try to run many trials without incurring tensorflow warning (try to optimize a pure policy run)
class PolicySimulation():
    '''
    @param trials the number of trials to conduct
    '''
    def __init__(self, ename, heurestic, scheduler, trials, batch_size, gamma):
        # Transfering attributes
        self.policy = heurestic.name + ' ' + scheduler.name
        self.file = '\'' + ename + ' ' + self.policy + '\''

        self.batch_size = batch_size

        # Load the simulations into a vector for processing
        self.sims = [
            Simulation(ename, i + 1, heurestic, scheduler, batch_size, gamma)
                for i in range(trials)
        ]
    
    def run(self, episodes, steps):
        # Open the file
        data_file = open(self.file, 'w')

        # Write the first line
        data_file.write(
            (['Episodes'] + [i for i in range(1, episodes + 1)]).join(',')
        )
        data_file.flush()

        # Construct the pool
        pool = []
        for sim in self.sims:
            pool.append(Thread(
                target = sim.run_trial,
                args = (episodes, steps, )
            ))
        
        # Launch the pool
        done = [False for i in range(self.sims)]
        for thread in pool:
            thread.start()
        
        # Collect threads and write their results
        while len(pool) > 0:
            for i in range(len(self.sims)):
                if (not done[i]) and (not pool[i].is_alive()):
                    pool[i].join()
                    done[i] = True

                    # TODO: write/save the results
                    data_file.write(
                        ([f'Trial #{i + 1}'] + self.sims[i].reward).join(',')
                    )
                    data_file.flush()

                    # Rerun the loop
                    del pool[i]
                    break