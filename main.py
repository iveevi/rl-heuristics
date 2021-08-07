import yaml
import time

from multiprocessing import Process

import notify
import heurestics
import schedulers
import runs

from upload import *
from time_buffer import *

# Preprocess YAML file
fout = open('config.yml', 'r')
pre_envs = yaml.safe_load(fout) [0]['Environments']

environments = dict()
for env in pre_envs:
	dicts = list(env.values())[0]

	ndict = dict()
	for d in dicts:
		name = next(iter(d))
		ndict[name] = d[name]
	env = {next(iter(env)): ndict}

	environments = {**environments, **env}

# Load up the processes
pool = []
ids = []
index = 0

dirn = setup(environments)
for env in environments:
    ecp = environments[env]
    hrs = ecp['heurestics']
    scs = ecp['schedulers']
    trials = ecp['trials']

    for hrd in hrs:
        for scd in scs:
            # Construct the heurestic from config
            hrname = next(iter(hrd))
            hr = getattr(heurestics, hrname)
            hr = heurestics.Heurestic(hrd[hrname], hr)

            # Construct the scheduler from config
            scname = next(iter(scd))
            sc = getattr(schedulers, scname)(*scd[scname])

            for i in range(trials):
                # TODO: use run_polciy right away
                pool.append(Process(target = runs.run_policy,
                    args = (env, ecp['skeleton'], hr,
                    sc, i, ecp['episodes'], ecp['steps'], dirn)))
                print('Adding \"' + env + ': ' + hr.name + ' and ' + sc.name +
                        ': ' + str(i + 1) + '\" as index #' + str(index))
                ids.append(env + ': ' + hr.name + ' and ' + sc.name + ': ' +
                        str(i + 1))
                index += 1

    if not ecp['ts-tutoring']:
        continue

    hrs = ecp['ts-heurestics']
    scs = ecp['ts-schedulers']
    for hrd in hrs:
        for scd in scs:
            # Construct the heurestic from config
            hrname = next(iter(hrd))
            hr = getattr(heurestics, hrname)
            hr = heurestics.Heurestic(hrd[hrname], hr)

            # Construct the scheduler from config
            scname = next(iter(scd))
            sc = getattr(schedulers, scname)(*scd[scname])

            for i in range(trials):
                pool.append(Process(target = runs.run_tutoring,
                    args = (env, ecp['skeleton'], hr, sc,
                    i, ecp['episodes'], ecp['steps'], dirn)))
                pid = 'Tutoring (TS): ' + hr.name + ' and ' + \
                    sc.name + ': ' + str(i + 1)
                print('Adding \"' + env + ': ' + pid + '\" as index #' + str(index))
                ids.append(env + ': ' + pid)
                index += 1

# Launch the processes
start = time.time()
notify.su_off = True

for proc in pool:
    proc.start()

# Start collection process
dones = [False] * len(pool)

k = 0
while len(pool) > 0:
    count = dones.count(True)
    print(f'Pool loop #{k}, count = {count}, target = {len(pool)}')

    if count == len(pool):
        break

    k += 1
    for i in range(len(pool)):
        if not dones[i] and not pool[i].is_alive():
            pool[i].join()
            dones[i] = True

            print(GREEN + 'Process finished -> ' + ids[i] + RESET)

    # No need to check all the time
    time.sleep(1)

# Log completion
msg = f'Completed all simulations in {fmt_time(time.time() - start)}, see `{dirn}`'
print(msg)
notify.notify(msg)

# Upload data
upload(dir)
