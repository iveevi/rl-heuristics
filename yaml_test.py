import yaml
from pprint import pprint

import heurestics
import scheduler

fin = open('config.yml', 'r')

envs = yaml.safe_load(fin)[0]['Environments']

# pprint(envs)

x = dict()
for env in envs:
	dicts = list(env.values())[0]
	pprint(dicts)

	ndict = dict()
	for d in dicts:
		name = next(iter(d))
		print('name = ', name)
		ndict[name] = d[name]
	env = {next(iter(env)): ndict}

	pprint(env)
	x |= env
	print('=' * 200)

pprint(x)
