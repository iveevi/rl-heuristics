import os

from multiprocessing import Process

'''
from environment_simulation import EnvironmentSimulation
from policy import Heurestic
from upload import setup, upload
from scheduler import *
from heurestics import *

# Define tests here (and later in another file)
etest = EnvironmentSimulation(
        'CartPole-v1',
        [
            Heurestic('Great HR', theta_omega)
        ],
        [
            LinearDecay(400, 50)
        ],
        1,
        100,
        500,
        32  # TODO: add bench episodes as a parameter
)

# Simulation routine
def do_sims():

    setup([etest])
    etest.run()

    # Prompt for upload
    upload(sudo = True)
'''

def get_token():
    if os.path.isfile('cache.txt'):
        cache = open('cache.txt', 'r')
        token = cache.readline()
    else:
        token = input('Enter bot token: ')
        cache = open('cache.txt', 'w')
        cache.write(token)
    return token

def notify(token, msg):
    os.system(f'python3 bot.py {token} \'{msg}\' &')

# Main routine
if __name__ == '__main__':
    token = get_token()

    notify(token, 'Simulations have terminated :thumbsup:')