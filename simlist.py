from environment_simulation import EnvironmentSimulation
from policy import Heurestic
from upload import *
from scheduler import *
from heurestics import *
from notify import *

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
        10,
        500,
        32
)

# Simulation routine
def do_sims():

    setup([etest])
    etest.run()

    # Prompt for upload
    upload(sudo = True)

    notify(f'Simulations have terminated (see `{directory}`):thumbsup:')