import os
import notify

from simlist import do_sims

# Main routine
if __name__ == '__main__':
    notify.su_off = True
    do_sims()