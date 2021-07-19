import os

su_off = False
interval = 500

def notify(msg):
    if not su_off:
        os.system(f'python3 bot.py \'@everyone: {msg}\' &')
    else:
        print('Ignoring notify per su_off...')

def log(msg):
    if not su_off:
        os.system(f'python3 bot.py \'{msg}\' &')
    else:
        print('Ignoring log per su_off...')
