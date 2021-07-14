import os

su_off = False
interval = 500

def notify(msg):
    if not su_off:
        os.system(f'python3 bot.py \'@everyone: {msg}\' &')

def log(msg):
    if not su_off:
        os.system(f'python3 bot.py \'{msg}\' &')
