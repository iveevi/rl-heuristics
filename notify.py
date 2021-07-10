import os

bot_token = None
interval = 10

# TODO: add a notification queue
def get_token():
    global bot_token
    if os.path.isfile('cache.txt'):
        cache = open('cache.txt', 'r')
        token = cache.readline()
    else:
        token = input('Enter bot token: ')
        cache = open('cache.txt', 'w')
        cache.write(token)
    bot_token = token

def notify(msg):
    os.system(f'python3 bot.py {bot_token} \'@everyone: {msg}\'')

def log(msg):
    os.system(f'python3 bot.py {bot_token} \'{msg}\'')