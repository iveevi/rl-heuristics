import discord
import time
import asyncio

from multiprocessing import Process

client = discord.Client()

@client.event
async def on_ready():
    print('Logged in as {0.user}'.format(client))
    
    channel = discord.utils.get(
        client.get_all_channels(),
        name = 'general'
    )

    await channel.send('Starting program...')

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith('$'):
        await message.channel.send('Hello!')

if __name__ == '__main__':
    token = input('Enter bot token: ')

    # process = Process(target = func, args = ())
    # process.start()

    client.run(token)
