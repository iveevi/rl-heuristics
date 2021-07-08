import discord
import asyncio

from threading import Thread

client = discord.Client()

def func():
    for i in range(1000):
        print('\ti = ', i)

@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))
    
    channel = discord.utils.get(
        client.get_all_channels(),
        name = 'general'
    )

    func()

    await channel.send('Starting program...')

    print('Another print')

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith('$hello'):
        await message.channel.send('Hello!')
