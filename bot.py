import os
import asyncio
import discord
import sys

from discord.ext import commands

client = discord.Client()
string = None
logged = True

@client.event
async def on_ready():
    print('Logged in as {0.user}'.format(client))

    channel = discord.utils.get(
        client.get_all_channels(),
        name = 'general'
    )

    await channel.send(sys.argv[1])
    await client.close()


if os.path.isfile('cache.txt'):
    cache = open('cache.txt', 'r')
    token = cache.readline()
else:
    token = input('Enter bot token: ')
    cache = open('cache.txt', 'w')
    cache.write(token)
client.run(token)