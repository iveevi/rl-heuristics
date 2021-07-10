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

    await channel.send(sys.argv[2])
    await client.close()

token = sys.argv[1]
client.run(token)