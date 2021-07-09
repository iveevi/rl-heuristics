import discord
import sys

from discord.ext import tasks

client = discord.Client()
channel = None
sock = None

@tasks.loop(seconds = 1)
async def test():
    await channel.send('Test')

@client.event
async def on_ready():
    global channel

    print('Logged in as {0.user}'.format(client))

    channel = discord.utils.get(
        client.get_all_channels(),
        name = 'general'
    )

    test.start()

    await channel.send('Starting program...')
    # await client.close()

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith('$'):
        await message.channel.send('Hello!')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Expected at least one argument')

        exit(-1)

    client.run(sys.argv[1])
