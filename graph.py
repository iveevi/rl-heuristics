import re
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme()
def graph_file(path):
    # Create the subplot (and change the theme)
    fig = plt.figure(figsize = (8, 6))

    scores = fig.add_subplot(3, 1, (1, 2))
    eps = fig.add_subplot(3, 1, 3)

    spi = 0
    for i in range(len(path)):
        if path[i] == '/':
            spi = i

    title = path[spi + 1:-4].replace('_', ' ')
    fig.suptitle(f'Statistics For {title}')

    # Read the file
    file = open(path)
    lines = [line[:-1] for line in file.readlines()]

    final = 0
    while lines[final][0] != 'B': final += 1

    # Collect episodes
    episodes = lines[0].split(',')
    epsilons = lines[1].split(',')
    episodes = [float(s) for s in episodes[1:]]

    for line in lines[2 : final]:
        split = line.split(',')

        label = split[0]
        values = split[1:]

        scores.plot(episodes, [float(s) for s in values], label = label)

    feps = lines[final].split(',')
    feps = [float(s) for s in feps[1:]]

    benchs = []
    for line in lines[final + 1:]:
        split = line.split(',')

        label = split[0]
        values = split[1:]

        benchs.append(values[0])
    
    print('Bench trials:', benchs)

    eps.plot(episodes, [float(s) for s in epsilons[1:]], label = 'Epsilon')

    # Axis labels
    scores.set_ylabel('Scores')
    eps.set_ylabel('Epsilon')
    eps.set_xlabel('Episodes')

    # Legends
    scores.legend()
    eps.legend()

    fig.tight_layout()
    plt.show()

    save = input('Save? [y/n] ')
    if save == 'y':
        index = path.find('/')
        pdfs = path[:index] + '/pdfs/'
        os.system(f'mkdir -p {pdfs}')
        loc = pdfs + path[spi + 1:-4] + '.pdf'
        print(f'Saving to {loc}...')
        fig.savefig(loc)
    print()

def file_averages(path):
    # Read the file
    file = open(path)
    lines = [line[:-1] for line in file.readlines()]

    final = 0
    while lines[final][0] != 'B': final += 1

    trials = [
        [float(s) for s in (line.split(','))[1:]] for line in lines[2:final - 1]
    ]

    finals = [
        [float(s) for s in (line.split(','))[1:]] for line in lines[final + 1:]
    ]

    trials_average = np.average(trials, axis = 0).tolist()
    finals_average = np.average(finals, axis = 0).tolist()

    policy = path[path.rindex('/') + 1: -4]
    print('\n' + policy + ', average training score:', np.average(trials_average))
    print(policy + ', average bench score:', np.average(finals_average))

    return (trials_average, finals_average)

def tutoring_averages(path):
    # Read the file
    file = open(path)
    lines = [line[:-1] for line in file.readlines()]

    final = 0
    while lines[final][0] != 'B': final += 1

    trials = [
        [float(s) for s in (line.split(','))[1:]] for line in lines[2:final - 1]
    ]

    finals_average = [
        float(line.split(',')[1]) for line in lines[final:]
    ]

    print('finals: ', finals_average)

    trials_average = np.average(trials, axis = 0).tolist()

    policy = path[path.rindex('/') + 1: -4]
    print('\n' + policy + ', average training score:', np.average(trials_average))
    print(policy + ', average bench score:', np.average(finals_average))

    return trials_average

def graph_full_environment(path, files):
    # Setup the figure
    plt.style.use('ggplot')

    fig = plt.figure(figsize = (8, 6))

    trials = fig.add_subplot(1, 1, 1)

    spi = 0
    for i in range(len(path)):
        if path[i] == '/':
            spi = i

    title = path[spi + 1:]
    fig.suptitle(f'Statistics For Environment {title}')

    # Collect averages
    avgs = [file_averages(path + '/' + file) for file in files if file != 'TS_Tutoring.csv']

    tutoring_averages(path + '/TS_Tutoring.csv')

    # Assume the lengths are the same
    tlen = len(avgs[0])

    # Episodes for each
    teps = [i + 1 for i in range(tlen)]

    # Plot
    for i in range(len(avgs)):
        trials.plot(teps, avgs[i], label = (files[i])[:-4].replace('_', ' '))

    # Axis labels
    trials.set_ylabel('Trial Scores')
    trials.set_xlabel('Episode')

    # Legends
    trials.legend()

    fig.tight_layout()
    plt.show()

    save = input('Save? [y/n] ')
    if save == 'y':
        index = path.find('/')
        pdfs = path[:index] + '/pdfs/'
        os.system(f'mkdir -p {pdfs}')
        loc = pdfs + path[spi + 1:] + '.pdf'
        print(f'Saving to {loc}...')
        fig.savefig(loc)
    print()

def graph_environment(path):
    # List of directories
    files = [f for f in os.listdir(path) if f.endswith('.csv')]

    print('\n' + '-' * 23)
    print('Available data files:')
    print('-' * 23)

    for i in range(len(files)):
            print(f'[{i + 1}] {files[i]}')

    # Flush with a newline
    print()
    while True:
            index = input('Select a file using its index: ')

            if index == 'q' or index == 'quit':
                break
            elif index == 'a' or index == 'all':
                graph_full_environment(path, files)

                continue

            index = int(index)
            if index < 1 or index > len(files):
                print('\tIndex is out of bounds\n')
            else:
                graph_file(path + '/' + files[index - 1])

def graph_directory(path):
    subdirs = [f.name for f in os.scandir(path) if f.is_dir()]

    print('\n' + '-' * 23)
    print('Available environments:')
    print('-' * 23)

    for i in range(len(subdirs)):
            print(f'[{i + 1}] {subdirs[i]}')

    while True:
            index = int(input('\nSelect an environment using its index: '))

            if index < 1 or index > len(subdirs):
                print('\tIndex is out of bounds')
            else:
                graph_environment(path + '/' + subdirs[index - 1])

                break

if __name__ == "__main__":
    # List of directories
    directories = []

    # Extract the directories
    regex = re.compile('results*')
    for root, dirs, files in os.walk("."):
        for dir in dirs:
            if regex.match(dir):
                directories.append(dir)

    directories.sort(reverse = True)
    if len(directories) == 1:
        graph_directory(directories[0])
    else:
        print('-' * 63)
        print('Available data directories (ordered from most to least recent):')
        print('-' * 63)

        for i in range(len(directories)):
            print(f'[{i + 1}] {directories[i]}')

        while True:
            index = int(input('\nSelect a directory using its index: '))

            if index < 1 or index > len(directories):
                print('\tIndex is out of bounds')
            else:
                graph_directory(directories[index - 1])

                break
