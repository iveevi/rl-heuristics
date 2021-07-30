import re
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy

sns.set_theme()
def get_trial(path):
    # print(f'path = {path}')
    fout = open(path, 'r')
    lines = fout.readlines()
    epsilons = [float(s) for s in (lines[0][:-1].split(','))[1:]]
    scores = [float(s) for s in (lines[1][:-1].split(','))[1:]]
    finals = [float(s) for s in (lines[2][:-1].split(','))[1:]]
    return epsilons, scores, finals

def get_ts_trial(path):
    # print(f'path = {path}')
    fout = open(path, 'r')
    lines = fout.readlines()
    epsilons = [float(s) for s in (lines[0][:-1].split(','))[1:]]
    kepsilons = [float(s) for s in (lines[1][:-1].split(','))[1:]]
    scores1 = [float(s) for s in (lines[2][:-1].split(','))[1:]]
    scores2 = [float(s) for s in (lines[3][:-1].split(','))[1:]]
    finals1 = [float(s) for s in (lines[4][:-1].split(','))[1:]]
    finals2 = [float(s) for s in (lines[5][:-1].split(','))[1:]]
    return epsilons, kepsilons, scores1, scores2, finals1, finals2

def get_policy(path):
    out = []
    trials = [tf for tf in os.listdir(path)]
    for tr in trials:
        eps, scs, fin = get_trial(path + '/' + tr)
        out.append(scs)
    return out

# Take only the first agents score (by symmetry it should also
# at least somewhat represent the other agent)
def get_ts_policy(path):
    out = []
    trials = [tf for tf in os.listdir(path)]
    for tr in trials:
        eps, keps, scs1, scs2, fin1, fin2 = get_ts_trial(path + '/' + tr)
        out.append(scs1)
    return out

def get_policy_average(path, pol):
    # Variables
    conf = 0.95
    window = 10

    if pol == 'TS_Tutoring':
        tscores = get_ts_policy(path + '/' + pol)
    else:
        tscores = get_policy(path + '/' + pol)

    tscores_avg = np.average(tscores, axis = 0)
    tscores_sem = scipy.stats.sem(tscores, axis = 0)

    n = len(tscores)
    h = tscores_sem * scipy.stats.t.ppf((1 + conf) / 2.0, n-1)

    # Smooth out both
    tscores_avg = np.convolve(tscores_avg, np.ones(window), 'valid') / window
    h = np.convolve(h, np.ones(window), 'valid') / window

    return tscores_avg, h

def graph_policy(path):
    # Trial files
    trials = [tf for tf in os.listdir(path)]

    # Create the subplot (and change the theme)
    fig = plt.figure(figsize = (8, 6))

    scores = fig.add_subplot(3, 1, (1, 2))
    eps = fig.add_subplot(3, 1, 3)

    print(trials)

    epsilons = []
    ascores = []
    for tf in trials:
        teps, scs, finals = get_trial(path + '/' + tf)
        ascores.append(scs)
        epsilons.append(teps)

    # Plotting the scores
    size = len(ascores[0])
    episodes = range(1, size + 1)
    for i in range(len(ascores)):
        scores.plot(episodes, ascores[i], label = trials[i][:-4].replace('_', ' '))

    # Plotting the epsilons
    eps.plot(episodes, epsilons[0], label = 'Epsilon')

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

def graph_ts_policy(path):
    # Trial files
    trials = [tf for tf in os.listdir(path)]

    # Create the subplot (and change the theme)
    fig = plt.figure(figsize = (8, 6))

    scores = fig.add_subplot(3, 1, (1, 2))
    eps = fig.add_subplot(3, 1, 3)

    print(trials)

    epsilons = []
    ascores = []
    for tf in trials:
        # TODO: print statistics about the finals
        teps, tkeps, scs1, scs2, f1, f2 = get_ts_trial(path + '/' + tf)
        ascores.append((scs1, scs2))
        epsilons.append((teps, tkeps))

    # Plotting the scores
    size = len(ascores[0][0])
    episodes = range(1, size + 1)
    for i in range(len(ascores)):
        scores.plot(episodes, ascores[i][0],
            label = trials[i][:-4].replace('_', ' ') + ' (A1)')
        scores.plot(episodes, ascores[i][1],
            label = trials[i][:-4].replace('_', ' ') + ' (A2)')

    # Plotting the epsilons
    eps.plot(episodes, epsilons[0][0], label = 'Epsilon')
    eps.plot(episodes, epsilons[0][1], label = 'K-epsilon')

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

def graph_policy_average(path, pol):
    # Setup the figure
    fig = plt.figure(figsize = (8, 6))
    trials = fig.add_subplot(1, 1, 1)

    # Axis labels
    trials.set_ylabel('Trial Scores')
    trials.set_xlabel('Episode')

    # Plot
    tscores_avg, h = get_policy_average(path, pol)
    episodes = range(1, len(tscores_avg) + 1)
    p = trials.plot(episodes, tscores_avg, label = pol)
    trials.fill_between(episodes, np.subtract(tscores_avg, h),
        np.add(tscores_avg, h), color = p[0].get_color(), alpha = 0.2)

    # Legends
    trials.legend()

    fig.tight_layout()
    plt.show()

    # TODO: put in another function
    save = input('Save? [y/n] ')
    if save == 'y':
        index = path.find('/')
        pdfs = path[:index] + '/pdfs/'
        os.system(f'mkdir -p {pdfs}')
        loc = pdfs + path[spi + 1:] + '.pdf'
        print(f'Saving to {loc}...')
        fig.savefig(loc)
    print()

def graph_environment_averages(path, dirs):
    # Setup the figure
    fig = plt.figure(figsize = (8, 6))
    trials = fig.add_subplot(1, 1, 1)

    # Axis labels
    trials.set_ylabel('Trial Scores')
    trials.set_xlabel('Episode')

    # Plotting
    for pol in dirs:
        tscores_avg, h = get_policy_average(path, pol)
        episodes = range(1, len(tscores_avg) + 1)
        p = trials.plot(episodes, tscores_avg, label = pol)
        trials.fill_between(episodes, np.subtract(tscores_avg, h),
            np.add(tscores_avg, h), color = p[0].get_color(), alpha = 0.2)

    # Legends
    trials.legend()

    fig.tight_layout()
    plt.show()

    # TODO: put in another function
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
    dirs = [d for d in os.listdir(path)]

    print('\n' + '-' * 23)
    print('Available data files:')
    print('-' * 23)

    for i in range(len(dirs)):
            print(f'[{i + 1}] {dirs[i]}')

    # Flush with a newline
    print()
    while True:
            index = input('Select a file using its index: ')

            commas = index.split(',')
            if index == 'q' or index == 'quit':
                break
            elif len(commas) > 1:
                ndirs = []
                fail = False
                for cm in commas:
                    i = int(cm)
                    if i < 1 or i > len(dirs):
                        print('\tIndex is out of bounds\n')
                        fail = True
                        break
                    else:
                        ndirs.append(dirs[i - 1])

                if fail:
                    continue

                graph_environment_averages(path, ndirs)
                continue
            elif index == 'a' or index == 'all':
                graph_environment_averages(path, dirs)

                continue
            elif index[-1] == 'a':
                i = int(index[:-1])

                graph_policy_average(path, dirs[i - 1])
                continue

            index = int(index)
            if index < 1 or index > len(dirs):
                print('\tIndex is out of bounds\n')
            else:
                if dirs[index - 1] == 'TS_Tutoring':
                    graph_ts_policy(path + '/' + dirs[index - 1])
                else:
                    graph_policy(path + '/' + dirs[index - 1])

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
    directories = [d for d in os.listdir('.') if d.startswith('results_')]

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
