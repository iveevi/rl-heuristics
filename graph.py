import matplotlib.pyplot as plt

def graph_single(path):
    # Create the subplot (and change the theme)
    plt.style.use('ggplot')

    fig = plt.figure(figsize = (8, 6))

    scores = fig.add_subplot(2, 2, 1)
    eps = fig.add_subplot(2, 2, 3)
    finals = fig.add_subplot(1, 2, 2)

    spi = 0
    for i in range(len(path)):
        if path[i] == '/':
            spi = i

    title = path[spi + 1:-4].replace('_', ' ')
    fig.suptitle(f'Statistics for {title}')

    # Read the file
    file = open(path)
    lines = [line[:-1] for line in file.readlines()]

    final = 0
    while lines[final][0] != 'F': final += 1

    # Collect episodes
    episodes = lines[0].split(',')
    epsilons = lines[1].split(',')
    episodes = [float(s) for s in episodes[1:]]

    for line in lines[2 : final - 1]:
        split = line.split(',')

        label = split[0]
        values = split[1:]

        scores.plot(episodes, [float(s) for s in values], label = label)
    
    feps = lines[final - 1].split(',')
    feps = [float(s) for s in feps[1:]]
    for line in lines[final:]:
        split = line.split(',')

        label = split[0]
        values = split[1:]

        finals.plot(feps, [float(s) for s in values], label = label) 
    
    eps.plot(episodes, [float(s) for s in epsilons[1:]], label = 'Epsilon')

    finals.set_ylabel('Bench Scores')
    scores.set_ylabel('Scores')
    eps.set_ylabel('Epsilon')
    eps.set_xlabel('Episodes')
    finals.set_xlabel('Episodes')
    
    fig.tight_layout()
    plt.show()

graph_single('results_2021-07-07_13:32:09.393927/CartPole-v1/Great_HR_and_Damped_Oscillator.csv')