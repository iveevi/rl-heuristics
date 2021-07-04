import matplotlib.pyplot as plt

csv_file = open('CartPole-v1_results.csv')

header = csv_file.readline()
values = dict()

fields = header[:-1].split(',')
for field in fields[1:]:
    values[field] = []

lines = csv_file.readlines()

episodes = [i for i in range(1, len(lines) + 1)]

for line in lines:
    vs = (line[:-1].split(','))[1:]

    # print(vs)
    for i in range(len(vs)):
        values[fields[i + 1]].append(float(vs[i]))

fig, (scores, epsilons) = plt.subplots(2)
fig.suptitle('CartPole-v1 Results')

# Plotting results
for i in range(1, len(values) + 1, 2): # Scores
    stripped = (fields[i])[1:-7]
    scores.plot(episodes, values[fields[i]], label = stripped)

for i in range(2, len(values) + 1, 2): # Epsilons
    stripped = (fields[i])[1:-9]
    epsilons.plot(episodes, values[fields[i]], label = stripped)

scores.legend(loc = 'upper left')
epsilons.legend(loc = 'upper left')
plt.show()
