import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


file1 = open('scores.txt', 'r')
Lines = file1.readlines()
 
count = 0
# Strips the newline character
scores = []
for line in Lines:
    count += 1
    print("Line{}: {}".format(count, line.strip()))
    scores.append(float(line.strip()))
    # if count >10: break

scores = np.array(scores)
x = np.array(range(len(scores)))
print(type(scores[0]))
print(x)

d = {'average_scores': scores, 'trials': x}
pdnumsqr = pd.DataFrame(d)


sns.set_style("darkgrid")
sns.lineplot(x='trials', y='average_scores', data=pdnumsqr)
plt.show()


file1 = open('timePerRun.txt', 'r')
Lines = file1.readlines()
 
count = 0
# Strips the newline character
scores = []
for line in Lines:
    count += 1
    print("Line{}: {}".format(count, line.strip()))
    scores.append(float(line.strip()))
    # if count >10: break

scores = np.array(scores)
x = np.array(range(len(scores)))
print(type(scores[0]))
print(x)

d = {'Average Time': scores, 'Trials': x}
pdnumsqr = pd.DataFrame(d)


sns.set_style("darkgrid")
sns.lineplot(x='Trials', y='Average Time', data=pdnumsqr)
plt.show()


