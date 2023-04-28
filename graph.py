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
    if count >200: break

scores = np.array(scores)
x = np.array(range(len(scores)))


d = {'average_scores': scores, 'trials': x}
pdnumsqr = pd.DataFrame(d)


sns.set_style("darkgrid")
sns.lineplot(x='trials', y='average_scores', data=pdnumsqr)
# plt.show()


file1 = open('scores2.txt', 'r')
Lines = file1.readlines()
 
count = 0
# Strips the newline character
scores2 = []
for line in Lines:
    count += 1
    print("Line{}: {}".format(count, line.strip()))
    scores2.append(float(line.strip()))
    if count > 200: break

scores2 = np.array(scores2)
x2 = np.array(range(len(scores2)))

d = {'average_scores': scores, 'trials': x}
pdnumsqr = pd.DataFrame(d)
# print(len(),len(),len())
df = pd.DataFrame({'year': x,
                   'random': scores,
                   'lessnaive': scores2
                   })
sns.lineplot(data=df[['random', 'lessnaive']])

# sns.set_style("darkgrid")
# sns.lineplot(x='trials', y='average_scores', data=pdnumsqr)
plt.show()


# file1 = open('timePerRun.txt', 'r')
# Lines = file1.readlines()
 
# count = 0
# # Strips the newline character
# scores = []
# for line in Lines:
#     count += 1
#     print("Line{}: {}".format(count, line.strip()))
#     scores.append(float(line.strip()))
#     # if count >10: break

# scores = np.array(scores)
# x = np.array(range(len(scores)))


# d = {'Average Time': scores, 'Trials': x}
# pdnumsqr = pd.DataFrame(d)


# sns.set_style("darkgrid")
# sns.lineplot(x='Trials', y='Average Time', data=pdnumsqr)
# plt.show()

file1 = open('scorePerMove.txt', 'r')
Lines = file1.readlines()
 
count = 0
# Strips the newline character
scores = []
for line in Lines:
    count += 1
    print("Line{}: {}".format(count, line.strip()))
    scores.append(float(line.strip()))
    if count >50: break

scores = np.array(scores)
x = np.array(range(len(scores)))


d = {'score_per_move': scores, 'trials': x}
pdnumsqr = pd.DataFrame(d)


sns.set_style("darkgrid")
sns.lineplot(x='trials', y='score_per_move', data=pdnumsqr)
# plt.show()


file1 = open('scorePerMove2.txt', 'r')
Lines = file1.readlines()
 
count = 0
# Strips the newline character
scores2 = []
for line in Lines:
    count += 1
    print("Line{}: {}".format(count, line.strip()))
    scores2.append(float(line.strip()))
    if count > 49: break

scores2 = np.array(scores2)
x2 = np.array(range(len(scores2)))

d = {'average_scores': scores, 'trials': x}
pdnumsqr = pd.DataFrame(d)
print(len(scores),len(scores2),len(x))
df = pd.DataFrame({'year': x,
                   'random': scores,
                   'lessnaive': scores2
                   })
sns.lineplot(data=df[['random', 'lessnaive']])

# sns.set_style("darkgrid")
# sns.lineplot(x='trials', y='average_scores', data=pdnumsqr)
plt.show()