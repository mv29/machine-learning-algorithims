"""
The model for kNN is the entire training dataset.
 When a prediction is required for a unseen data instance, the kNN algorithm will search through the training dataset for the k-most similar instances.
  The prediction attribute of the most similar instances is summarized and returned as the prediction for the unseen instance.

The similarity measure is dependent on the type of data. For real-valued data, the Euclidean distance can be used.
Other other types of data such as categorical or binary data, Hamming distance can be used.
"""

import numpy as np
from sklearn import datasets
iris = datasets.load_iris()
iris_data = iris.data
iris_labels = iris.target

np.random.seed(42)
indices = np.random.permutation(len(iris_data))
n_training_samples = 100
learnset_data = iris_data[indices[:-n_training_samples]]
learnset_labels = iris_labels[indices[:-n_training_samples]]
testset_data = iris_data[indices[-n_training_samples:]]
testset_labels = iris_labels[indices[-n_training_samples:]]

import matplotlib.pyplot as plt
colours = ("r", "b")
X = []# curly braces dictionary and square brackets list
for iclass in range(3):
    X.append([[], [], []])
    for i in range(len(learnset_data)):
        if learnset_labels[i] == iclass:
            X[iclass][0].append(learnset_data[i][0])
            X[iclass][1].append(learnset_data[i][1])
            X[iclass][2].append(sum(learnset_data[i][2:]))
colour = ("k", "g", "b")
fig = plt.figure()
ax = fig.add_subplot(211)
for iclass in range(3):
       ax.scatter(X[iclass][0], X[iclass][1], X[iclass][2], c=colour[iclass])
#plt.show()
ax = fig.add_subplot(212)
for iclass in range(3):
       ax.hist(X[iclass][0], color=colour[iclass])
plt.show()


def distance(instance1, instance2):
    # just in case, if the instances are lists or tuples:
    instance1 = np.array(instance1)
    instance2 = np.array(instance2)
    return np.linalg.norm(instance1 - instance2)

def get_neighbors(training_set,labels,test_instance,k,distance=distance):
    distances = []
    for index in range(len(training_set)):
        dist = distance(test_instance, training_set[index])
        distances.append((training_set[index], dist, labels[index]))
    distances.sort(key=lambda x: x[1])
    neighbors = distances[:k]
    return neighbors


from collections import Counter

def vote_distance_weights(neighbors, all_results=True):
    class_counter = Counter()
    number_of_neighbors = len(neighbors)
    print(number_of_neighbors,"no of neigh")
    for index in range(number_of_neighbors):
        dist = neighbors[index][1]
        label = neighbors[index][2]
        class_counter[label] += 1 / (dist**2 + 1)
        # weighting/normalizing the contribution of each neihgbour for better prediction
        # As neighbour whcich are closers will have more identical characterstics

    print(len(class_counter),"lol")
    labels, votes = zip(*class_counter.most_common())
    #print(labels, votes)
    winner = class_counter.most_common(1)[0][0]
    votes4winner = class_counter.most_common(1)[0][1]
    if all_results:
        total = sum(class_counter.values(), 0.0)
        for key in class_counter:
             class_counter[key] /= total
        return winner, class_counter.most_common()
    else:
        return winner, votes4winner / sum(votes)

for i in range(n_training_samples):
    neighbors = get_neighbors(learnset_data,
                              learnset_labels,
                              testset_data[i],
                              6,
                              distance=distance)
    print("index: ", i,
          ", result of vote: ", vote_distance_weights(neighbors,
                                                      all_results=True))