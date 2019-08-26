import numpy as np
import matplotlib.pyplot as plt

def feature_normalization(X):
    means = []
    stds = []
    for column in range(0, X.shape[1]):
        means.append(np.mean(X[:, column]))
        stds.append(np.std(X[:, column]))
    X = (X - means) // stds
    return X, means, stds
data = np.loadtxt("ex1data2.txt", delimiter=',')
X = np.array(data[:,:2])
y = np.array([data[:,2]])

X, mu, sigma = feature_normalization(X)

print(mu , sigma)