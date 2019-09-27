from scipy.io import loadmat
import numpy as np
from scipy.special import expit as sigmoid


def propagateForward(X_row, thetas):
    features = X_row
    for i in range(len(thetas)):
        theta = thetas[i]
        z = np.dot(theta, features)
        a = sigmoid(z)
        if i == len(thetas)-1:
            return a
        a = np.insert(a,0,1)    #add bias unit
        features = a


def predictNN(X_row, thetas):
    classes = [i for i in range(num_labels)]
    output = propagateForward(X_row, thetas)
    return classes[np.argmax(np.array(output))]


def accuracy(X, y, thetas):
    correct = 0
    total = X.shape[0]
    for i in range(total):
        if predictNN(X[i,:], thetas) == y[i]:
            correct += 1
    return (correct / total) * 100


data = loadmat('ex3data1')

X = data['X']
y = data['y']

X = np.hstack((np.ones((len(y),1)),X))

y[y==10] = 0

weights = loadmat('ex3weights')

theta1 = weights['Theta1']
theta2 = weights['Theta2']

print('Theta 1 dimensions: {}'.format(theta1.shape))
print('Theta 2 dimensions: {}'.format(theta2.shape))

# Setup the parameters you will use for this exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 0 to 9

# swap first and last columns of Theta2, due to legacy from MATLAB indexing,
# since the weight file ex3weights.mat was saved based on MATLAB indexing
theta2 = np.roll(theta2, 1, axis=0)

thetas = [theta1, theta2]


acc = accuracy(X, y, thetas)
print('Accuracy: {}'.format(acc))