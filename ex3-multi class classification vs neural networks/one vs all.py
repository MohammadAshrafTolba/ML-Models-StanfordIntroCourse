import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.special import expit as sigmoid
from scipy.optimize import minimize
import utilities as u


def hypothesis(X, theta):
    return sigmoid(np.dot(X, theta))

EPSILON = 1e-5

def lrCostFunc(theta, X, y, my_lambda = 0):
    #print('In cost function, X shape: {}'.format(X.shape))
    theta = theta.reshape((401, 1))
    #print('in cost function, theta shape: {}'.format(theta.shape))

    h = hypothesis(X, theta)
    m = X.shape[0]
    term1 = np.dot(y.T, np.log(h + EPSILON))
    term2 = np.dot((1-y).T, np.log(1-h + EPSILON))
    reg_term = np.dot(theta[1:].T, theta[1:]) * (my_lambda/(m*2))
    left_hand = np.sum(term1 - term2) / m
    cost = left_hand + np.sum(reg_term)

    grad = np.dot(X.T, h-y) / m
    #reg_term2 = theta[1:] * (my_lambda/m)
    grad[1:] = grad[1:] + reg_term
    return cost, grad

def costGradient(X, theta, my_lambda):
    theta = theta.reshape((401, 1))
    grad = np.dot(X.T, h - y) / m
    reg_term = theta[1:] * (my_lambda/m)
    grad[1:] = grad[1:] + reg_term
    return grad

def OneVsAll(X, y, num_labels, my_lambda=0):
    n = X.shape[1]                                  # n = number of features
    all_theta = np.zeros((n, num_labels))

    def findOptTheta(current_class):
        initial_theta = all_theta[:,current_class]
        result = minimize(lrCostFunc,
                          initial_theta,
                          (X,(y==current_class),my_lambda),
                          jac=True,
                          tol=1e-6,
                          method='TNC',
                          options={'maxiter' : 500})
        all_theta[:,current_class] = result.x

    for current_class in range(num_labels):
        print('Optimizing hand written number: {}...'.format(current_class))
        findOptTheta(current_class)
    print('--Done--\n')
    return all_theta

def predictOneVsAll(X_row, optimized_theta):
    classes = [i for i in range(10)]
    predictions = [0]*optimized_theta.shape[1]
    for i in range(0, 10):
        predictions[i] = hypothesis(X_row, optimized_theta[:,i])
    #print(predictions)
    return classes[predictions.index(max(predictions))]

def accuracy(X, y, optimized_theta):
    correct = 0
    total = X.shape[0]
    for i in range(X.shape[0]):
        if predictOneVsAll(X[i,:], optimized_theta) == y[i]:
            correct += 1
    return (correct/total) *100


data = loadmat('ex3data1')
#print(data)

X = data['X']
y = data['y']

y[y==10] = 0

m = len(y)

print('X dimensions: {}'.format(X.shape))
print('y dimensions: {}'.format(y.shape))

rand_indices = np.random.choice(m, 100, replace=False)
sample = X[rand_indices, :]

u.displayData(sample)   #visualizing a sample from the training data
plt.show()

X = np.hstack((np.ones((len(y),1)),X))

my_lambda = 0.8

print('==Optimizing theta==')
optimized_theta = OneVsAll(X, y, 10, my_lambda)

acc = accuracy(X, y, optimized_theta)
print('Accuracy: {}%'.format(acc))