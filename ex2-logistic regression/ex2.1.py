import numpy as np
from matplotlib import pyplot as plt
from scipy.special import expit as sigmoid


#np.seterr(divide = 'warn')

def plot_data(pos, neg):
    plt.plot(pos[:,0], pos[:,1], '.', label='Admitted')
    plt.plot(neg[:, 0], neg[:, 1], 'x', label='Rejected')
    plt.xlabel("Exam1 score")
    plt.ylabel("Exam2 score")
    plt.title('Data Classification')
    plt.legend()
    plt.grid(True)

def plot_boundary(X, theta, pos, neg):
    plot_data(pos, neg)
    boundary_xs = np.array([np.min(X[:, 1]), np.max(X[:, 1])])
    boundary_ys = (-1. / theta[2]) * (theta[0] + theta[1] * boundary_xs)
    plt.plot(boundary_xs, boundary_ys, 'b-', label='Decision Boundary')
    plt.legend()

def plot_cost_change(j_history):
    plt.plot(range(len(j_history)), j_history, '-', label='Cost function')
    plt.xlabel('Iterations')
    plt.ylabel('Cost function')
    plt.title('Cost function change')
    plt.legend()
    plt.grid(True)

def feature_scale(X):
    means, stds = [], []
    for i in range(X.shape[1]):
        means.append(np.mean(X[:,i]))
        stds.append(np.std(X[:,i]))
    X = (X - means) / stds
    return X

def hypothesis(X, theta):
    return sigmoid(np.dot(X,theta))

EPSILON = 1e-5

def compute_cost(X, y, theta, mylambda = 0):
    h = hypothesis(X, theta)
    m = y.shape[0]
    term1 = np.dot(-y.T, np.log(h + EPSILON))
    term2 = np.dot(-(1-y).T, np.log(1-h + EPSILON))
    return np.sum(term1 + term2) / m

def gradient_descent(X, y, theta, alpha, iterations):
    m = y.shape[0]
    j_history = []
    for i in range(0, iterations):
        h = hypothesis(X, theta)
        grad = np.dot(X.T, (h-y)) / m
        theta = theta - alpha * grad
        j_history.append(compute_cost(X, y, theta))
    return theta, j_history

def accuracy_measure(X, theta):
    m = X.shape[0]
    p = np.zeros(m)
    for i in range(m):
        if hypothesis(X[i], theta) >= 0.5:
            p[i] = 1
        else:
            p[i] = 0
    accuracy = np.mean(y[:, 0] == p) * 100
    print('Training accuracy: {}%'.format(accuracy))


#----------------------------------------------------------------------------

data = np.loadtxt("ex2data1.txt", delimiter=',')
X, y = data[:, 0:2], np.array([data[:, 2]]).T

m = len(y)

positive = np.array([X[i] for i in range(X.shape[0]) if y[i] == 1])
negative = np.array([X[i] for i in range(X.shape[0]) if y[i] == 0])

#Plotting Data
plt.figure(1)
plot_data(positive, negative)
plt.show()

X = np.append(np.ones((X.shape[0], 1)), X, axis = -1)    #Adding columns of 1s as x0

print('Some test cases:')

#Test case 1

initial_theta = np.zeros((X.shape[1],1))    #3x1 Dimensions

#print('initial theta dimensions: {}'.format(initial_theta.shape))
print('Cost at initial theta: {}'.format(compute_cost(X, y, initial_theta)))
print('grad1:')
h1 = hypothesis(X, initial_theta)
grad1 = np.dot(X.T, h1-y)/m
print('Gradient at initial theta:')
print(grad1)

#Test case 2

test_theta = np.array([[-24], [0.2], [0.2]])

cost = compute_cost(X, y, test_theta)
print('Cost at test theta: {}'.format(cost))

h2 = hypothesis(X, test_theta)
grad2 = np.dot(X.T, h2-y)/len(y)
print('Gradient at test theta:')
print(grad2)
#-------------------------------------------------------------------
#Using Gradient Descent algorithm

theta, j_history = gradient_descent(X, y, test_theta, 0.001, 300)
print('Optimized theta:')
print(theta)
print('Final minimized cost: {}'.format(j_history[-1]))
plt.figure(2)
plot_cost_change(j_history)
plt.show()
plt.figure(3)
plot_boundary(X, theta, positive, negative)
plt.show()
accuracy_measure(X, theta)
#-------------------------------------------------------------------
print('Predicting a test case:')
prob = hypothesis(np.array([[1, 48, 85]]) , theta)
print('Predicted probability: {}'.format(prob))
print('Expected value: 0.775 +/- 0.002\n')
