import numpy as np
from matplotlib import pyplot as plt


# visualizing the data
def visualize_data(population, profit):
    plt.plot(population, profit, '.')
    plt.grid(True)
    plt.title("Given Population/Profit")
    plt.xlabel('Population in 10,000s')
    plt.ylabel('Profit in $10,000')

# hypothesis function
def hypothesis(X, theta):
    return np.dot(X, theta)

# Cost function
def compute_cost(population, profit, theta):
    return np.sum(pow(hypothesis(population, theta) - profit, 2)) / (len(profit)*2)

# gradient descent
def gradeient_descent(population, profit, theta, alpha, num_iters):
    j_history = []  # list to store cost function values along all iterations
    for i in range(0, num_iters):
        theta = theta - (alpha * np.dot(population.T, hypothesis(population, theta) - profit))/(len(profit))
        j_history.append(compute_cost(population, profit, theta))
    return theta, j_history

# Visualizing the convergence of the Cost funcion
def plot_convergence(j_history):
    plt.plot(range(len(j_history)), j_history, '-')
    plt.grid(True)
    plt.title("Convergence of cost function")
    plt.xlabel("Iterations")
    plt.ylabel("Cost function")

# Visualizing the linear regression
def show_linear_regression(population, profit, final_theta):
    plt.plot(population[:, 1], profit, '.')
    plt.plot(population[:, 1], hypothesis(population, final_theta))
    plt.legend(["Training data", "Linear Regression"])
    plt.xlabel('Population in 10,000s')
    plt.ylabel('Profit in $10,000')

# Loading the required data from ex1data1.tx file into 2 lists
data = np.loadtxt("ex1data1.txt", delimiter=',')
population, profit = np.transpose(np.array([data[:,0]])), np.transpose(np.array([data[:,1]]))

plt.figure(1)
visualize_data(population, profit)

population = np.hstack([np.ones(population.shape), population]) #adding columns of 1s as x0 to the X matrix

initial_theta = np.zeros((population.shape[1], 1))
print(population.shape, initial_theta.shape, hypothesis(population,initial_theta).shape)
j = compute_cost(population, profit, initial_theta)
print("Cost for theta of all zeroes is {}".format(j))   # j should be 32.072

iterations = 1500
alpha = 0.01
theta , j_history = gradeient_descent(population, profit, initial_theta, alpha, iterations)
print("Final parameters: ({},{})".format(theta[0][0], theta[1][0]))

plt.figure(2)
plot_convergence(j_history)

plt.figure(3)
show_linear_regression(population, profit, theta)

plt.show()
# Predicting values for population sizes of 35,000 and 70,000 (main objective)
predict1 = np.dot([1, 3.5], theta)  # 1 for the extra feature of x0, 3.5 for x1 (35,000 / 10000)
print('For population = 35,000, we predict a profit of {}'.format(predict1*10000))

predict2 = np.dot([1, 7], theta)
print('For population = 70,000, we predict a profit of {}'.format(predict2*10000))



