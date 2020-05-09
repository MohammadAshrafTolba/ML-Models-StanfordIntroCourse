import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression


# Loading the required data from ex1data1.tx file into 2 lists
data = np.loadtxt("../ex1data1.txt", delimiter=',')
X_train, y_train = np.transpose(np.array([data[:,0]])), np.transpose(np.array([data[:,1]]))

print(X_train.shape)

plt.figure(1)
plt.plot(X_train, y_train, '.')
plt.title("Given Population/Profit")
plt.xlabel('Population in 10,000s')
plt.ylabel('Profit in $10,000')


# fitting linear regression on the training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

plt.figure(2)
plt.plot(X_train, y_train, '.')
plt.plot(X_train, regressor.predict(X_train), '-')
plt.legend(["Training data", "Linear Regression"])
plt.xlabel('Population in 10,000s')
plt.ylabel('Profit in $10,000')

plt.show()

# predicting output for 35,000 and 70,000
X_test = np.transpose(np.array([[3.5, 7.0]]))
prediction = regressor.predict(X_test)
print("For 35,000 and 70,0000 outputs are {} respectively".format(prediction*10000))