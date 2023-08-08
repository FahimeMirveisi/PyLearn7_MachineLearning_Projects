
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.animation as animation
from perceptron_version_Employee import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression


### Make a linear dataset using scikit-learn library

#%%

#Generate data
X, Y, coef = make_regression(n_samples=200, # number of samples
                             n_features=1, # number of features
                             n_informative=1, # number of useful features
                             noise=45, # bias and standard deviation of the guassian noise
                             coef=True, # true coefficient used to generated the data
                             random_state=0) # set for same data points for each run

# Scale feature x (years of experience) to range 0 to 30
X = np.interp(X, (X.min(), X.max()), (0,20))

# Scale target y (salary) to range 5 to 200 (Million Toman)
Y = np.interp(Y, (Y.min(), Y.max()), (5, 200))

# plt.ion() #interactive plot on
plt.plot(X, Y, '.', label='training data')
plt.xlabel('Years of experience')
plt.ylabel('Salary (Million Toman)')
plt.title('Experience and Salary')



### Split data to train and test datasets



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=True, test_size=0.2)

X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
Y_train = Y_train.reshape(-1, 1)
Y_test = Y_test.reshape(-1, 1)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)



### Implement the perceptron algorithm as a class
### Fit your model on the employee's salary dataset


perceptron = Perceptron(1, 0.001, 0.01, 10)

X_train_losses, X_train_losses_epoch = perceptron.fit_and_losses(X_train, Y_train)

X_test_losses = perceptron.evaluate(X_test, Y_test)
print("X_test_losses: ",X_test_losses)

Y_pred_X_train = perceptron.predict(X_train)
print("Y_pred_X_train: ",Y_pred_X_train)

perceptron.plott(X_train, Y_train)



### Plot data graph and loss graph as 2 subplots in 1 window

#%%
'''
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 10))

ax1.clear()
ax1.scatter(X_train, Y_train, color = "blue")
ax1.plot(X_train, Y_pred_X_train, color = "red")
ax1.set_xlabel('Years of experience')
ax1.set_ylabel('Salary (Million Toman)')
ax1.set_title('Experience and Salary')

ax2.clear()
ax2.plot(X_train_losses, color ="cyan")
ax2.set_xlabel("each data train")
ax2.set_ylabel("Loss")
ax2.set_title('Loss and each data train')

ax3.clear()
ax3.plot(X_train_losses_epoch, color ="yellow")
ax3.set_xlabel("epochs")
ax3.set_ylabel("Loss")
ax3.set_title('Loss and epochs')
plt.pause(0.01)
'''