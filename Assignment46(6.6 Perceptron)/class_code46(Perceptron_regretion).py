import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use("TkAgg")
import time
from sklearn.model_selection import train_test_split


data = pd.read_csv("data/weight-height.csv")

X = data["Height"].values
Y = data["Weight"].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,shuffle=True, test_size=0.99)

print(X_train.shape)

X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
Y_train = Y_train.reshape(-1, 1)
Y_test = Y_test.reshape(-1, 1)

# print(X_train.shape)
# print(X_test.shape)

#plt.scatter(X_train, Y_train, color = "blue")
#plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2)
# training
w = np.random.rand(1, 1)
b = np.random.rand(1, 1)
# print(w)
learning_rate_w = 0.0001
learning_rate_b = 0.01

losses = []
epochs = 20
for j in range(epochs):
    for i in range(X_train.shape[0]):
        x = X_train[i]
        y = Y_train[i]

        y_pred = x * w + b

        error = y - y_pred

        # SGD update
        w = w + (error * x * learning_rate_w)
        b = b + (error * learning_rate_b )
        print(w)
        #time.sleep(0.5)

        #mae

        #loss = np.mean(np.abs(error))
        #losses.append(loss)

        #mse

        loss = np.mean(error ** 2)
        losses.append(loss)

        Y_pred = X_train * w + b
        ax1.clear()
        ax1.scatter(X_train, Y_train, color = "blue")
        ax1.plot(X_train, Y_pred, color = "red")

        ax2.clear()
        ax2.plot(losses)
        plt.pause(0.01)