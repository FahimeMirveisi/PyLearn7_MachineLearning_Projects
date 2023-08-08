import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use("TkAgg")
from sklearn.model_selection import train_test_split
from perceptron_version_abalone import Perceptron


data = pd.read_csv('abalone.csv')
data.head()

data["Sex"] = data["Sex"].replace(["F", "M", "I"], [0, 1, 2])


data.corr()

"""
plt.scatter(data['Length'], data['Height'])
plt.xlabel('Length of abalon')
plt.ylabel('Height of abalon')
plt.title('Length and Height of abalone')
plt.show()
"""

### Train with single layer perceptron


X = np.array(data["Length"])
Y = np.array(data["Height"])

print("X.shape (data[Length]):", X.shape)
print("Y.shape (data[Height]):", Y.shape)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=True, test_size=0.2)

X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
Y_train = Y_train.reshape(-1, 1)
Y_test = Y_test.reshape(-1, 1)

print("X_train.shape: ", X_train.shape)
print("X_test.shape: ", X_test.shape)
print("Y_train.shape: ", Y_train.shape)
print("Y_test.shape: ", Y_test.shape)


perceptron = Perceptron(1, 0.001, 0.01, 10)

X_train_losses, X_train_losses_epoch, w, b = perceptron.fit_and_losses(X_train, Y_train)
# print("X_train_losses: ", X_train_losses)
# print("X_train_losses_epoch: ", X_train_losses_epoch)
print("final w: ", w)
print("final b: ", b)
X_test_losses = perceptron.evaluate(X_test, Y_test)
# print("X_test_losses: ",X_test_losses)

Y_pred_X_train = perceptron.predict(X_train)
# print("Y_pred_X_train: ",Y_pred_X_train)

perceptron.plott(X_train, Y_train)
# Plot data graph and loss graph as 3 subplots in 1 window


'''
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

ax1.clear()
ax1.scatter(X_train, Y_train, color = "blue")
ax1.plot(X_train, Y_pred_X_train, color = "red")
ax1.set_xlabel('Length of abalon')
ax1.set_ylabel('Height of abalon')
ax1.set_title('Length and Height of abalone')

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