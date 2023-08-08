import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from perceptron_version_BostonHousing import Perceptron
import matplotlib; matplotlib.use("TkAgg")
from sklearn.model_selection import train_test_split
from matplotlib import cm
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation


data = pd.read_csv('BostonHousing.csv')
data.head()
'''
plt.scatter(data['lstat'],data['rm'])
plt.xlabel("LSTAT")
plt.ylabel("MEDV")
plt.show()
'''
X = np.array([data["lstat"], data["rm"]])
Y = np.array(data["medv"])
print("X.shape([data[rm], data[zn]]): ", X.shape)
print("Y.shape(data[medv]): ", Y.shape)

X = X.reshape(-1, 2)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=True, test_size=0.2)

X_train = X_train.reshape(-1, 2)
X_test = X_test.reshape(-1, 2)
Y_train = Y_train.reshape(-1, 1)
Y_test = Y_test.reshape(-1, 1)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

perceptron = Perceptron(2, 0.001, 0.01, 10)

X_train_losses, X_train_losses_epoch, w, b = perceptron.fit_and_losses(X_train, Y_train)
#print(X_train_losses)
#print(X_train_losses_epoch)
print('b out:', b)
print('w out: ', w)
print('b shape:', b.shape)
print('w shape: ', w.shape)
X_test_losses = perceptron.evaluate(X_test, Y_test)
#print("X_test_losses: ",X_test_losses)

Y_pred_X_train = perceptron.predict(X_train)
#print("Y_pred_X_train: ",Y_pred_X_train)


X1 = X_train[:, 0]
X2 = X_train[:, 1]

x1, x2 = np.meshgrid(X1, X2)

print("x1:", x1)
print("x2:", x2)
print("x1.shape:", x1.shape)
print("x2.shape:", x2.shape)


fig = plt.figure(figsize=(10, 7))
ax1 = fig.add_subplot(121, projection="3d")
ax2 = fig.add_subplot(122)

ax1.scatter(X_train[:, 0], X_train[:, 1], Y_train, c='r', marker='o')
surface = x1 * w[0, 0] + x2 * w[0, 1] + b
ax1.plot_surface(x1, x2, surface, alpha=0.5, shade=True, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax1.set_xlabel('lstat')
ax1.set_ylabel('rm')
ax1.set_zlabel('medv')

ax2.plot( X_train_losses_epoch)
ax2.set_xlabel('iterations')
ax2.set_ylabel('values')


plt.pause(0.01)











#*******************************************************************************

'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ims = []

eq = w[0]*x1 + w[1]*x2 + b

ax = fig.add_subplot(111, projection='3d')

ax.clear()

ax.scatter(X1, X2, Y_train, color='green', alpha=0.5)
ax.plot_surface(x1, x2, eq)
ax.set_xlabel('rm')
ax.set_ylabel('zn')
ax.set_zlabel('price')
ims.append([ax])
plt.pause(0.001)
plt.show()
'''
#ani = animation.ArtistAnimation(fig, ims, interval=10, blit=False)

#writer = animation.PillowWriter(fps=15, metadata=dict(artist='Me'), bitrate=100)
#ani.save("movie.gif", writer=writer)

# plt.show()
