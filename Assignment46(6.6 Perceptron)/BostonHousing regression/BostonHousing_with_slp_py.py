import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from perceptron_version_BostonHousing import Perceptron
# import matplotlib; matplotlib.use("TkAgg")
from sklearn.model_selection import train_test_split


data = pd.read_csv('data/BostonHousing.csv')
data.head()

plt.scatter(data['lstat'],data['medv'])
plt.xlabel("LSTAT")
plt.ylabel("MEDV")
plt.show()

X = np.array([data["lstat"], data["rm"]])
Y = np.array(data["medv"])
print(X.shape)

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

#X_train_losses, X_train_losses_epoch = perceptron.fit_and_loss(X_train, Y_train)
X_train_losses, X_train_losses_epoch, w, b = perceptron.fit_and_losses(X_train, Y_train)
#print(X_train_losses)
#print(X_train_losses_epoch)
print('b:', b)
print('w: ', w)
print('b shape:', b.shape)
print('w shape: ', w.shape)
X_test_losses = perceptron.evaluate(X_test, Y_test)
#print("X_test_losses: ",X_test_losses)

Y_pred_X_train = perceptron.predict(X_train)
#print("Y_pred_X_train: ",Y_pred_X_train)

#from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

X1 = X_train[:, 0]
X2 = X_train[:, 1]

x1, x2 = np.meshgrid(X1, X2)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ims = []
print("w: ",w.shape)
print("b: ",b.shape)

eq = w[0]*x1 + w[1]*x2 + b

# ax = fig.add_subplot(111, projection='3d')

ax.clear()

ax.scatter(X1, X2, Y_train, color='green', alpha=0.5)
ax.plot_surface(x1, x2, eq)
ax.set_xlabel('rm')
ax.set_ylabel('zn')
ax.set_zlabel('price')
ims.append([ax])
plt.pause(0.001)
plt.show()

#ani = animation.ArtistAnimation(fig, ims, interval=10, blit=False)

#writer = animation.PillowWriter(fps=15, metadata=dict(artist='Me'), bitrate=100)
#ani.save("movie.gif", writer=writer)

#plt.show()