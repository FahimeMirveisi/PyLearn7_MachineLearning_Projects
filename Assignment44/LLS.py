"""

lls class

"""

import numpy as np

class lls:

    def __init__(self, X, Y):
        self.X_train = np.array(X)
        self.Y_train = np.array(Y)
        self.X_train = self.X_train.reshape(-1, X.shape[1])

    def train(self, X_train, Y_train):

        self.w = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_train.T , X_train)), X_train.T), Y_train)


    def predict(self, x):
        Y_pred = self.w * x
        return Y_pred
