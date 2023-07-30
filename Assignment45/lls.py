import numpy as np
from numpy.linalg import inv


class LinearLeastSquare:

    def __init__(self):
        # self.w = None
        pass

    def fit(self, X_train, Y_train):

        # train
        self.w = inv(X_train.T @ X_train) @ X_train.T @ Y_train


    def predict(self, X_test):
        #test
        Y_pred = X_test @ self.w
        return Y_pred

    def evaluate(self, X_test, Y_test, metric):
        Y_pred = self.predict(X_test)

        if metric == "mae":
            loss = np.sum(np.abs(Y_test - Y_pred)) / len(Y_test)
        elif metric == "mse":
            loss = np.sum((Y_test - Y_pred)**2) / len(Y_test)

        return loss
