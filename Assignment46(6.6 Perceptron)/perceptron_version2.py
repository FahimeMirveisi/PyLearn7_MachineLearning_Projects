import numpy as np


class Perceptron:

    def __init__(self, w_lr, b_lr, epochs):

        self.w = np.random.rand(1, 1)
        self.b = np.random.rand(1, 1)
        self.w_lr = w_lr
        self.b_lr = b_lr
        self.epochs = epochs
        self.X_train_losses = []
        self.X_test_losses = []
        self.X_train_losses_epoch = []

    def SGD_update(self, error):

        self.w = self.w + (error * self.x * self.w_lr)
        self.b = self.b + (error * self.b_lr)

    def fit_and_loss(self, X_train, Y_train):

        for epoch in range(self.epochs):
            for i in range(X_train.shape[0]):

                self.x = X_train[i]
                self.y = Y_train[i]

                self.y_pred = self.x * self.w + self.b

                error = self.y - self.y_pred

                loss = np.mean(error ** 2)
                self.X_train_losses.append(loss)

                self.SGD_update(error)

            self.X_train_losses_epoch.append(loss)

        return self.X_train_losses, self.X_train_losses_epoch

    def predict(self, X_train):

        Y_pred = X_train * self.w + self.b

        return Y_pred