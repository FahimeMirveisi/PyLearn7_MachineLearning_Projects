
import numpy as np

class Perceptron:

    def __init__(self, w_lr=0.001, b_lr=0.01, epochs=10):

        # input_size = None

        self.w = np.random.rand(1, 1)
        self.b = np.random.rand(1, 1)
        self.w_lr = w_lr
        self.b_lr = b_lr
        self.epochs = epochs

    def calculate_error(self, y_real, y_hat):
        # y_real = y_test or y_train
        error = y_real - y_hat
        return error

    def mse_loss(self, y, y_pred):

        error = self.calculate_error(self.y, self.y_pred)
        loss = np.mean(error ** 2)

        return loss

    def SGD_optimizer(self,x, y):

        self.y_pred = self.x * self.w + self.b
        error = self.calculate_error(self.y, self.y_pred)

        loss = self.mse_loss(self.y, self.y_pred)

        self.w = self.w + (error * self.x * self.w_lr)
        self.b = self.b + (error * self.b_lr)

        return self.w, self.b, self.y_pred



    def fit(self, X_train, Y_train):

        X_train_losses = []
        # Iterating until the number of epochs
        for epoch in range(self.epochs):
            for i in range(X_train.shape[0]):
                self.x = X_train[i]
                self.y = Y_train[i]


                self.w, self.b, self.y_pred = self.SGD_optimizer(self.x, self.y)

                X_train_losses = np.append(self.y, self.y_pred)


        return X_train_losses


    def predict(self, X_test):

        Y_pred = (X_test * self.w) + self.b
        print("Y_pred shape in predict:", Y_pred.shape)
        print("Y_pred in predict:", Y_pred)

        return Y_pred

    def evaluate(self, X_test, Y_test):

        self.Y_pred = self.predict(X_test)
        #X_test_losses = []
        #error = self.calculate_error(Y_test, Y_pred)
        X_test_losses = self.mse_loss(Y_test, self.Y_pred)
        #X_test_losses.append(loss)

        return X_test_losses



