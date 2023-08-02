
import numpy as np

class Perceptron:

    def __init__(self, w_lr, b_lr, epochs):

        # input_size = None
        self.w = None
        self.b = None
        self.w_lr = w_lr
        self.b_lr = b_lr
        self.epochs = epochs

    def fit(self, X_train, Y_train):

        # Initializing weights and bias
        self.w = np.random.rand(1, 1)
        self.b = np.random.rand(1, 1)

        # Iterating until the number of epochs
        for epoch in range(self.epochs):
            #for i in range(X_train.shape[0]):
                #x = X_train[i]
                #y = Y_train[i]

            y_pred = X_train @ self.w + self.b

                #y_pred = x * self.w + self.b

            error = Y_train - y_pred

            self.w = self.w + (error * X_train * self.w_lr)
            self.b = self.b + (error * self.b_lr)
        return self.w, self.b


    def predict(self, X_test):

        Y_pred = (X_test @ self.w) + self.b

        return Y_pred

    def evaluate(self, X_test, Y_test, metric):

        Y_pred = self.predict(X_test)
        losses = []

        error = Y_test - Y_pred

        if metric == "mae":
            loss = np.mean(np.abs(error))
            #losses.append(loss)
        elif metric == "mse":
            loss = np.mean((error)**2)
            #losses.append(loss)

        return loss
