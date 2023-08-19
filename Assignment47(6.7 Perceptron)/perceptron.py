import numpy as np
from tqdm import tqdm

class Perceptron:
    def __init__(self, learning_rate, input_length, function):
        self.learning_rate = learning_rate
        self.function = function
        self.weights = np.random.rand(input_length)
        self.bias = np.random.rand(1)

    def activation(self, x):
        if self.function == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.function == 'relu':
            return np.maximum(0, x)
        elif self.function == 'tanh':
            return np.tanh(x)
        elif self.function == 'linear':
            return x
        else:
            raise Exception("Unknown activation function")

    def fit(self, X_train, Y_train, X_test, Y_test, epochs):
        each_epoch_losses_train = []
        each_epoch_accuracy_train = []
        each_epoch_losses_test = []
        each_epoch_accuracy_test = []

        for epoch in tqdm(range(epochs)):
            for x, y in zip(X_train, Y_train):
                # forwarding
                y_pred = x @ self.weights + self.bias
                y_pred = self.activation(y_pred)

                # back propagation
                error = y - y_pred

                # updating
                self.weights += error * x * self.learning_rate
                self.bias += error * self.learning_rate

                # calculate loss and accuracy in each epoch
            train_loss, train_acc = self.evaluate(X_train, Y_train)
            test_loss, test_acc = self.evaluate(X_test, Y_test)

            each_epoch_losses_train.append(train_loss)
            each_epoch_accuracy_train.append(train_acc)
            each_epoch_losses_test.append(test_loss)
            each_epoch_accuracy_test.append(test_acc)

        return np.array(each_epoch_losses_train), np.array(each_epoch_accuracy_train),\
               np.array(each_epoch_losses_test), np.array(each_epoch_accuracy_test)
#        return each_epoch_losses_train, each_epoch_accuracy_train,\
#               each_epoch_losses_test, each_epoch_accuracy_test

    def calculate_loss(self, X_test, Y_test, metric):
        Y_pred = self.predict(X_test)
        if metric == 'mse':
            return np.mean(np.square(Y_test - Y_pred))
        elif metric == 'mae':
            return np.mean(np.abs(Y_test - Y_pred))
        else:
            raise Exception('Unknown metric')

#    def calculate_accuracy(self, X_test, Y_test):
#        Y_pred = self.predict(X_test)
#        RSS = np.sum((Y_test - Y_pred)**2)
#        TSS = np.sum((Y_test - np.mean(Y_test))**2)
#        accuracy = 1 - RSS/TSS
#        return accuracy

    def calculate_accuracy(self, X_test, Y_test):
        Y_pred = self.predict(X_test)
        Y_pred = np.where(Y_pred > 0.5, 1, 0)
        accuracy = np.mean(Y_pred == Y_test)
        return accuracy

    def predict(self, X_test):
        Y_pred = []
        for x_test in X_test:
            y_pred = x_test @ self.weights + self.bias
            y_pred = self.activation(y_pred)
            Y_pred.append(y_pred)
        return np.array(Y_pred)

    def evaluate(self, X_test, Y_test):
        loss = self.calculate_loss(X_test, Y_test, 'mse')
        accuracy = self.calculate_accuracy(X_test, Y_test)
        return loss, accuracy
