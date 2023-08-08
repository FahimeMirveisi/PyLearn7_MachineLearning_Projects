import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    
    def __init__(self, input_size, w_lr, b_lr, epochs):

        self.input_size = input_size
        self.w = np.random.rand(1, 1)
        print("init w:",self.w)
        self.b = np.random.rand(1, 1)
        print("init b:",self.b)
        self.w_lr = w_lr
        self.b_lr = b_lr
        self.epochs = epochs
        self.X_train_losses = []
        self.X_test_losses = []
        self.X_train_losses_epoch = []
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3)

    def plott(self, X_train, Y_train):

        self.ax1.clear()
        self.ax1.scatter(X_train, Y_train, color="blue")
        self.ax1.plot(X_train, X_train * self.w + self.b, color="red")
        self.ax1.set_xlabel('Length of abalon')
        self.ax1.set_ylabel('Height of abalon')
        self.ax1.set_title('Length and Height of abalone')

        self.ax2.clear()
        self.ax2.plot(self.X_train_losses, color="cyan")
        self.ax2.set_xlabel("fitting of each data")
        self.ax2.set_ylabel("Loss")
        self.ax2.set_title('Loss and fitting of each data')

        self.ax3.clear()
        self.ax3.plot(self.X_train_losses_epoch, color="yellow")
        self.ax3.set_xlabel("epochs")
        self.ax3.set_ylabel("Loss")
        self.ax3.set_title('Loss and epochs')

        plt.pause(0.0000001)

    def mse_loss(self, y_real , y_hat):

        self.error = y_real - y_hat
        loss = np.mean(self.error ** 2)

        return loss

    def SGD_update(self, error):

        self.w = self.w + (error * self.x * self.w_lr)
        #print(self.w)
        self.b = self.b + (error * self.b_lr)
        #print(self.b)

        return self.w, self.b

    def fit_and_losses(self, X_train, Y_train):
        
        
        
        for epoch in range(self.epochs):
            for i in range(X_train.shape[0]):
                self.x = X_train[i]
                #print(self.x)
                self.y = Y_train[i]
                #print(self.y)

                # vy_pred = x @ self.w + self.b
                self.y_pred = self.x * self.w + self.b
                loss = self.mse_loss(self.y, self.y_pred)
                self.X_train_losses.append(loss)
                final_w, final_b = self.SGD_update(self.error)
                
                #self.plott(X_train, Y_train)

            self.X_train_losses_epoch.append(loss)

        return self.X_train_losses, self.X_train_losses_epoch ,final_w, final_b


    def predict(self, X_train):

        Y_pred = X_train * self.w + self.b
        #print("Y_pred shape in predict:", Y_pred.shape)
        #print("Y_pred in predict:", Y_pred)

        return Y_pred

    def evaluate(self, X_test, Y_test):

        Y_pred = self.predict(X_test)
        #error = self.calculate_error(Y_test, Y_pred)
        X_test_losses = self.mse_loss(Y_test, Y_pred)
        #X_test_losses.append(loss)

        return X_test_losses

