import numpy as np

class LogisticRegression:
    def __init__(self, lr, num_iter):
        self.lr = lr
        self.num_iter = num_iter

    def sigmoid_function(self, z):
        return np.array((1 / (1 + np.exp(-z))),dtype=np.longdouble)

    def fit(self, X, y):
        intercept = np.ones((X.shape[0], 1))
        X = np.concatenate((intercept, X), axis=1)

        # Initialising weights to 0
        self.weights = np.zeros(X.shape[1])
        i = 0
        while i< self.num_iter:
            z = np.dot(X, self.weights)
            h = self.sigmoid_function(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.weights = self.weights - self.lr * gradient
            i += 1

    def predict_prob(self, X):
        intercept = np.ones((X.shape[0], 1))
        X = np.concatenate((intercept, X), axis=1)
        return self.sigmoid_function(np.dot(X, self.weights))


    def predict(self, X, threshold):
        return self.predict_prob(X) >= threshold