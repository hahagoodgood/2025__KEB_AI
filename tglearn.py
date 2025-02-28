import numpy as np

class LinearRegression:
    def __init__(self):
        self.slope = None # Weight(가중치)
        self.intercept = None # Bias(편향)

    def fit(self, X, y):
        """
        learning function
        :param x: independent variable (2d array format)
        :param y: dependent variable (2d array format)
        :return: void
        """
        X_mean = np.mean(X)
        y_mean = np.mean(y)

        denominator = np.sum(pow(X-X_mean, 2))
        numerator = np.sum((X-X_mean)*(y-y_mean))

        # 기울기 Weight
        self.slope = numerator / denominator 
        
        #Bias(편향)
        self.intercept = y_mean - (self.slope * X_mean)

    def predict(self, X) -> np.ndarray:
        """
        predict value for input
        :param X: new indepent variable
        :return: predict value for input (2d array format)
        """

        return self.slope * np.array(X) + self.intercept
