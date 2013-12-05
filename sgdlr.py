import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

import logistic

class SGDLogisticRegression(BaseEstimator, ClassifierMixin):
    """Stochastic Gradient Descent version of logistic regression.
        Implemented in Cython.
    """

    def __init__(self, eta0=1.0, n_iter=5):
        pass
        self.eta0 = eta0
        self.n_iter = n_iter

    def fit(self, X, y):
        self.classes_, indices = np.unique(y, return_inverse=True)
        self.theta_ = logistic.fast_logistic_gradient_descent(X, y, max_iter=self.n_iter, eta0=self.eta0)
        return self

    def predict(self, X):
        return logistic.label_data(X, self.theta_, binarize=True)

    def predict_proba(self, X):
        a = logistic.label_data(X, self.theta_, binarize=False)
        return np.vstack([1.0 - a, a]).T

class SGDModifiedLogisticRegression(BaseEstimator, ClassifierMixin):
    """Same as SGD Logistic Regression, but adds a b**2 value 
            which is learned in order to make the maximum
            probability 1.0.
        Implemented in Cython.
    """

    def __init__(self, eta0=1.0, n_iter=5, b=None):
        pass
        self.eta0 = eta0
        self.n_iter = n_iter
        self.b = b

    def fit(self, X, y):
        self.classes_, indices = np.unique(y, return_inverse=True)
        self.theta_, self.b_ = logistic.fast_modified_logistic_gradient_descent(X, y, max_iter=self.n_iter, eta0=self.eta0, b=self.b)
        return self

    def predict(self, X):
        return logistic.label_data(X, self.theta_, self.b_**2, binarize=True)

    def predict_proba(self, X):
        a = logistic.label_data(X, self.theta_, self.b_**2, binarize=False)
        return np.vstack([1.0 - a, a]).T


