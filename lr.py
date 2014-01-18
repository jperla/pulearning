import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

import logistic

class SGDLogisticRegression(BaseEstimator, ClassifierMixin):
    """Stochastic Gradient Descent version of logistic regression.
        Implemented in Cython.
    """

    def __init__(self, eta0=1.0, n_iter=5, alpha=0):
        self.eta0 = eta0
        self.n_iter = n_iter
        self.alpha = alpha

    def fit(self, X, y):
        self.classes_, indices = np.unique(y, return_inverse=True)
        self.theta_ = logistic.fast_logistic_gradient_descent(X, y, max_iter=self.n_iter, eta0=self.eta0, alpha=self.alpha)
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

    def log_likelihood(self, X, y):
        def likelihood(x, s, theta, b):
            """Calculates the likelihood of one example"""
            assert x.shape[1] + 1 == theta.shape[0]
            ewx = np.exp(-1 * (x.dot(theta[1:]) + theta[0]))
            first_term = ((1.0) / (1.0 + (b * b) + ewx)) ** s
            second_term = (((b * b) + ewx) / (1.0 + (b * b) + ewx)) ** (1.0 - s)
            return first_term * second_term
        likelihoods = np.array([likelihood(X[i,:], y[i], self.theta_, self.b_) for i in range(X.shape[0])])
        return np.sum(np.log(likelihoods))

class LBFGSLogisticRegression(BaseEstimator, ClassifierMixin):
    """L-BFGS version of logistic regression.
    """

    def __init__(self, l2_regularization=0):
        self.l2_regularization = l2_regularization

    def fit(self, X, y):
        self.classes_, indices = np.unique(y, return_inverse=True)
        self.theta_ = lr.lbfgs_logistic_regression(X, y, l2_regularization=self.l2_regularization)
        return self

    def predict(self, X):
        return logistic.label_data(X, self.theta_, binarize=True)

    def predict_proba(self, X):
        a = logistic.label_data(X, self.theta_, binarize=False)
        return np.vstack([1.0 - a, a]).T

