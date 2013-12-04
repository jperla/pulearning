import numpy as np
import logistic
from sklearn.base import BaseEstimator, ClassifierMixin

class SGDLogisticRegression(BaseEstimator, ClassifierMixin):
    """Stochastic Gradient Descent version of logistic regression.
        Implemented in Cython.
    """

    def __init__(self, alpha=1.0, n_iter=5):
        pass
        self.alpha = alpha
        self.n_iter = n_iter

    def fit(self, X, y):
        self.classes_, indices = np.unique(y, return_inverse=True)
        self.theta_ = logistic.fast_logistic_gradient_descent(X, y, max_iter=self.n_iter, alpha=self.alpha)
        return self

    def predict(self, X):
        return logistic.label_data(X, self.theta_, binarize=True)

    def predict_proba(self, X):
        return logistic.label_data(X, self.theta_, binarize=False)

class SGDModifiedLogisticRegression(BaseEstimator, ClassifierMixin):
    """Same as SGD Logistic Regression, but adds a b**2 value 
            which is learned in order to make the maximum
            probability 1.0.
        Implemented in Cython.
    """

    def __init__(self, alpha=1.0, n_iter=5):
        pass
        self.alpha = alpha
        self.n_iter = n_iter

    def fit(self, X, y):
        self.classes_, indices = np.unique(y, return_inverse=True)
        self.theta_, self.b_ = logistic.fast_modified_logistic_gradient_descent(X, y, max_iter=self.n_iter, alpha=self.alpha)
        return self

    def predict(self, X):
        return logistic.label_data(X, self.theta_, self.b_**2, binarize=True)

    def predict_proba(self, X):
        return logistic.label_data(X, self.theta_, self.b_**2, binarize=False)


