import numpy as np
import scipy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.fixes import unique

import logistic

class SGDPosonlyMultinomialLogisticRegression(BaseEstimator, ClassifierMixin):
    """Posonly logistic regression.
    """

    def __init__(self, eta0=0.1, n_iter=5, c=None):
        self.eta0 = eta0
        self.n_iter = n_iter
        self.c = c

    def fit(self, X, y):
        self.classes_, indices = unique(y, return_inverse=True)
        self.minimumC_ = float(np.sum(y)) / len(y)
        self.q_ = (1.0 / (1.0 - self.minimumC_)) - 1.0
        self.b_, self.w_ = logistic.posonly_multinomial_logistic_gradient_descent(X, y, max_iter=self.n_iter, eta0=self.eta0, c=self.c)
        return self

    def predict(self, X):
        return np.array([t >=0.5  for t in self.predict_proba(X)[:,1]])

    def predict_proba(self, X):
        probas = []
        # TODO: speed up classification by working with sparse matrices
        if isinstance(X, scipy.sparse.csr.csr_matrix):
            X = np.array(X.todense())
        N = X.shape[0]
        X = np.hstack([np.ones(N).reshape((N, 1)), X])
        for r in range(N):
            logPL, logPU, logN, _ = logistic.posonly_multinomial_log_probabilities(self.w_.dot(X[r]), self.b_, self.q_)
            P = np.exp(logistic.logsumexp2(logPL, logPU))
            N = 1 - P
            try:
                N2 = np.exp(logN)
            except:
                pass
            else:
                assert abs(N2 - N) < 0.001
            assert abs(1.0 - (P + N)) < 0.001
            probas.append([N, P])
        return np.array(probas)

    def final_c(self):
        return (1.0 / (1.0 + self.q_ + np.exp(-1 * self.b_)))

class SGDLogisticRegression(BaseEstimator, ClassifierMixin):
    """Stochastic Gradient Descent version of logistic regression.
        Implemented in Cython.
    """

    def __init__(self, eta0=1.0, n_iter=5, alpha=0, learning_rate='default'):
        self.eta0 = eta0
        self.n_iter = n_iter
        self.alpha = alpha
        self.learning_rate = learning_rate

    def fit(self, X, y):
        self.classes_, indices = unique(y, return_inverse=True)
        self.theta_ = logistic.fast_logistic_gradient_descent(X, y, max_iter=self.n_iter, eta0=self.eta0, alpha=self.alpha, learning_rate=self.learning_rate)
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
        self.classes_, indices = unique(y, return_inverse=True)
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

    def __init__(self, alpha=0, n_iter=15000):
        self.alpha = alpha
        self.n_iter = n_iter

    def fit(self, X, y):
        self.classes_, indices = unique(y, return_inverse=True)
        self.theta_ = logistic.lbfgs_logistic_regression(X, y, alpha=self.alpha, n_iter=self.n_iter)
        return self

    def predict(self, X):
        return logistic.label_data(X, self.theta_, binarize=True)

    def predict_proba(self, X):
        a = logistic.label_data(X, self.theta_, binarize=False)
        return np.vstack([1.0 - a, a]).T

