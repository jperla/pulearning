#!/usr/bin/env python

from matplotlib import pyplot
import sklearn
import sklearn.dummy
import numpy as np

np.seterr(all='raise')

import lr
import logistic

def calculate_posonly(c, pos_sample, unlabeled, max_iter=100):
    """Accepts Positive samples and unlabeled sets.
            Also accepts validation set equivalents.
            Also accepts maximum number of iterations for regression.
       Returns a 2-tuple of the learned parameters, and estimators.
            First element is 3-tuple of learned LR params, modified LR, and b.
            Second element is 5-tuple of estimators, 
                    (e1, e2, e3, e1_hat, e4_hat) according to the paper.
    """
    X = np.vstack([pos_sample, unlabeled])
    y = np.hstack([np.array([1] * pos_sample.shape[0]),
                np.array([0] * unlabeled.shape[0]),])
    X, y = sklearn.utils.shuffle(X, y)
    X = sklearn.preprocessing.normalize(X, axis=0)
    
    print 'starting LR...'
    posonly = lr.SGDPosonlyMultinomialLogisticRegression(n_iter=max_iter, c=c)
    posonly.fit(X, y)
    print 'done LR...'
    return posonly


def add_x2_y2(a):
    """Accepts an (N,2) array, adds 2 more columns
        which are first col squared, second col squared.
    """
    return logistic.vstack([a.T, a[:,0]**2, a[:,1]**2]).T

def gen_sample(c, p, n):
    """Accepts two integers.
        Returns a new dataset of x,y gaussians plus 
            x^2 and y^2 in a 2-tuple of 2 arrays; (p,4) and (n,4) 
    """
    pos = gaussian(mean_pos, cov_pos, p)
    pos = add_x2_y2(pos)
    neg = gaussian(mean_neg, cov_neg, n)
    neg = add_x2_y2(neg)
    return (pos, neg,) + logistic.sample_positive(c, pos, neg)

if __name__ == '__main__':
    cs = np.linspace(0.05, 1, 20)
    table = []

    n_pos = 500
    mean_pos = [2, 2]
    cov_pos = [[1, 1], [1, 4]]

    n_neg = 1000
    mean_neg = [-2, -3]
    cov_neg = [[4, -1], [-1, 4]]

    gaussian = np.random.multivariate_normal

    cs = [0.1, 0.2, 0.5, 0.9, 0.01, 0.05]
    for c in cs:
        positive, negative, positive_labeled, unlabeled = gen_sample(c, n_pos, n_neg)
        X = np.vstack([positive_labeled, unlabeled])
        y = np.hstack([np.array([1] * positive_labeled.shape[0]),
                       np.array([0] * unlabeled.shape[0]),])
        X, y = sklearn.utils.shuffle(X, y)
        scaler = sklearn.preprocessing.Scaler()
        scaler.fit(X)
        X = scaler.transform(X)

        testX = np.vstack([positive, negative])
        testY = np.hstack([np.array([1] * positive.shape[0]),
                           np.array([0] * negative.shape[0]),])
        testX, testY = sklearn.utils.shuffle(testX, testY)
        testX = scaler.transform(testX)
        
        print 'c:', c

        posonly = lr.SGDPosonlyMultinomialLogisticRegression(n_iter=50, c=c, eta0=0.1)
        posonly.fit(X, y)
        print 'posonly:', posonly.score(testX, testY)

        true_sgd = sklearn.linear_model.SGDClassifier(loss='log')
        true_sgd.fit(testX, testY)
        print 'maximum:', true_sgd.score(testX, testY)

        sgd = sklearn.linear_model.SGDClassifier(loss='log')
        sgd.fit(X, y)
        print 'naive sgd:', sgd.score(testX, testY)

        dumb = sklearn.dummy.DummyClassifier(strategy='most_frequent',random_state=0)
        dumb.fit(X, y)
        print dumb.score(testX, testY)

        dumb = sklearn.dummy.DummyClassifier(strategy='stratified',random_state=0)
        dumb.fit(X, y)
        print dumb.score(testX, testY)

