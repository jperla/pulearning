#!/usr/bin/env python

from matplotlib import pyplot
import sklearn
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

    cs = [0.1,]
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
        testX = scaler.transform(testX)
        
        print 'starting LR...'
        posonly = lr.SGDPosonlyMultinomialLogisticRegression(n_iter=100, c=c)
        posonly.fit(X, y)
        print 'done LR...'
        print 'posonly tested:', posonly.score(testX, testY)

        print 'starting LR...'
        sgd = sklearn.linear_model.SGDClassifier(loss='log')
        sgd.fit(X, y)
        print 'done LR...'
        print 'sgd tested:', sgd.score(testX, testY)



        '''
        # unit area ellipse
        fig = pyplot.figure()
        ax = fig.add_subplot(111)
        ax.scatter(pos[:,0], pos[:,1], s=6, c='b', marker='+')
        ax.scatter(neg[:,0], neg[:,1], s=6, c='r', marker='o', lw=0)

        delta = 0.01
        x, y = np.arange(-8.0, 8.0, delta), np.arange(-10.0, 10.0, delta)
        X, Y = np.meshgrid(x, y)

        assert X.shape == Y.shape
        shape = X.shape

        data = np.hstack([X.flatten().reshape(-1, 1), Y.flatten().reshape(-1,1)])
        assert data.shape[0] == (shape[0] * shape[1]) and data.shape[1] == 2
        data = add_x2_y2(data)

        # plot the LR on the true labels
        labels = logistic.label_data(data, thetaTrue, normalizer=0.0, binarize=False)
        labels.shape = shape
        CS = pyplot.contour(X, Y, labels, [0.10,], colors='#0000FF')
        
        labels = logistic.label_data(data, theta, normalizer=0.0, binarize=False)
        labels.shape = shape
        CS = pyplot.contour(X, Y, labels, [0.10,], colors='#AAAAFF')

        labels = posonly.predict_proba(data)[:,0]
        labels.shape = shape
        CS = pyplot.contour(X, Y, labels, [0.10,], colors='#FF0000')
        #pyplot.clabel(CS, inline=1, fontsize=10)


        print 'b: ', b
        print 'c ~ ', 1.0 / (1.0 + b*b)
        labels = logistic.label_data(data, thetaM, normalizer=(b*b), binarize=False)
        labels.shape = shape
        CS = pyplot.contour(X, Y, labels, [0.10,], colors='r')
        #pyplot.clabel(CS, inline=1, fontsize=15)

        pyplot.title('Logistic regression on synthetic data')
        pyplot.show()
        '''