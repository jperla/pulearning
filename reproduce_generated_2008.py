#!/usr/bin/env python

from matplotlib import pyplot
import numpy as np

import logistic


def add_x2_y2(a):
    """Accepts an (N,2) array, adds 2 more columns
        which are first col squared, second col squared.
    """
    return logistic.vstack([a.T, a[:,0]**2, a[:,1]**2]).T

def gen_sample(p, n):
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
    validation_fractions = [0.01, 0.05, 0.10, 0.30, 0.50, 1.0]
    table = []

    n_pos = 500
    mean_pos = [2, 2]
    cov_pos = [[1, 1], [1, 4]]

    n_neg = 1000
    mean_neg = [-2, -3]
    cov_neg = [[4, -1], [-1, 4]]

    gaussian = np.random.multivariate_normal

    cs = [0.20,]
    print cs
    for c in cs:
        vf = 0.2
    #for vf in validation_fractions:


        pos, neg, pos_sample, unlabeled = gen_sample(n_pos, n_neg)
        # validation set:
        _, _, v_p, v_u = gen_sample(int(vf * n_pos), int(vf * n_neg))

        data = (pos_sample, unlabeled, v_p, v_u)
        #data, fixers = logistic.normalize_pu_data(*data)
        params, estimators = logistic.calculate_estimators(*data, max_iter=1000)
        theta, thetaM, b = params

        t = ('vf:', vf, 'c:', c, ) + estimators
        print t
        table.append(t)

        # run the LR on the true data
        (thetaTrue, _, _), _ = logistic.calculate_estimators(*(pos, neg, v_p, v_u), max_iter=1000)

        # unit area ellipse
        fig = pyplot.figure()
        ax = fig.add_subplot(111)
        ax.scatter(p[:,0], p[:,1], s=6, c='b', marker='+')
        ax.scatter(n[:,0], n[:,1], s=6, c='r', marker='o', lw=0)

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
        #pyplot.clabel(CS, inline=1, fontsize=10)


        print 'b: ', b
        print 'c ~ ', 1.0 / (1.0 + b*b)
        labels = logistic.label_data(data, thetaM, normalizer=(b*b), binarize=False)
        labels.shape = shape
        CS = pyplot.contour(X, Y, labels, [0.10,], colors='r')
        #pyplot.clabel(CS, inline=1, fontsize=15)

        pyplot.title('Logistic regression on synthetic data')
        pyplot.show()
