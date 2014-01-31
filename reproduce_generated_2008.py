#!/usr/bin/env python

import scipy
from matplotlib import pyplot
import sklearn
import numpy as np

np.seterr(all='raise')

import lr
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

    if 'FULL_GRAPH' in locals() and FULL_GRAPH:
        speed_multiple = 1
    else:
        speed_multiple = 5

    n_pos = 500 / speed_multiple
    mean_pos = [2, 2]
    cov_pos = [[1, 1], [1, 4]]

    n_neg = 1000 / speed_multiple
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





        # pos only
        X = np.vstack([pos_sample, unlabeled])
        y = np.hstack([np.array([1] * pos_sample.shape[0]),
                        np.array([0] * unlabeled.shape[0]),])
        X, y = sklearn.utils.shuffle(X, y)
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        X = scipy.sparse.csr_matrix(X)

        posonly = lr.SGDPosonlyMultinomialLogisticRegression(n_iter=1000, eta0=0.01, c=None)
        posonly.fit(X, y)
        print 'posonly c:', posonly.final_c()


        pos_sample = scipy.sparse.csr_matrix(pos_sample)
        unlabeled = scipy.sparse.csr_matrix(unlabeled)

        testX = np.vstack([pos, neg])
        testy = np.hstack([np.array([1] * pos.shape[0]),
                            np.array([0] * neg.shape[0]),])
        scaler = sklearn.preprocessing.Scaler()
        scaler.fit(testX)
        testX = scaler.transform(testX)




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
        ax.scatter(pos[:,0], pos[:,1], s=6, c='b', marker='+')
        ax.scatter(neg[:,0], neg[:,1], s=6, c='r', marker='o', lw=0)

        delta = 0.01 * speed_multiple
        x, y = np.arange(-8, 8, delta), np.arange(-10, 10, delta)
        X, Y = np.meshgrid(x, y)

        assert X.shape == Y.shape
        shape = X.shape

        data = np.hstack([X.flatten().reshape(-1, 1), Y.flatten().reshape(-1,1)])
        assert data.shape[0] == (shape[0] * shape[1]) and data.shape[1] == 2
        data = add_x2_y2(data)
        scaled_data = scaler.transform(data)

        # plot the LR on the true labels
        labels = logistic.label_data(data, thetaTrue, normalizer=0.0, binarize=False)
        labels.shape = shape
        CS = pyplot.contour(X, Y, labels, [0.50,], colors='#0000FF')
        CS.collections[0].set_label('LR True Labels')
        
        labels = logistic.label_data(data, theta, normalizer=0.0, binarize=False)
        labels.shape = shape
        CS = pyplot.contour(X, Y, labels, [0.10,], colors='#AAAAFF')
        CS.collections[0].set_label('LR Pos-only Labels')

        labels = posonly.predict_proba(scaled_data)[:,1]
        labels.shape = shape
        CS = pyplot.contour(X, Y, labels, [0.50,], colors='#00FF00')
        CS.collections[0].set_label('POLR Pos-only Labels')

        print 'b: ', b
        print 'c ~ ', 1.0 / (1.0 + b*b)
        labels = logistic.label_data(data, thetaM, normalizer=(b*b), binarize=False)
        labels.shape = shape
        CS = pyplot.contour(X, Y, labels, [0.10,], colors='r')
        CS.collections[0].set_label('CLR Pos-only Labels')

        pyplot.title('Logistic regression on synthetic data')

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc=3)

        name = 'syntheticlr'
        fig.savefig('pdf/%s.png' % name)
        if speed_multiple > 1:
            fig.savefig('pdf/%s-fast.png' % name)
        else:
            fig.savefig('pdf/%s-full.png' % name)

        if 'SUPPRESS_PLOT' not in locals() or not SUPPRESS_PLOT:
            pyplot.show()
