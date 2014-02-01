#!/usr/bin/env python

from matplotlib import pyplot
import sklearn
import sklearn.dummy
import numpy as np
import scipy

np.seterr(all='raise')

import lr
import logistic

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

    if 'FULL_GRAPH' in locals() and FULL_GRAPH:
        speed_multiple = 1
    else:
        speed_multiple = 10

    n_pos = 500 / speed_multiple
    mean_pos = [2, 2]
    cov_pos = [[1, 1], [1, 4]]

    n_neg = 1000 / speed_multiple
    mean_neg = [-2, -3]
    cov_neg = [[4, -1], [-1, 4]]

    gaussian = np.random.multivariate_normal

    optimal_points = []
    posonly_points = []
    naive_points = []
    
    if speed_multiple > 1:
        cs = [(0.1 * i) for i in xrange(1, 10)]
    else:
        cs = [(0.01 * i) for i in xrange(1, 100)]
        cs.extend([(0.99 + (0.001 * i)) for i in xrange(1, 10)])
        cs.extend([(0.001 * i) for i in xrange(1, 10)])

        '''
        cs = []
        cs.extend([(0.001 * i) for i in xrange(1, 50)])
        cs.extend([(0.001 * i) for i in xrange(1, 50)])
        '''

    for c in cs:
        positive, negative, positive_labeled, unlabeled = gen_sample(c, n_pos, n_neg)
        
        # skip data with no positive labels
        if positive_labeled.shape[0] < 3:
            continue

        X = np.vstack([positive_labeled, unlabeled])
        y = np.hstack([np.array([1] * positive_labeled.shape[0]),
                       np.array([0] * unlabeled.shape[0]),])
        X, y = sklearn.utils.shuffle(X, y)
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        X = scipy.sparse.csr_matrix(X)

        testX = np.vstack([positive, negative])
        testY = np.hstack([np.array([1] * positive.shape[0]),
                           np.array([0] * negative.shape[0]),])
        testX, testY = sklearn.utils.shuffle(testX, testY)
        testX = scaler.transform(testX)
        testX = scipy.sparse.csr_matrix(testX)

        optimalTrainX, optimalTrainY = testX, testY

        # generate independent test sample
        positive, negative, _, _ = gen_sample(c, n_pos, n_neg)
        trueTestX = np.vstack([positive, negative])
        trueTestY = np.hstack([np.array([1] * positive.shape[0]),
                               np.array([0] * negative.shape[0]),])
        trueTestX, trueTestY = sklearn.utils.shuffle(trueTestX, trueTestY)
        trueTestX = scaler.transform(trueTestX)
        #trueTestX = scipy.sparse.csr_matrix(trueTestX)
        testX, testY = trueTestX, trueTestY
        
        print 'c:', c

        n_iter = 1000

        posonly = lr.SGDPosonlyMultinomialLogisticRegression(n_iter=n_iter, eta0=0.1)
        posonly.fit(X, y)
        t = posonly.score(testX, testY)
        #t = sklearn.metrics.roc_auc_score(testY, posonly.predict_proba(testX)[:,1])
        posonly_points.append([c, t])
        print 'posonly:', t, 'c:', posonly.final_c()

        sgd_params = {'alpha':[1e-100, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
                      'loss':['log', 'hinge'],
                      'penalty':['l2', 'l1'],
                     }
        true_sgd = sklearn.linear_model.SGDClassifier(loss='log', alpha=1e-3)
        #true_sgd = sklearn.grid_search.GridSearchCV(true_sgd, sgd_params)
        true_sgd.fit(optimalTrainX, optimalTrainY)
        #print true_sgd.best_params_
        t = true_sgd.score(testX, testY)
        #t = sklearn.metrics.roc_auc_score(testY, true_sgd.predict_proba(testX)[:,1])
        optimal_points.append([c, t])
        print 'maximum:', t

        sgd = sklearn.linear_model.SGDClassifier(loss='log', alpha=1e-100)
        sgd = sklearn.grid_search.GridSearchCV(sgd, sgd_params)
        sgd.fit(X, y)
        #print sgd.best_params_
        t = sgd.score(testX, testY)
        '''
        try:
            t = sklearn.metrics.roc_auc_score(testY, sgd.predict_proba(testX)[:,1])
        except:
            t = 0.0
        '''
        naive_points.append([c, t])
        print 'naive sgd:', t

    optimal_points = np.array(sorted(optimal_points))
    posonly_points = np.array(sorted(posonly_points))
    naive_points = np.array(sorted(naive_points))

    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    ax.plot(posonly_points[:,0], posonly_points[:,1], 'b+--', label="POLR pos-only labels")
    ax.plot(optimal_points[:,0], optimal_points[:,1], 'go-', label="LR true labels")
    ax.plot(naive_points[:,0], naive_points[:,1], 'rx-', label="LR pos-only labels")

    ax.set_title('Comparing logistic regression on synthetic data')
    ax.set_xlabel('C')
    ax.set_ylabel('Test Accuracy')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=4)

    name = 'simulated'
    fig.savefig('pdf/%s.png' % name)
    if speed_multiple > 1:
        fig.savefig('pdf/%s-fast.png' % name)
    else:
        fig.savefig('pdf/%s-full.png' % name)

    if 'SUPPRESS_PLOT' not in locals() or not SUPPRESS_PLOT:
        fig.show()

