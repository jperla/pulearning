#!/usr/bin/env python

import numpy as np

import pyximport; pyximport.install()
import clogistic

def label_data(data, theta, normalizer=0.0, binarize=True):
    """Accepts data array and theta parameters.
        data is of size (NxD).
        theta is of size (Dx1)
       Returns logistic sigmoid of all points, a (Nx1) array.
       Also accepts binarize boolean, 
            which rounds everything to 1 or 0 if Tru if True.
    """
    data = prepend_column_of_ones(data)
    s = logistic_sigmoid(np.dot(data, theta), normalizer)
    if binarize:
        s = np.array([(1 if a > 0.5 else 0) for a in s])
    return s




###########################################
# Standard Stochastic Logistic Regression 
###########################################


def prepend_and_vars(X):
    """Adds bias term column to X, returns new X, theta zeros, and
        new shape of X/theta.
    """
    # prepend col of ones for intercept term
    X = prepend_column_of_ones(X) 
    N,M = X.shape
    theta = np.zeros((M,), dtype=float)
    return X, theta, N, M

alpha = 0.1
max_iters = 100

def prepend_column_of_ones(X):
    """Accepts data array of points (NxD).
        Returns new array with 1's prepended as first column, (Nx(D+1))
    """
    N,M = X.shape
    return np.hstack([np.ones((N,1)), X])

def fast_modified_logistic_gradient_descent(X, S):
    """Same but uses Cython."""
    X, theta, N, M = prepend_and_vars(X)

    alpha = 0.01
    l = (alpha / max_iters)
    S = np.array(S, dtype=float)
    
    b = clogistic.modified_logistic_regression(theta, X, S, N, M, max_iters, l)

    return theta, b

def modified_logistic_gradient_descent(X, S):
    """Accepts same as logistic regression.
        Returns 2-tuple of weights theta, and also upper bound variable b.
        Returns (theta, b).

        See the paper "A Probabilistic Approach 
                to the Positive and Unlabeled Learning Problem by Jaskie."
    """
    X, theta, N, M = prepend_and_vars(X)
    alpha = 0.01
    max_iters = 100

    l = alpha / max_iters
    assert len(S) == N

    b = 1.0
    #TODO: jperla: can this be faster?
    for t in xrange(1, max_iters):
        for i in xrange(N):
            x = X[i,:]
            s = S[i]

            ewx = np.exp(-np.dot(x, theta))
            #assert isinstance(ewx, float)
            b2ewx = (b * b) + ewx
            #assert isinstance(b2ewx, float)

            p = ((s - 1.0) / b2ewx) + (1.0 / (1.0 + b2ewx))
            #assert isinstance(p, float)

            dLdw = (p * ewx) * x
            #assert dLdw.shape == (M+1,)

            dLdb = -2 * p * b
            #assert isinstance(dLdb, float)

            theta = theta + (l * dLdw)
            b = b + (l * dLdb)

        print t, (1.0 / (1.0 + (b * b)))
        if t % 10 == 0:
            print t, (1.0 / (1.0 + (b * b)))
    return theta, b



def fast_logistic_gradient_descent(X, y):
    """Computes same as below, but uses Cython module."""
    X, theta, N, M = prepend_and_vars(X)

    l = (alpha / max_iters)
    y = np.array(y, dtype=float)
    
    clogistic.logistic_regression(theta, X, y, N, M, max_iters, l)

    return theta

def logistic_gradient_descent(X, y):
    """Accepts data X, an NxM matrix.
        Accepts y, an Nx1 array of binary values (0 or 1)
        Returns an Mx1 array of logistic regression parameters.

        Based on Andrew Ng's Matlab implementation: 
            http://cs229.stanford.edu/section/matlab/logistic_grad_ascent.m
    """
    X, theta, N, M = prepend_and_vars(X)

    for t in xrange(1, max_iters + 1):
        l = (alpha / t)
        for r in xrange(N):
            hx = logistic_sigmoid(np.dot(X[r], theta))
            #assert isinstance(hx, float)
            theta += l * (y[r] - hx) * X[r]

        if t % 30 == 0:
            print t
            hx = logistic_sigmoid(np.dot(X, theta))
            ll =  np.sum((y * np.log(hx)) + ((1.0 - y) * np.log(1.0 - hx)))
            print 'll: %s' % ll

    '''
    l = (alpha / max_iters)
    for t in xrange(1, max_iters + 1):
        hx = logistic_sigmoid(np.dot(X, theta))
        assert hx.shape == (N,)
        theta += l * (1.0 / N) * np.dot((y-hx), X)

        if t == 1 or t % 500 == 0:
            ll =  np.sum((y * np.log(hx)) + ((1.0 - y) * np.log(1.0 - hx)))
            print t
            print 'll: %s' % ll
    '''

    return theta

def logistic_sigmoid(v, normalizer=0.0):
    """Returns 1 / (1 + n + e^(-v))"""
    return 1.0 / (1 + normalizer + np.exp(-v))

def generate_random_points(N, center=np.array([0,0]), scale=np.array([1,1])):
    """Accepts an integer N of number of points to generate.
        Also accepts a numpy array of center point.
        Also accepts a scale array of floats, 
            width and height of the random gaussian.
       Returns a 2D array of points of size (N x center.size)
    """
    points = np.random.normal(size=(N,center.size))
    return (points * scale) + center

def graph_pos_neg(pos, neg):
    """Accepts two 2D arrays of points.
        Shows a graph of the points.  First array in blue, second in red. 
    """
    from matplotlib.pyplot import figure, show

    # unit area ellipse
    fig = figure()
    ax = fig.add_subplot(111)
    ax.scatter(pos[:,0], pos[:,1], s=3, c='b', marker='x')
    ax.scatter(neg[:,0], neg[:,1], s=3, c='r', marker='x')
    show()

def split_labeled_data(data, labels):
    """Accepts data array of d-dimensional points (N x d) 
        Also accpets an array of labels (Nx1)
       The labels array label it as positive and negative (1 and 0).
       Returns two arrays of size (Pxd) and (Qxd) where P+Q=N
    """
    assert data.shape[0] == len(labels)
    p = 1 #np.max(labels)
    n = 0 #np.min(labels)

    assert 0 == len([m for m in labels if m not in [p, n]]), 'only pos/neg'

    #TODO: jperla: remove duplication here?
    pos = np.array([data[i,:] for i,label in enumerate(labels) if label == p])
    neg = np.array([data[i,:] for i,label in enumerate(labels) if label == n])
    
    assert pos.shape[0] + neg.shape[0] == data.shape[0]
    return pos, neg

def graph_labeled_data(data, labels):
    """Accepts same input as split_labeled_data()
       Shows graph of data with positive labeled points blue, negative red.
    """
    pos, neg = split_labeled_data(data, labels)
    graph_pos_neg(pos, neg)

def generate_pos_neg_points(N, proportion_positive, positive_center):
    """Assumes negative points will be centered around (0,0).
        Generates N points, where proportion_positive are positive points.
        Returns a 2-tuple of (pos, neg) points.
    """
    num_pos = int(N * proportion_positive)
    num_neg = N - num_pos
    pos = generate_random_points(num_pos, 
                                 center=positive_center, 
                                 scale=np.array([1.0, 1.5]))
    neg = generate_random_points(num_neg,
                                 scale=np.array([1.5, 1.5]))
    return pos, neg
    
def generate_well_separable(N, pp):
    """Returns 2 well-separated sets of points. No overlap at all."""
    return generate_pos_neg_points(N, pp, positive_center=np.array([10, 10]))
    
def generate_mostly_separable(N, pp):
    """Returns 2 mostly-separated sets of points. A tiny bit of overlap."""
    return generate_pos_neg_points(N, pp, positive_center=np.array([5, 5]))
    
def generate_some_overlap(N, pp):
    """Returns 2 close by sets of points. Some noticeable overlap."""
    return generate_pos_neg_points(N, pp, positive_center=np.array([2, 3]))

def generate_complete_overlap(N, pp):
    """Returns 2 sets of points with same centers. Total overlap."""
    return generate_pos_neg_points(N, pp, positive_center=np.array([0, 0]))

def sample_split(a, num_split):
    """Accepts an array.  
        Splits the data along the first axis of the array.
        Returns 2 arrays which can be vstack ed to form original a (scrambled).
    """
    a_scrambled = np.random.permutation(a)
    return a_scrambled[:num_split], a_scrambled[num_split:]

def sample_positive(c, pos, neg):
    """Accepts a proportion float c, and two arrays of points (NxD).
        Selects the fraction c of the pos points, completely at random.
        Returns a 2-tuple of arrays of points (PxD) and (QxD) where N=P+Q.
            The first part is the random selection of pos points.
            The second is the remaining pos points and negative points.
    """
    assert 0 < c <= 1
    num_sample = int(pos.shape[0] * c)

    pos_sample, remaining = sample_split(pos, num_sample)
    unlabeled = np.vstack([remaining, neg])
    assert pos_sample.shape[1] == unlabeled.shape[1]

    return pos_sample, unlabeled

def logistic_regression_from_pos_neg(pos, neg):
    """Accepts two arrays of NxD points.
        Learns parameters theta to separate two sets using logistic regression.
        Returns theta array of size (Dx1)
    """
    assert pos.shape[1] == neg.shape[1]
    X = np.vstack([pos, neg])
    y = np.hstack([np.array([1] * len(pos)),
                   np.array([0] * len(neg)),])
    theta = logistic_gradient_descent(X, y)
    return theta

def calculate_estimators(pos_sample, unlabeled, 
                         validation_pos_sample, validation_unlabeled):
    """Accepts Positive samples and unlabeled sets.
            Also accepts validation set equivalents.
       Returns 5-tuple of estimators, (e1, e2, e3, e1_hat, e4_hat)
            according to the paper.
    """
    X = np.vstack([pos_sample, unlabeled])
    y = np.hstack([np.array([1] * len(pos_sample)),
                   np.array([0] * len(unlabeled)),])

    # shuffle data so that it's nice and random
    total = np.hstack([X, y.reshape(y.shape[0], 1)])
    np.random.shuffle(total)
    X = total[:,:-1]
    y = total[:,-1]

    print 'starting LR...'
    thetaR = fast_logistic_gradient_descent(X, y)
    print 'done LR...'
    print 'starting modified LR...'
    thetaMR, b = fast_modified_logistic_gradient_descent(X, y)
    print 'done modified LR...'

    '''
    thetaR = logistic_gradient_descent(X, y)
    thetaMR, b = modified_logistic_gradient_descent(X, y)
    '''
    #thetaMR, b = thetaR, 0.0

    s = validation_pos_sample
    u = validation_unlabeled

    gR_s = label_data(s, thetaR, binarize=False)
    gR_V = label_data(np.vstack([s, u]), thetaR, binarize=False)
    e1 = sum(gR_s) / float(len(s))
    e2 = (sum(gR_s) / sum(gR_V))
    e3 = max(gR_V)

    gMR_s = label_data(s, thetaMR, (b * b), binarize=False)
    e1_hat = sum(gMR_s) / len(s)
    e4_hat = (1.0 / (1.0 + (b * b)))

    return e1, e2, e3, e1_hat, e4_hat

if __name__ == '__main__':
    pps = [0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4,]
    cs = [0.1, 0.3, 0.5, 0.7, 0.9,]
    pps = [0.5, 0.9, 0.1,]
    cs = [0.1, 0.5, 0.9,]

    dists = [generate_well_separable,
             generate_mostly_separable,
             generate_some_overlap,
             generate_complete_overlap,
    ]

    num_points = 10000

    table = []
    for pp in pps:
        for d in dists:
            pos, neg = d(num_points, pp)
            for c in cs:
                pos_sample, unlabeled = sample_positive(c, pos, neg)
                # validation set:
                v_p, v_u = sample_positive(c, *d(num_points, pp))
                #v_p, v_u = d(num_points, pp)

                estimators = calculate_estimators(pos_sample, unlabeled,
                                                  v_p, v_u)

                t = (pp, d.func_name, c, estimators)
                print t
                table.append(t)

                #e1, e2, e3, e1_hat, e4_hat = estimators
                
        
    '''
    pos, neg = generate_mostly_separable(num_points, pp)
    pos, neg = generate_complete_overlap(num_points, pp)
    pos, neg = generate_some_overlap(num_points, pp)
    pos, neg = generate_well_separable(num_points, pp)

    graph_pos_neg(pos, neg)
    graph_pos_neg(pos_sample, unlabeled)

    theta = logistic_regression_from_pos_neg(pos, neg)

    data = generate_random_points(10000,
                                  center=np.array([2,2]), 
                                  scale=np.array([10,10]))


    labelsR = label_data(data, thetaR, binarize=True)
    labelsMR = label_data(data, thetaMR, (b*b), binarize=True)

    print 'e1', e1
    print 'e2', e2
    print 'e3', e3
    print 'e1_hat', e1_hat
    print 'e4_hat', e4_hat

    # visually test that this works
    graph_labeled_data(data, labelsR)
    graph_labeled_data(data, labelsMR)
    '''

