#!/usr/bin/env python

import numpy as np
from itertools import izip

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

def prepend_column_of_ones(X):
    """Accepts data array of points (NxD).
        Returns new array with 1's prepended as first column, (Nx(D+1))
    """
    N,M = X.shape
    return np.hstack([np.ones((N,1)), X])

def modified_logistic_gradient_descent(X, S):
    """Accepts same as logistic regression.
        Returns 2-tuple of weights theta, and also upper bound variable b.
        Returns (theta, b).

        See the paper "A Probabilistic Approach 
                to the Positive and Unlabeled Learning Problem by Jaskie."
    """
    max_iters = 100
    alpha = 0.01 / max_iters
    N,M = X.shape

    assert len(S) == N

    # prepend col of ones for intercept term
    X = prepend_column_of_ones(X) 

    theta = np.zeros((M+1,))
    b = 1.0
    for t in xrange(1, max_iters):
        for i in xrange(N):
            x = X[i,:]
            s = S[i]

            ewx = np.exp(-np.dot(x, theta))
            assert isinstance(ewx, float)
            b2ewx = (b * b) + ewx
            assert isinstance(b2ewx, float)

            p = ((s - 1) / b2ewx) + (1 / (1 + b2ewx))
            assert isinstance(p, float)

            dLdw = (p * ewx) * x
            assert dLdw.shape == (M+1,)

            dLdb = -2 * p * b
            assert isinstance(dLdb, float)

            theta = theta + (alpha * dLdw)
            b = b + (alpha * dLdb)
        print t
    return theta, b

def logistic_gradient_descent(X, y):
    """Accepts data X, an NxM matrix.
        Accepts y, an Nx1 array of binary values (0 or 1)
        Returns an Mx1 array of logistic regression parameters.

        Based on Andrew Ng's Matlab implementation: 
            http://cs229.stanford.edu/section/matlab/logistic_grad_ascent.m
    """
    max_iters = 1000
    alpha = 0.01 / max_iters
    N,M = X.shape

    # prepend col of ones for intercept term
    X = prepend_column_of_ones(X) 

    theta = np.zeros((M+1,))
    for t in xrange(1, max_iters):
        hx = logistic_sigmoid(np.dot(X, theta))
        assert hx.shape == (N,)
        theta = theta + (alpha * np.dot((y-hx), X))
        print t
    return theta

def logistic_sigmoid(v, normalizer=0.0):
    """Returns 1 / (1 + e^(-v))"""
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

def sample_positive(c, pos, neg):
    assert 0 < c <= 1
    num_sample = int(pos.shape[0] * c)
    pos_scrambled = np.random.permutation(pos)

    pos_sample = pos_scrambled[:num_sample]
    unlabeled = np.vstack([pos_scrambled[num_sample:], neg])
    assert pos_sample.shape[1] == unlabeled.shape[1]

    # shuffle to make it more random
    #np.random.shuffle(unlabeled)

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


if __name__ == '__main__':
    pp = 0.60
    num_points = 10000
    c = 0.50

    pos, neg = generate_mostly_separable(num_points, pp)
    pos, neg = generate_complete_overlap(num_points, pp)
    pos, neg = generate_some_overlap(num_points, pp)
    pos, neg = generate_well_separable(num_points, pp)

    pos_sample, unlabeled = sample_positive(c, pos, neg)

    #graph_pos_neg(pos, neg)
    #graph_pos_neg(pos_sample, unlabeled)

    theta = logistic_regression_from_pos_neg(pos, neg)

    X = np.vstack([pos_sample, unlabeled])
    y = np.hstack([np.array([1] * len(pos_sample)),
                   np.array([0] * len(unlabeled)),])
    thetaR = logistic_gradient_descent(X, y)
    thetaMR, b = modified_logistic_gradient_descent(X, y)

    e2 = ( sum(label_data(pos_sample, thetaR, binarize=False)) / 
                sum(label_data(np.vstack([pos_sample, unlabeled]),
                                         thetaR,
                                         binarize=False)) )
    e4 = (1 / (1 + (b * b)))

    data = generate_random_points(10000,
                                  center=np.array([2,2]), 
                                  scale=np.array([10,10]))

    labelsR = label_data(data, thetaR, binarize=True)
    labelsMR = label_data(data, thetaMR, (b*b), binarize=True)

    print 'e2', e2
    print 'e4', e4

    # visually test that this works
    graph_labeled_data(data, labelsR)
    graph_labeled_data(data, labelsMR)

