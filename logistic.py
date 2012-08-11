#!/usr/bin/env python

import numpy as np
import scipy.sparse
import sklearn
import sklearn.preprocessing
import sklearn.decomposition

import clogistic

def vstack(arrays):
    """Wrapper that switches between dense array stack and sparse."""
    if all(isinstance(a, np.ndarray) for a in arrays):
        return np.vstack(arrays)
    else:
        # remove empty arrays, since scipy has a problem with that
        arrays = [a for a in arrays if a.shape[0] > 0 and a.shape[1] > 0]
        return scipy.sparse.vstack(arrays, format=arrays[-1].getformat())

def hstack(arrays):
    """Wrapper that switches between dense array stack and sparse."""
    if all(isinstance(a, np.ndarray) for a in arrays):
        return np.hstack(arrays)
    else:
        # remove empty arrays, since scipy has a problem with that
        arrays = [a for a in arrays if a.shape[0] > 0 and a.shape[1] > 0]
        return scipy.sparse.hstack(arrays, format=arrays[-1].getformat())

def label_data(data, theta, normalizer=0.0, binarize=True):
    """Accepts data array and theta parameters.
        data is of size (NxD).
        theta is of size (Dx1)
       Returns logistic sigmoid of all points, a (Nx1) array.
       Also accepts binarize boolean, 
            which rounds everything to 1 or 0 if Tru if True.
    """
    #data = prepend_column_of_ones(data)
    #s = logistic_sigmoid(np.dot(data, theta), normalizer)
    s = logistic_sigmoid(data.dot(theta[1:]) + theta[0], normalizer)
    if binarize:
        s = np.array([(1 if a > 0.5 else 0) for a in s])
    return s




###########################################
# Standard Stochastic Logistic Regression 
###########################################

ALPHA = 0.1
MAX_ITER = 100


def prepend_and_vars(X):
    """Adds bias term column to X, returns new X, theta zeros, and
        new shape of X/theta.
    """
    # prepend col of ones for intercept term
    X = prepend_column_of_ones(X) 
    N,M = X.shape
    theta = np.zeros((M,)) + 0.01
    return X, theta, N, M

def prepend_column_of_ones(X):
    """Accepts data array of points (NxD).
        Returns new array with 1's prepended as first column, (Nx(D+1))
    """
    N,M = X.shape
    return hstack([np.ones((N,1)), X])

def switch_array(array, dense_func, sparse_func):
    """Accepts an array, and 2 functions (lambda maybe?) of what to do 
        if the array is a dense or sparse function.
       Returns the output of appropriate one
    """
    if isinstance(array, np.ndarray):
        return dense_func()
    elif isinstance(array, scipy.sparse.csr.csr_matrix):
        return sparse_func()
    else:
        raise Exception("Unknown array datatype")
    

def fast_modified_logistic_gradient_descent(X, S, max_iter=MAX_ITER, b=1.0, alpha=ALPHA):
    """Same but uses Cython."""
    X, theta, N, M = prepend_and_vars(X)

    S = np.array(S, dtype=float)
    
    b = switch_array(X,
                lambda: clogistic.modified_logistic_regression(theta, X, S, N, M, alpha, max_iter, b),
                lambda: clogistic.sparse_modified_logistic_regression(theta, X, S, N, M, alpha, max_iter, b))


    return theta, b

def modified_logistic_gradient_descent(X, S, max_iter=MAX_ITER, b=1.0, alpha=ALPHA, i=0):
    """Accepts same as logistic regression.
        Returns 2-tuple of weights theta, and also upper bound variable b.
        Returns (theta, b).

        See the paper "A Probabilistic Approach 
                to the Positive and Unlabeled Learning Problem by Jaskie."
    """
    X, theta, N, M = prepend_and_vars(X)

    assert S.shape[0] == N

    #TODO: jperla: can this be faster?
    for t in xrange(i, max_iter):
        l = alpha / (1.0 + t)
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
            b = b + ((l * 0.1) * dLdb)

        print t, (1.0 / (1.0 + (b * b)))
        if t % 10 == 0:
            print t, (1.0 / (1.0 + (b * b)))
    return theta, b



def fast_logistic_gradient_descent(X, y, max_iter=MAX_ITER, alpha=ALPHA):
    """Computes same as below, but uses Cython module."""
    X, theta, N, M = prepend_and_vars(X)
    
    y = np.array(y, dtype=np.float)

    if isinstance(X, scipy.sparse.csr.csr_matrix):
        clogistic.sparse_logistic_regression(theta, X, y, N, M, alpha, max_iter)
    elif isinstance(X, np.ndarray):
        clogistic.logistic_regression(theta, X, y, N, M, alpha, max_iter)
    else:
        raise Exception("Unknown array datatype")

    return theta

def logistic_gradient_descent(X, y, max_iter=MAX_ITER, alpha=ALPHA, i=0):
    """Accepts data X, an NxM matrix.
        Accepts y, an Nx1 array of binary values (0 or 1)
        Returns an Mx1 array of logistic regression parameters.

        Based on Andrew Ng's Matlab implementation: 
            http://cs229.stanford.edu/section/matlab/logistic_grad_ascent.m
    """
    X, theta, N, M = prepend_and_vars(X)
    
    for t in xrange(i, max_iter):
        l = alpha / (1.0 + t)
        for r in xrange(N):
            hx = logistic_sigmoid(np.dot(X[r], theta))
            #assert isinstance(hx, float)
            #import pdb; pdb.set_trace()
            theta += l * (y[r] - hx) * X[r]

        if t % 30 == 0:
            print t
            hx = logistic_sigmoid(np.dot(X, theta))
            ll =  np.sum((y * np.log(hx)) + ((1.0 - y) * np.log(1.0 - hx)))
            print 'll: %s' % ll

    '''
    l = (alpha / max(MAX_ITER, max_iter))
    for t in xrange(1, max_iter + 1):
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
    return 1.0 / (1.0 + normalizer + np.exp(-v))

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
    a_scrambled = sklearn.utils.shuffle(a)
    if num_split == a_scrambled.shape[0]:
        return a_scrambled, np.zeros((0, a.shape[1]))
    else:
        return a_scrambled[:num_split], a_scrambled[num_split:]
    #return sklearn.cross_validation.train_test_split(a, test_size=num_split)

def sample_positive(c, pos, neg):
    """Accepts a proportion float c, and two arrays of points (NxD).
        Selects the fraction c of the pos points, completely at random.
        Returns a 2-tuple of arrays of points (PxD) and (QxD) where N=P+Q.
            The first part is the random selection of pos points.
            The second is the remaining pos points and negative points.
    """
    assert 0 < c <= 1
    num_sample = int(pos.shape[0] * c)

    #pos_sample, remaining = sklearn.cross_validation.train_test_split(pos, test_size=c)
    pos_sample, remaining = sample_split(pos, num_sample)
    unlabeled = vstack([remaining, neg])
    assert pos_sample.shape[1] == unlabeled.shape[1]

    return pos_sample, unlabeled

def logistic_regression_from_pos_neg(pos, neg):
    """Accepts two arrays of NxD points.
        Learns parameters theta to separate two sets using logistic regression.
        Returns theta array of size (Dx1)
    """
    assert pos.shape[1] == neg.shape[1]
    X = vstack([pos, neg])
    y = hstack([np.array([1] * pos.shape[0]),
                   np.array([0] * neg.shape[0]),])
    theta = logistic_gradient_descent(X, y)
    return theta

def calculate_estimators(pos_sample, unlabeled, 
                         validation_pos_sample, validation_unlabeled,
                         max_iter=100):
    """Accepts Positive samples and unlabeled sets.
            Also accepts validation set equivalents.
            Also accepts maximum number of iterations for regression.
       Returns 5-tuple of estimators, (e1, e2, e3, e1_hat, e4_hat)
            according to the paper.
    """
    X = vstack([pos_sample, unlabeled])
    y = hstack([np.array([1] * pos_sample.shape[0]),
                np.array([0] * unlabeled.shape[0]),])
    X, y = sklearn.utils.shuffle(X, y)

    print 'starting LR...'
    thetaR = fast_logistic_gradient_descent(X, y, max_iter=max_iter)
    print 'done LR...'
    print 'starting modified LR...'
    thetaMR, b = fast_modified_logistic_gradient_descent(X, y, max_iter=max_iter, alpha=0.01)
    print 'done modified LR...'

    s = validation_pos_sample
    u = validation_unlabeled

    print 'calculating estimators...'

    gR_s = label_data(s, thetaR, binarize=False)
    gR_V = label_data(vstack([s, u]), thetaR, binarize=False)
    e1 = sum(gR_s) / float(s.shape[0])
    e2 = (sum(gR_s) / sum(gR_V))
    e3 = max(gR_V)

    print 'mid done...'

    gMR_s = label_data(s, thetaMR, (b * b), binarize=False)
    e1_hat = sum(gMR_s) / s.shape[0]
    e4_hat = (1.0 / (1.0 + (b * b)))

    print 'done calculating estimators...'

    return e1, e2, e3, e1_hat, e4_hat

def normalize_pu_data(pos_sample, unlabeled, v_p, v_u):
    """Accepts positive data, unlabeled, and validation arrays.
        1. Removes the mean (recenters to zero). 
        2. Decorrelates Input (PCA, ICA, or NNMF)
        3. Sets variance back to 1.

        Returns a 2-tuple of a 4-tuple of the cleaned up data,
                         and a 3-tuple of 2 objects used to clean, 
                                                    plus fixer func.
    """
    d = vstack([pos_sample, unlabeled])

    recenterer = sklearn.preprocessing.Scaler(with_std=False)
    d = recenterer.fit_transform(d)

    # does not work with ICA for some reason
    # decorrelater = sklearn.decomposition.FastICA(whiten=True)
    decorrelater = sklearn.decomposition.PCA(whiten=True)
    decorrelater.fit(d)

    fixer = lambda d: decorrelater.transform(recenterer.transform(d))

    return ((fixer(pos_sample), fixer(unlabeled), fixer(v_p), fixer(v_u)), 
             (recenterer, decorrelater, fixer))



if __name__ == '__main__':
    pps = [0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4,]
    cs = [0.1, 0.3, 0.5, 0.7, 0.9,]
    cs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,]

    pps = [0.5, 0.9, 0.1,]
    cs = [0.5, 0.1, 0.9, 0.01,]

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

                data = (pos_sample, unlabeled, v_p, v_u)
                #data, fixers = normalize_pu_data(*data)

                estimators = calculate_estimators(*data, max_iter=100)

                t = (pp, d.func_name, c,) + estimators
                print t
                table.append(t)

                #e1, e2, e3, e1_hat, e4_hat = estimators
                
    # save the table for graphing
    import jsondata
    jsondata.save('table.json', table)
        

