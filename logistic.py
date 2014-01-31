#!/usr/bin/env python

import numpy as np
import scipy.optimize
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
        theta is of size ((D+1)x1) (theta is 1 bigger for the constant)
       Returns logistic sigmoid of all points, a (Nx1) array.
       Also accepts binarize boolean, 
            which rounds everything to 1 or 0 if True.
    """
    assert((data.shape[1] + 1) == theta.shape[0])
    #data = prepend_column_of_ones(data)
    #s = logistic_sigmoid(np.dot(data, theta), normalizer)
    s = logistic_sigmoid(data.dot(theta[1:]) + theta[0], normalizer)
    if binarize:
        s = np.array([(1 if a >= 0.5 else 0) for a in s])
    return s


###########################################
# Standard Stochastic Logistic Regression 
###########################################

ETA0 = 0.1
MAX_ITER = 100


def prepend_and_vars(X):
    """Adds bias term column to X, returns new X, theta zeros, and
        new shape of X/theta.
    """
    # prepend col of ones for intercept term
    X = prepend_column_of_ones(X) 
    N,M = X.shape
    theta = np.zeros((M,)) + (1.0 / M)
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

DEFAULT_B = 1.0
    
def lbfgs_modified_logistic_regression(X, y, b=None):
    """Same as modified LR, but solved using lbfgs."""
    X, theta, N, M = prepend_and_vars(X)

    if b is None:
        fix_b, b = False, DEFAULT_B
    else:
        fix_b, b = True, b

    def f(w, g, X, y):
        """Accepts x, and g.  Returns value at x, and gradient at g.
        """
        b = w[0]
        theta = w[1:]
        value = np.sum(np.abs(y - (1.0 / (1.0 + (b ** 2) + X.dot(theta)))))
        # now fill in the g

        ewx = np.exp(-X.dot(theta))
        b2ewx = (b * b) + ewx
        p = ((y - 1.0) / b2ewx) + (1.0 / (1.0 + b2ewx))
        
        dLdw = (p * ewx).reshape((X.shape[0], 1)) * X

        if not fix_b:
            w[0] = np.sum(-2 * b * p)
        w[1:] = np.sum(dLdw, axis=0)
        return value
    import lbfgs
    w = np.hstack([np.array([b,]), theta])
    answer = lbfgs.fmin_lbfgs(f, w, args=(X, y,))
    theta, b = answer[1:], answer[0]
    return theta, b

def regularized_lcl_loss_function(w, X, y, alpha):
    """Accepts w (the set of parameters), 
        and X, the matrix of training data (NxD), 
            where N is the number of samples, 
                and D the number of dimensions (features).
        and y, the labels of the features in X.
        Returns value and the gradient of the loss function (LCL - regularization) at w.
    """
    N,D = X.shape
    assert y.shape[0] == N
    assert len(y) == N
    
    ewx = np.exp(-1 * X.dot(w))
    assert ewx.shape[0] == N
    assert len(ewx) == N

    p = (1.0 / (1.0 + ewx))
    assert p.shape[0] == N
    assert len(p) == N

    lcl = 0.0
    for i in xrange(N):
        if y[i] == 1:
            lcl += np.log(p[i])
        else:
            lcl += np.log(1 - p[i])

    value_regularization = alpha * np.sum(w ** 2)

    value = -lcl + value_regularization

    # now calculate the gradient
    t = (y - p)
    assert t.shape[0] == N
    assert len(t) == N

    gradient_regularization = 2 * alpha * w
    assert gradient_regularization.shape[0] == D
    assert len(gradient_regularization) == D
    gradient = -(t.reshape((1, N)).dot(X)).reshape((D,)) + gradient_regularization

    #w[:] = gradient[:]
    return value, gradient

def lbfgs_logistic_regression(X, y, alpha=0, n_iter=15000):
    """Solves a logistic regression optimization for the data X and labels y, solved using lbfgs.
       Returns the found parameters (adds intercept term, so should have one more than the number of columns in X).
    """
    X, w, N, M = prepend_and_vars(X)

    final_w, f, d = scipy.optimize.fmin_l_bfgs_b(regularized_lcl_loss_function, w, args=(X, y, alpha), m=1000, maxfun=n_iter)
    return final_w

def fast_modified_logistic_gradient_descent(X, S, max_iter=MAX_ITER, b=None, eta0=ETA0):
    """Same but uses Cython."""
    X, theta, N, M = prepend_and_vars(X)

    S = np.array(S, dtype=float)
    
    if b is None:
        fix_b, b = False, DEFAULT_B
    else:
        fix_b, b = True, b

    b = switch_array(X,
                lambda: clogistic.modified_logistic_regression(theta, X, S, N, M, eta0, max_iter, b, fix_b=fix_b),
                lambda: clogistic.sparse_modified_logistic_regression(theta, X, S, N, M, eta0, max_iter, b, fix_b=fix_b))


    return theta, b

def modified_logistic_gradient_descent(X, S, max_iter=MAX_ITER, b=None, eta0=ETA0, i=0):
    """Accepts same as logistic regression.
        Returns 2-tuple of weights theta, and also upper bound variable b.
        Returns (theta, b).

        See the paper "A Probabilistic Approach 
                to the Positive and Unlabeled Learning Problem by Jaskie."
    """
    X, theta, N, M = prepend_and_vars(X)
    
    if b is None:
        fix_b, b = False, DEFAULT_B
    else:
        fix_b, b = True, b

    assert S.shape[0] == N

    #TODO: jperla: can this be faster?
    for t in xrange(i, max_iter):
        l = eta0 / (1.0 + t)
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
            if not fix_b:
                b = b + ((l) * dLdb)

        print t, (1.0 / (1.0 + (b * b)))
        if t % 10 == 0:
            print t, (1.0 / (1.0 + (b * b)))
    return theta, b

def fast_logistic_gradient_descent(X, y, max_iter=MAX_ITER, eta0=ETA0, alpha=0, learning_rate='default'):
    """Computes same as logistic_gradient_descent(), but uses Cython module."""
    X, theta, N, M = prepend_and_vars(X)
    
    y = np.array(y, dtype=np.float)

    if isinstance(X, scipy.sparse.csr.csr_matrix):
        assert alpha == 0, 'sparse does not support regularization'
        assert learning_rate == 'default', 'sparse does not support different learning rates'
        clogistic.sparse_logistic_regression(theta, X, y, N, M, eta0, max_iter)
    elif isinstance(X, np.ndarray):
        assert alpha == 0, 'non-sparse does not support regularization'
        assert learning_rate == 'default', 'non-sparse does not support different learning rates'
        clogistic.logistic_regression(theta, X, y, N, M, eta0, max_iter, alpha, learning_rate)
    else:
        raise Exception("Unknown array datatype")

    return theta

def logistic_gradient_descent(X, y, max_iter=MAX_ITER, eta0=ETA0, i=0):
    """Accepts data X, an NxM matrix.
        Accepts y, an Nx1 array of binary values (0 or 1)
        Returns an Mx1 array of logistic regression parameters.

        Based on Andrew Ng's Matlab implementation: 
            http://cs229.stanford.edu/section/matlab/logistic_grad_ascent.m
    """
    X, theta, N, M = prepend_and_vars(X)
    
    for t in xrange(i, max_iter):
        l = eta0 / (1.0 + t)
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
    l = (eta0 / max(MAX_ITER, max_iter))
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

def posonly_multinomial_log_probabilities(wx, b, q):
    return clogistic.posonly_multinomial_log_probabilities(wx, b, q)

def logsumexp2(wx, b):
    return clogistic.logsumexp2(wx, b)

def posonly_multinomial_log_probability_of_label(wx, y, b):
    return clogistic.posonly_multinomial_log_probability_of_label(wx, y, b)

def posonly_multinomial_logistic_gradient_descent(X, y, max_iter=MAX_ITER, eta0=ETA0, c=None):
    """Accepts data X, an NxM matrix.
        Accepts y, an Nx1 array of binary values (0 or 1)
        Returns c and the weighted the parameter vectors.

        Based on Andrew Ng's Matlab implementation: 
            http://cs229.stanford.edu/section/matlab/logistic_grad_ascent.m
    """
    if c is None:
        fix_b = False
        b = 0.0
    else:
        fix_b = True
        minimumC = float(np.sum(y)) / len(y)
        q = (1.0 / (1.0 - minimumC)) - 1.0
        b = -np.log((1.0 / c) - 1.0 - q)
        print 'Fixing c == %s' % c

    X, theta, N, M = prepend_and_vars(X)
    S = np.array(y, dtype=float)
    return switch_array(X,
                lambda: slow_posonly_multinomial_logistic_gradient_descent(theta, X, S, N, M, eta0, max_iter, b, fix_b),
                lambda: clogistic.sparse_posonly_logistic_gradient_descent(theta, X, S, N, M, eta0, max_iter, b, fix_b)
    )


def slow_posonly_multinomial_logistic_gradient_descent(w, X, y, N, M, eta0=ETA0, max_iter=MAX_ITER, b=0.0, fix_b=False):
    for iteration in xrange(0, max_iter):
        alpha = eta0
        for r in xrange(N):
            x, label = X[r], y[r]

            #c = 1.0 / (1.0 + np.exp(b))
            #print r, iteration, x, b, c, w

            wx = w.dot(x)
            logPpl, logPpu, logPn, logZ = clogistic.posonly_multinomial_log_probabilities(wx, b, 0)

            # calculate w
            dw = 0.0
            if label == 0:
                dw += np.exp(logPn - clogistic.logsumexp2(logPpu, logPn))
            dw -= np.exp(logPn)

            # calculate b
            db = 0.0
            if label == 0:
                db += np.exp(logPpl - clogistic.logsumexp2(logPpu, logPn))

                # double checking, probably no longer needed, remove TODO
                db2 = np.exp(-1 * clogistic.logsumexp2(b, wx))
                if (abs(db - db2) > 0.0001):
                    print db, db2
                    assert db == db2
            db -= np.exp(logPpl)
             
            w += alpha * (dw * x)
            if not fix_b:
                b += alpha * db
       
        if iteration % 20 == 0:
            c = 1.0 / (1.0 + np.exp(b))
            ll =  np.sum(posonly_multinomial_log_probability_of_label(w.dot(X[r]), y[r], b) for r in xrange(N))
            print c, b, w
            print iteration, 'll: %s' % ll
    return b, w

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
       Returns a 2-tuple of the learned parameters, and estimators.
            First element is 3-tuple of learned LR params, modified LR, and b.
            Second element is 5-tuple of estimators, 
                    (e1, e2, e3, e1_hat, e4_hat) according to the paper.
    """
    X = vstack([pos_sample, unlabeled])
    y = hstack([np.array([1] * pos_sample.shape[0]),
                np.array([0] * unlabeled.shape[0]),])
    X, y = sklearn.utils.shuffle(X, y)

    print 'starting LR...'
    thetaR = fast_logistic_gradient_descent(X, y, max_iter=max_iter)
    print 'done LR...'
    print 'starting modified LR...'
    thetaMR, b = fast_modified_logistic_gradient_descent(X, y, max_iter=max_iter, eta0=0.01)
    #thetaMR, b = lbfgs_modified_logistic_regression(X, y)

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

    return ((thetaR, thetaMR, b), (e1, e2, e3, e1_hat, e4_hat))

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

                _, estimators = calculate_estimators(*data, max_iter=100)

                t = (pp, d.func_name, c,) + estimators
                print t
                table.append(t)

                #e1, e2, e3, e1_hat, e4_hat = estimators
                
    # save the table for graphing
    import jsondata
    jsondata.save('table.json', table)
        
