#!/usr/bin/env python
from functools import partial

import scipy
import numpy as np

import logistic

def test_manual_standard_logistic_regression():
    X = np.array([[0, 4, 1],
                  [3, 0, 0],
                  [0, 0, 2]])
    y = np.array([0, 1, 0])
    Xsparse = scipy.sparse.csr_matrix(X)


    for i in xrange(40):
        theta = logistic.logistic_gradient_descent(X, y, max_iter=i, eta0=0.1)
        thetaFast = logistic.fast_logistic_gradient_descent(X, y, max_iter=i, eta0=0.1)

        thetaM, b = logistic.modified_logistic_gradient_descent(X, y, max_iter=i, b=0.0, eta0=0.1)
        assert b == 0.0
        thetaMFast, b = logistic.fast_modified_logistic_gradient_descent(X, y, max_iter=i, b=0.0, eta0=0.1)
        assert b == 0.0

        thetaSparse = logistic.fast_logistic_gradient_descent(Xsparse, y, max_iter=i, eta0=0.1)
        thetaMSparse, b = logistic.fast_modified_logistic_gradient_descent(Xsparse, y, max_iter=i, b=0.0, eta0=0.1)
        assert b == 0.0

        close = partial(np.allclose, rtol=0.01, atol=0.0001)
        '''
        if i == 0:
            assert np.allclose(theta, [0.01, 0.01, 0.01, 0.01])
        elif i == 1:
            answer = [0.0095, 0.0105, 0.0095, 0.009]
            assert close(theta, answer)
            theta[2] += 0.0003
            assert not close(theta, answer)
            theta[2] -= 0.0003
        elif i == 2:
            answer = [0.009,  0.011,  0.009,  0.008]
            assert close(theta, answer)
            assert not close(theta * 0.8, answer)
        elif i == 3:
            answer = [0.0085,  0.0115,  0.0085,  0.007]
            assert close(theta, answer)
            assert not close(theta + 0.0003, answer)
        '''

        assert np.allclose(theta, thetaFast)
        
        # and make sure that the modified version is a superset of normal one
        assert np.allclose(theta, thetaM)
        assert np.allclose(theta, thetaMFast)
        assert np.allclose(theta, thetaSparse)
        assert np.allclose(theta, thetaMSparse)


    for i in xrange(40):
        # make sure C version is same as other version
        thetaM, bM = logistic.modified_logistic_gradient_descent(X, y, max_iter=i, b=1.0, eta0=0.01)
        thetaMFast, bMFast = logistic.fast_modified_logistic_gradient_descent(X, y, max_iter=i, b=1.0, eta0=0.01)
        thetaMSparse, bMSparse = logistic.fast_modified_logistic_gradient_descent(Xsparse, y, max_iter=i, b=1.0, eta0=0.01)

        assert np.allclose(thetaM, thetaMFast)
        assert np.allclose(thetaM, thetaMSparse)
        assert not np.allclose(thetaM + 0.001, thetaMFast)
        assert bM == bMFast == bMSparse

        #TODO: jperla: hardcode some values in 

def test_end_to_end_logistic_regression():
    pos, neg = logistic.generate_well_separable(100, 0.50)

    #graph_pos_neg(pos, neg)

    X = logistic.vstack([pos, neg])
    y = logistic.hstack([np.array([1] * len(pos)), 
                   np.array([0] * len(neg)),])
    data = logistic.generate_random_points(100, 
                                           center=np.array([2,2]), 
                                           scale=np.array([5,5]))

    #theta = logistic.logistic_gradient_descent(X, y)

    thetaC = logistic.fast_logistic_gradient_descent(X, y)
    theta = thetaC
    #assert np.allclose(theta, thetaC)

    labels = logistic.label_data(data, theta, binarize=True)
    assert len([l for l in labels if l == 0]) > 10
    assert len([l for l in labels if l == 1]) > 10
    labels = logistic.label_data(data, thetaC, binarize=True)
    assert len([l for l in labels if l == 0]) > 10
    assert len([l for l in labels if l == 1]) > 10

    small_data = np.array([[-1, -1], [11, 11]])
    labels2 = logistic.label_data(small_data, theta, binarize=True)
    assert np.allclose([0, 1], labels2)
    assert not np.allclose([1, 1], labels2)
    labels2 = logistic.label_data(small_data, thetaC, binarize=True)
    assert np.allclose([0, 1], labels2)
    assert not np.allclose([1, 1], labels2)

    #TODO: jperla: test split_labeled_data()
    #graph_labeled_data(data, labels) 

