from __future__ import division

import logging
import functools

import numpy as np

cimport numpy as np
cimport cython
from cpython cimport bool

import sparse

DTYPE = np.double
ctypedef np.double_t DTYPE_t

cdef inline double exp(double v): return (2.71828182845904523536 ** v)
cdef inline double sigmoid(double v): return 1.0 / (1.0 + exp(-v))

cdef inline double MAX(double v1, double v2): 
    if v1 > v2:
        return v1
    else:
        return v2

from libc.math cimport exp, log

cdef inline double inline_logsumexp2(double v1, double v2):
    max = MAX(v1, v2)
    return log(exp(v1 - max) + exp(v2 - max)) + max

cdef inline double inline_logsumexp3(double v1, double v2, double v3):
    max = MAX(MAX(v1, v2), v3)
    return log(exp(v1 - max) + exp(v2 - max) + exp(v3 - max)) + max

cdef inline double inline_logsumexp4(double v1, double v2, double v3, double v4):
    max = MAX(MAX(MAX(v1, v2), v3), v4)
    return log(exp(v1 - max) + exp(v2 - max) + exp(v3 - max) + exp(v4 - max)) + max

def wrap_fast_cython(f):
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapped

@wrap_fast_cython
def logsumexp2(double v1, double v2):
    """Exposes fasterlogsumexp"""
    return inline_logsumexp2(v1, v2)
    
@wrap_fast_cython
def logsumexp3(double v1, double v2, double v3):
    """Exposes fasterlogsumexp"""
    return inline_logsumexp3(v1, v2, v3)

@wrap_fast_cython
def logsumexp4(double v1, double v2, double v3, double v4):
    """Exposes fasterlogsumexp"""
    return inline_logsumexp4(v1, v2, v3, v4)

@wrap_fast_cython
def logistic_regression(np.ndarray[DTYPE_t, ndim=1] theta not None, 
                        np.ndarray[DTYPE_t, ndim=2] X not None, 
                        np.ndarray[DTYPE_t, ndim=1] y not None, 
                        int N,
                        int M,
                        double eta0, 
                        int max_iter, 
                        double alpha,
                        str learning_rate,
                        ):
    """Cython version of stochastic gradient descent of 
        logistic regression

        Accepts parameters theta which will be modified in place.
        Accepts max_iter number of times to loop.
        Accepts eta0 learning rate double.
        Accepts alpha, the l2 regularization parameter.
        Accepts X which is a numpy array, an (N,M) array
            and an array y which is an (N,1) aray and
            where N is the number of rows, and 
                  M is dimensionality of data.
    """
    cdef double wx, hx, z, lambda_
    cdef int t, r, m
    for t in range(0, max_iter):
        if learning_rate == 'constant':
            lambda_ = eta0
        else:
            lambda_ = eta0 / (1.0 + t)    

        for r in range(N):
            wx = 0.0
            for m in range(M):
                wx += X[r,m] * theta[m]
            hx = sigmoid(wx)
            z = lambda_ * (y[r] - hx)
            for m in range(M):
                theta[m] += z * X[r,m]
                if m > 0: # do not regularize intercept term
                    theta[m] -= (alpha * 2 * lambda_ * theta[m])
    return theta

@wrap_fast_cython
def modified_logistic_regression(
                        np.ndarray[DTYPE_t, ndim=1] theta not None, 
                        np.ndarray[DTYPE_t, ndim=2] X not None, 
                        np.ndarray[DTYPE_t, ndim=1] S not None, 
                        int N, 
                        int M,
                        double eta0, 
                        int max_iter,
                        double b,
                        bool fix_b,
                       ):
    """Cython version of stochastic gradient descent of 
        logistic regression

        Accepts parameters theta which will be modified in place.
        Accepts max_iter number of times to loop.
        Accepts eta0 learning rate double.
        Accepts X which is a numpy array, an (N,M) array
            and an array y which is an (N,1) aray and
            where N is the number of rows, and 
                  M is dimensionality of data.
        Accepts fix_b which if True, will not change regularizer b with each update.
    """
    cdef double s, wx, ewx, b2ewx, p, dLdb, dLdw, pewx
    cdef double lambda_
    cdef int t, r, m

    for t in range(0, max_iter):
        lambda_ = eta0 / (1.0 + t)
        for r in range(N):
            wx = 0.0
            for m in range(M):
                wx += X[r,m] * theta[m]

            ewx = exp(-wx)
            b2ewx = (b * b) + ewx

            p = ((S[r] - 1.0) / b2ewx) + (1.0 / (1.0 + b2ewx))

            if not fix_b:
                dLdb = -2 * p * b
                b = b + ((lambda_) * dLdb)

            pewx = p * ewx
            for m in range(M):
                dLdw = pewx * X[r,m]
                theta[m] += (lambda_ * dLdw)
    return b



@wrap_fast_cython
def sparse_modified_logistic_regression(
                        np.ndarray[DTYPE_t, ndim=1] theta not None, 
                        object sparseX not None, 
                        np.ndarray[DTYPE_t, ndim=1] S not None, 
                        int N, 
                        int M,
                        double eta0, 
                        int max_iter,
                        double b,
                        bool fix_b,
                       ):
    """Same as non-sparse but uses a faster sparse matrix.
    """
    cdef double x, s, wx, ewx, b2ewx, p, dLdb, dLdw, pewx
    cdef double lambda_
    cdef long t, r, m
    cdef long c, d
    cdef double value
    cdef int param
    cdef long index

    cdef np.ndarray[DTYPE_t, ndim=1] data
    cdef np.ndarray[int, ndim=1] indices
    cdef np.ndarray[int, ndim=1] indptr

    data, indices, indptr = sparseX.data, sparseX.indices, sparseX.indptr

    for t in range(0, max_iter):
        lambda_ = eta0 / (1.0 + t)
        for r in range(N):
            wx = 0.0
            c = indptr[r]
            d = indptr[r+1]
            for index in range(c, d):
                param = indices[index]
                value = data[index]
                wx += value * theta[param]

            ewx = exp(-wx)
            b2ewx = (b * b) + ewx

            p = ((S[r] - 1.0) / b2ewx) + (1.0 / (1.0 + b2ewx))

            if not fix_b:
                dLdb = -2 * p * b
                b = b + ((lambda_) * dLdb)

            pewx = p * ewx
            for index in range(c, d):
                param = indices[index]
                value = data[index]
                dLdw = pewx * value 
                theta[param] += (lambda_ * dLdw)
    return b

@wrap_fast_cython
def posonly_multinomial_log_probabilities(double wx, double b, double q):
    """Accepts x (1xD) vector, and beta parameteri vectors, (c labeling constant, w_p the positive parameters and w_n the negative ones).
        Returns 3-tuple that sums to 1 of the log odds of positive labeled, positive unlabeled, and negative.
    """
    logZ = inline_logsumexp4(0, log(q), -b, wx)
    logPL = inline_logsumexp2(-b, log(q)) - logZ
    logPU = -logZ
    logN = wx - logZ
    return (logPL, logPU, logN, logZ)

@wrap_fast_cython
def posonly_multinomial_log_probability_of_label(wx, label, b, q):
    logPL, logPU, logN, logZ = posonly_multinomial_log_probabilities(wx, b, q)
    assert abs(logsumexp3(logPL, logPU, logN)) < 1e-7

    if label == 1:
        return logPL
    elif label == 0:
        return logsumexp2(logPU, logN)
    else:
        raise Exception("unknown label")

@wrap_fast_cython
def sparse_posonly_logistic_gradient_descent(
                        np.ndarray[DTYPE_t, ndim=1] theta not None, 
                        object sparseX not None, 
                        np.ndarray[DTYPE_t, ndim=1] S not None, 
                        int N, 
                        int M,
                        double eta0, 
                        int max_iter,
                        double b,
                        bool fix_b,
                       ):
    """Accepts data X, an NxM matrix.
        Accepts y, an Nx1 array of binary values (0 or 1)
        Returns c and the weighted the parameter vectors.

        Based on Andrew Ng's Matlab implementation: 
            http://cs229.stanford.edu/section/matlab/logistic_grad_ascent.m
    """
    cdef double x, s, wx, ewx, b2ewx, p, dLdb, dLdw, pewx
    cdef double lambda_
    cdef double calculated_c
    cdef long t, r, m
    cdef long c, d
    cdef double value
    cdef int param
    cdef long index
    cdef double ll
    cdef double minimumC, q

    cdef np.ndarray[DTYPE_t, ndim=1] data
    cdef np.ndarray[int, ndim=1] indices
    cdef np.ndarray[int, ndim=1] indptr
    
    data, indices, indptr = sparseX.data, sparseX.indices, sparseX.indptr

    minimumC = float(np.sum(S)) / len(S)
    q = (1.0 / (1.0 - minimumC)) - 1.0
    print 'minC:', minimumC
    print 'q:', q

    for t in range(0, max_iter):
        eta = eta0
        for r in range(N):
            # calculate wx
            wx = 0.0
            c = indptr[r]
            d = indptr[r+1]
            for index in range(c, d):
                param = indices[index]
                value = data[index]
                wx += value * theta[param]

            label = S[r]

            #print r, t, x, b, c, theta
            logPpl, logPpu, logPn, logZ = posonly_multinomial_log_probabilities(wx, b, q)

            # calculate dw
            dw = 0.0
            if label == 0:
                dw += exp(logPn - inline_logsumexp2(logPpu, logPn))
            dw -= exp(logPn)

            # calculate db
            db = 1.0
            if label == 1:
                db -= exp(-logPpl)
            db *= (exp(logPpl) - exp(log(q) - logZ))

            # debug: verify db
            '''
            db2 = 0.0
            expb = exp(-b)
            if label == 1:
                db2 += -1 * expb / (expb + q)
            db2 += (expb / (1 + q + expb + exp(wx)))
            print 'dbs:', db, db2
            assert abs(db - db2) < 0.001
            '''
             
            # update dw
            for index in range(c, d):
                param = indices[index]
                value = data[index]
                theta[param] += eta * (dw * value)
            # update b
            if not fix_b:
                b += eta * db
       
        if t % 20 == 0:
            ll = 0.0
            for r in range(N):
                # calculate wx
                wx = 0.0
                c = indptr[r]
                d = indptr[r+1]
                for index in range(c, d):
                    param = indices[index]
                    value = data[index]
                    wx += value * theta[param]
                ll +=  posonly_multinomial_log_probability_of_label(wx, S[r], b, q)
            calculated_c = 1.0 / (1.0 + exp(-b))
            logging.debug('c: %s, b: %s, theta: %s', calculated_c, b, theta)
            logging.debug('t: %s, ll: %s', t, ll)
    return b, theta

@wrap_fast_cython
def sparse_logistic_regression(np.ndarray[DTYPE_t, ndim=1] theta not None, 
                        object sparseX not None,
                        np.ndarray[DTYPE_t, ndim=1] y not None, 
                        int N, 
                        int M,
                        double eta0, 
                        int max_iter, 
                        ):
    """Same as non-sparse but uses a faster sparse matrix.
    """
    cdef double wx, hx, z, x
    cdef long t, r, m, c, d
    cdef double value
    cdef int param
    cdef long index

    cdef np.ndarray[DTYPE_t, ndim=1] data
    cdef np.ndarray[int, ndim=1] indices
    cdef np.ndarray[int, ndim=1] indptr

    data, indices, indptr = sparseX.data, sparseX.indices, sparseX.indptr
    index = 0
    for t in range(0, max_iter):
        lambda_  = eta0 / (1.0 + t)
        for r in range(N):
            wx = 0.0
            c = indptr[r]
            d = indptr[r+1]
            for index in range(c, d):
                param = indices[index]
                value = data[index]
                wx += value * theta[param]
            hx = sigmoid(wx)
            z = lambda_ * (y[r] - hx)
            for index in range(c, d):
                param = indices[index]
                value = data[index]
                theta[param] += z * value
    return theta

