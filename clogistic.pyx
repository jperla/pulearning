from __future__ import division

import numpy as np

cimport numpy as np
cimport cython

DTYPE = np.double
ctypedef np.double_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
def logistic_sigmoid(double v, double normalizer=0.0):
    """Returns 1 / (1 + e^(-v))"""
    return 1.0 / (1.0 + normalizer + (2.71828182845904523536 ** (-v)))


@cython.boundscheck(False)
@cython.wraparound(False)
def logistic_regression(np.ndarray[DTYPE_t, ndim=1] theta not None, 
                        np.ndarray[DTYPE_t, ndim=2] X not None, 
                        np.ndarray[DTYPE_t, ndim=1] y not None, 
                        int N, 
                        int M,
                        int max_iters, 
                        double lambda_, 
                        ):
    """Cython version of stochastic gradient descent of 
        logistic regression

        Accepts parameters theta which will be modified in place.
        Accepts max_iters number of times to loop.
        Accepts lambda_ learning rate double.
        Accepts X which is a numpy array, an (N,M) array
            and an array y which is an (N,1) aray and
            where N is the number of rows, and 
                  M is dimensionality of data.
    """
    cdef double wx, hx
    cdef int t, r, m
    for t in range(1, max_iters + 1):
        for r in range(N):
            wx = 0.0
            for m in range(M):
                wx += X[r,m] * theta[m]
            hx = logistic_sigmoid(wx)
            for m in range(M):
                theta[m] += lambda_ * (y[r] - hx) * X[r,m]
    return theta

@cython.boundscheck(False)
@cython.wraparound(False)
def modified_logistic_regression(
                        np.ndarray[DTYPE_t, ndim=1] theta not None, 
                        np.ndarray[DTYPE_t, ndim=2] X not None, 
                        np.ndarray[DTYPE_t, ndim=1] S not None, 
                        int N, 
                        int M,
                        int max_iters, 
                        double lambda_, 
                       ):
    """Cython version of stochastic gradient descent of 
        logistic regression

        Accepts parameters theta which will be modified in place.
        Accepts max_iters number of times to loop.
        Accepts lambda_ learning rate double.
        Accepts X which is a numpy array, an (N,M) array
            and an array y which is an (N,1) aray and
            where N is the number of rows, and 
                  M is dimensionality of data.
    """
    cdef double b = 1.0
    cdef double s, wx, ewx, b2ewx, p, dLdb
    cdef int t, r
    #TODO: jperla: can this be faster?
    for t in range(1, max_iters):
        for r in range(N):
            wx = 0.0
            for m in range(M):
                wx += X[r,m] * theta[m]

            #TODO: jperla: make exp an inline func
            ewx = (2.71828182845904523536 ** (-wx))
            b2ewx = (b * b) + ewx

            p = ((S[r] - 1.0) / b2ewx) + (1.0 / (1.0 + b2ewx))

            dLdb = -2 * p * b
            b = b + (lambda_ * dLdb)

            for m in range(M):
                dLdw = (p * ewx) * X[r,m]
                theta[m] += (lambda_ * dLdw)

    return b
