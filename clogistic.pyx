from __future__ import division

import numpy as np

cimport numpy as np
cimport cython

DTYPE = np.double
ctypedef np.double_t DTYPE_t

cdef inline double exp(double v): return (2.71828182845904523536 ** v)
cdef inline double sigmoid(double v): return 1.0 / (1.0 + exp(-v))

@cython.boundscheck(False)
@cython.wraparound(False)
def logistic_regression(np.ndarray[DTYPE_t, ndim=1] theta not None, 
                        np.ndarray[DTYPE_t, ndim=2] X not None, 
                        np.ndarray[DTYPE_t, ndim=1] y not None, 
                        int N, 
                        int M,
                        int max_iter, 
                        double lambda_, 
                        ):
    """Cython version of stochastic gradient descent of 
        logistic regression

        Accepts parameters theta which will be modified in place.
        Accepts max_iter number of times to loop.
        Accepts lambda_ learning rate double.
        Accepts X which is a numpy array, an (N,M) array
            and an array y which is an (N,1) aray and
            where N is the number of rows, and 
                  M is dimensionality of data.
    """
    cdef double wx, hx, z, x
    cdef int t, r, m
    for t in range(1, max_iter + 1):
        for r in range(N):
            wx = 0.0
            for m in range(M):
                x = X[r,m]
                if x > 0:
                    wx += x * theta[m]
            hx = sigmoid(wx)
            z = lambda_ * (y[r] - hx)
            for m in range(M):
                x = X[r,m]
                if x > 0:
                    theta[m] += z * x
    return theta

@cython.boundscheck(False)
@cython.wraparound(False)
def modified_logistic_regression(
                        np.ndarray[DTYPE_t, ndim=1] theta not None, 
                        np.ndarray[DTYPE_t, ndim=2] X not None, 
                        np.ndarray[DTYPE_t, ndim=1] S not None, 
                        int N, 
                        int M,
                        int max_iter,
                        double lambda_, 
                        double b,
                       ):
    """Cython version of stochastic gradient descent of 
        logistic regression

        Accepts parameters theta which will be modified in place.
        Accepts max_iter number of times to loop.
        Accepts lambda_ learning rate double.
        Accepts X which is a numpy array, an (N,M) array
            and an array y which is an (N,1) aray and
            where N is the number of rows, and 
                  M is dimensionality of data.
    """
    cdef double x, s, wx, ewx, b2ewx, p, dLdb, dLdw, pewx
    cdef int t, r, m

    for t in range(1, max_iter+1):
        for r in range(N):
            wx = 0.0
            for m in range(M):
                x = X[r,m]
                if x > 0:
                    wx += x * theta[m]

            ewx = exp(-wx)
            b2ewx = (b * b) + ewx

            p = ((S[r] - 1.0) / b2ewx) + (1.0 / (1.0 + b2ewx))

            dLdb = -2 * p * b
            b = b + (lambda_ * dLdb)

            pewx = p * ewx
            for m in range(M):
                x = X[r,m]
                if x > 0:
                    dLdw = pewx * x 
                    theta[m] += (lambda_ * dLdw)
    return b

