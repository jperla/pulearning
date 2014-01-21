from __future__ import division

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

def wrap_fast_cython(f):
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapped


@wrap_fast_cython
def logistic_regression(np.ndarray[DTYPE_t, ndim=1] theta not None, 
                        np.ndarray[DTYPE_t, ndim=2] X not None, 
                        np.ndarray[DTYPE_t, ndim=1] y not None, 
                        int N,
                        int M,
                        double eta0, 
                        int max_iter, 
                        double alpha,
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
        lambda_ = eta0 / (1.0 + t)    
        for r in range(N):
            wx = 0.0
            for m in range(M):
                wx += X[r,m] * theta[m]
            hx = sigmoid(wx)
            l2_norm = 0
            z = lambda_ * (y[r] - hx)
            for m in range(M):
                theta[m] += z * X[r,m] - (alpha * 2 * lambda_ * theta[m])
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

