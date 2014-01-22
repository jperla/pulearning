#!/usr/bin/env python
import numpy

def checkgrad(func, x, params=[], e=1e-6):
    N = len(x)

    d = (2 * e * numpy.random.random_sample((N,))) - e

    _, gx = func(x, *params)

    x2, gx2 = func(x + d, *params)
    x1, gx1 = func(x - d, *params)

    r = (x2 - x1) / (2 * gx.dot(d) )
    return r

def checkgrad_random(func, D=5, params=[], scale=1e-6):
    for i in xrange(10):
        xr = scale * numpy.random.random_sample((D, ))
        print checkgrad(func, xr, params=params)

if __name__ == '__main__':
    def f(x):
        return numpy.sum(x ** 2), 2 * x

    x1 = numpy.array([3, 4, 5])
    print checkgrad(f, x1)

    checkgrad_random(f)

    # this one should be wrong
    def g(x):
        return numpy.sum(x ** 2), 3 * x
    x1 = numpy.array([3, 4, 5])
    print checkgrad(g, x1)

    checkgrad_random(g)


