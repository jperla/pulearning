#!/usr/bin/env python

import numpy as np

import logistic


if __name__ == '__main__':
    cs = np.linspace(0.05, 1, 20)
    validation_fractions = [0.01, 0.05, 0.10, 0.30, 0.50, 1.0]

    n_pos = 500
    mean_pos = [2, 2]
    cov_pos = [[2, 1], [1, .5]]

    n_neg = 1000
    mean_neg = [-2, -3]
    cov_neg = [[2, 1], [1, 2]]

    gaussian = np.random.multivariate_normal


    print cs
    for c in cs:
        vf = 0.2
    #for vf in validation_fractions:

        def gen_sample(p, n):
            """Accepts two integers.
                Returns a new dataset of x,y gaussians plus 
                    x^2 and y^2 in a 2-tuple of 2 arrays; (p,4) and (n,4) 
            """
            def add_x2_y2(a):
                """Accepts an (N,2) array, adds 2 more columns
                    which are first col squared, second col squared.
                """
                return logistic.vstack([a.T, a[:,0]**2, a[:,1]**2]).T
            pos = gaussian(mean_pos, cov_pos, p)
            pos = add_x2_y2(pos)
            neg = gaussian(mean_neg, cov_neg, n)
            neg = add_x2_y2(neg)
            return logistic.sample_positive(c, pos, neg)

        pos_sample, unlabeled = gen_sample(n_pos, n_neg)
        # validation set:
        v_p, v_u = gen_sample(int(vf * n_pos), int(vf * n_neg))

        data = (pos_sample, unlabeled, v_p, v_u)
        #data, fixers = logistic.normalize_pu_data(*data)
        _, estimators = logistic.calculate_estimators(*data, max_iter=1000)

        t = ('vf:', vf, 'c:', c, ) + estimators
        print t


    #TODO: jperla: graph this!
