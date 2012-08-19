import os

import numpy as np

import logistic

# read in data
folder = 'proteindata'
filenames = ['P', 'N', 'Q']
max_key = 24081

npy_filenames = 'pos', 'neg', 'test_pos'

if __name__=='__main__':
    pos, neg, test_pos = (np.load(os.path.join(folder, 'data.%s.swissprot.npy' % d)) for d in npy_filenames)

    print 'read data...'

    # set up data

    table = []
    for cp in [1.0, 0.5, 0.1, 0.7, 0.6, 0.4, 0.3, 0.2, 0.9, 0.8]:
        # split out the validation set separately
        split_half = lambda a: logistic.sample_split(a, len(a) / 2)
        half_pos, v_pos = split_half(pos)
        half_neg, v_neg = split_half(neg)
        half_test_pos, v_test_pos = split_half(test_pos)

        # figure out the subset to sample (c)
        u = logistic.vstack([half_neg, half_test_pos])
        pos_sample, unlabeled = logistic.sample_positive(cp, half_pos, u)

        # create validation set the same way
        u = logistic.vstack([v_neg, v_test_pos])
        v_p, v_u = logistic.sample_positive(cp, v_pos, u)

        print 'set up data...'

        _, estimators = logistic.calculate_estimators(pos_sample, unlabeled, v_p, v_u)

        t = (cp, 
         len(half_pos), len(half_neg), len(half_test_pos), 
         estimators,
         float(int(len(half_pos) * cp)) / (len(half_test_pos) + len(half_pos)),
        )
        table.append(t)

        print t

