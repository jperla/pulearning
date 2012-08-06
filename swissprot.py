import os

import numpy as np

import logistic

def array_from_dict(d, max_key):
    """Accepts a dictionary, and an integer max_key or size of array.
        Returns a 1D array of size max_key, filled according to d.
    """
    a = np.zeros((max_key,))
    for k,v in d.iteritems():
        a[k] = v
    return a

def read_sparse(filename, max_key):
    """Accepts filename.
        Returns a dense array.
    """
    with open(filename, 'r') as f:
        lines = [p[3:].strip('\r\n ').split(' ') for p in f.readlines()]
        ds = [dict((int(q.split(':')[0]), int(q.split(':')[1])) for q in l) 
                    for l in lines[1:]]
        return np.array([array_from_dict(d, max_key) for d in ds])


# read in data
folder = 'proteindata'
filenames = ['P', 'N', 'Q']
max_key = 24081

npy_filenames = 'pos', 'neg', 'test_pos'

if __name__=='__main__':
    pos, neg, test_pos = (np.load(os.path.join(folder, 'data.%s.swissprot.npy' % d)) for d in npy_filenames)

    # set up data

    table = []
    for cp in [1.0, 0.5, 0.1, 0.7, 0.6, 0.4, 0.3, 0.2, 0.9, 0.8]:
        # split out the validation set separately
        split_half = lambda a: logistic.sample_split(a, len(a) / 2)
        half_pos, v_pos = split_half(pos)
        half_neg, v_neg = split_half(neg)
        half_test_pos, v_test_pos = split_half(test_pos)

        # figure out the subset to sample (c)
        u = np.vstack([half_neg, half_test_pos])
        pos_sample, unlabeled = logistic.sample_positive(cp, half_pos, u)

        # create validation set the same way
        u = np.vstack([v_neg, v_test_pos])
        v_p, v_u = logistic.sample_positive(cp, v_pos, u)

        estimators = logistic.calculate_estimators(pos_sample, unlabeled, v_p, v_u)

        t = (cp, 
         len(half_pos), len(half_neg), len(half_test_pos), 
         estimators,
         float(int(len(half_pos) * cp)) / (len(half_test_pos) + len(half_pos)),
        )
        table.append(t)

        print t

