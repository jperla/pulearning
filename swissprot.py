import os

import numpy
import scipy.io
import sklearn.decomposition
import sklearn.preprocessing

import logistic


def normalize_pu_nonnegative_data(pos_sample, unlabeled, v_p, v_u):
    """Same as above but works for non-negative data
    """
    d = logistic.vstack([pos_sample, unlabeled])

    # decorrelater = sklearn.decomposition.PCA(whiten=False)
    decorrelater = sklearn.decomposition.NMF()
    #decorrelater.fit(d)

    transformer = sklearn.preprocessing.Scaler()
    transformer.fit(d)

    #fixer = lambda d: transformer.transform(decorrelater.transform(d))
    fixer = lambda d: transformer.transform(d)

    return ((fixer(pos_sample), fixer(unlabeled), fixer(v_p), fixer(v_u)), 
             (decorrelater, transformer, fixer))


# read in data
folder = 'proteindata'
filenames = ['P', 'N', 'Q']
max_key = 24081

mtx_filenames = 'pos', 'neg', 'test_pos'

if __name__=='__main__':
    pos, neg, test_pos = (scipy.io.mmread(os.path.join(folder, 'data.swissprot.%s.mtx' % d)) for d in mtx_filenames)
    numpy.save(os.path.join(folder, 'data.pos.swissprot.npy'), pos.todense())
    numpy.save(os.path.join(folder, 'data.neg.swissprot.npy'), neg.todense())
    numpy.save(os.path.join(folder, 'data.test_pos.swissprot.npy'), test_pos.todense())

    print 'read data...'

    table = []
    for cp in [1.0, 0.5, 0.1, 0.7, 0.6, 0.4, 0.3, 0.2, 0.9, 0.8]:
        # split out the validation set separately
        split = lambda a: logistic.sample_split(a, int(0.8 * a.shape[0]))
        half_pos, v_pos = split(pos)
        half_neg, v_neg = split(neg)
        half_test_pos, v_test_pos = split(test_pos)

        # figure out the subset to sample (c)
        u = logistic.vstack([half_neg, half_test_pos])
        pos_sample, unlabeled = logistic.sample_positive(cp, half_pos, u)

        # create validation set the same way
        u = logistic.vstack([v_neg, v_test_pos])
        v_p, v_u = logistic.sample_positive(cp, v_pos, u)

        print 'set up data...'

        data = (pos_sample, unlabeled, v_p, v_u)
        #data, fixers = normalize_pu_nonnegative_data(*data)
        print 'not-normalized...'
        #print 'normalized...'
        _, estimators = logistic.calculate_estimators(*data, max_iter=100)

        t = (cp, 
         half_pos.shape[0], half_neg.shape[0], half_test_pos.shape[0], 
         estimators,
         float(int(half_pos.shape[0] * cp)) / (half_test_pos.shape[0] + half_pos.shape[0]),
        )
        table.append(t)

        print t

