import os

import numpy as np

import logistic
import swissprot


if __name__=='__main__':
    from swissprot import folder, filenames, max_key

    pos, neg, test_pos = (swissprot.read_sparse(os.path.join(folder, f), max_key) for f in filenames)

    np.save('data.pos.swissprot.npy', pos)
    np.save('data.neg.swissprot.npy', neg)
    np.save('data.test_pos.swissprot.npy', test_pos)
