import os

import numpy as np
import scipy.sparse
import scipy.io

import jsondata

def array_from_dict(d, max_key):
    """Accepts a dictionary, and an integer max_key or size of array.
        Returns a 1D array of size max_key, filled according to d.
    """
    a = np.zeros((max_key,))
    for k,v in d.iteritems():
        a[k] = v
    return a

def read_sparse(filename):
    """Accepts filename string.
        Returns list of dictionaries.
    """
    with open(filename, 'r') as f:
        lines = [p[3:].strip('\r\n ').split(' ') for p in f.readlines()]
        ds = [dict((int(q.split(':')[0]), int(q.split(':')[1])) for q in l) 
                    for l in lines]
        return ds

def dense_from_read_sparse(filename, max_key):
    """Accepts filename and max_key integer.
        Returns a dense array.
    """
    ds = read_sparse(filename)
    return np.array([array_from_dict(d, max_key) for d in ds])

if __name__=='__main__':
    from swissprot import folder, filenames, max_key

    pos, neg, test_pos = (read_sparse(os.path.join(folder, f)) for f in filenames)

    '''
    jsondata.save('data.pos.swissprot.json', pos)
    jsondata.save('data.neg.swissprot.json', neg)
    jsondata.save('data.test_pos.swissprot.json', test_pos)
    '''
    def fill_sparse_matrix(list_of_dicts, sparse_matrix):
        """Accepts list of dicts, where keys are column indices,
                                    values cell values.
           Accepts sparse matrix.
           Fills the sparse matrix.
        """
        for i, d in enumerate(list_of_dicts):
            for j,value in d.iteritems():
                sparse_matrix[i, int(j)] = value

    sparse_pos = scipy.sparse.lil_matrix((len(pos), max_key), dtype=np.float)
    fill_sparse_matrix(pos, sparse_pos)
    sparse_neg = scipy.sparse.lil_matrix((len(neg), max_key), dtype=np.float)
    fill_sparse_matrix(neg, sparse_neg)
    sparse_test_pos = scipy.sparse.lil_matrix((len(test_pos), max_key), dtype=np.float)
    fill_sparse_matrix(test_pos, sparse_test_pos)

    scipy.io.mmwrite(os.path.join(folder, 'data.swissprot.pos'), sparse_pos)
    scipy.io.mmwrite(os.path.join(folder, 'data.swissprot.neg'), sparse_neg)
    scipy.io.mmwrite(os.path.join(folder, 'data.swissprot.test_pos'), sparse_test_pos)

