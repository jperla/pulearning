import numpy as np

import sparse

def test_simple():
    X = np.array([[0, 4, 1],
                  [0, 3, 0],
                  [2, 0, 0],
    ])

    lod = sparse.list_of_dicts_from_dense(X)
    assert lod == [{1: 4, 2: 1}, {1: 3}, {0: 2}]
    s = sparse.matrix_from_list_of_dicts(lod)

    assert isinstance(s, tuple)
    assert len(s) == 2
    assert np.allclose(s[0], [2, 1, 1])
    assert np.allclose(s[1], np.array([0,1,4,0,2,1,1,1,3,2,0,2,]))

    Y = sparse.to_dense(s)

    assert not sparse.is_sparse_matrix(X)
    assert sparse.is_sparse_matrix(s)
    assert not sparse.is_sparse_matrix(Y)

    assert sparse.shape(s) == X.shape == Y.shape
    assert np.allclose(X, Y)

