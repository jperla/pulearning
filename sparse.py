import numpy as np

DTYPE = np.float

def is_sparse_matrix(matrix):
    """Accepts an input. 
        Returns False if it is a normal dense numpy array.
        Returns True if it is a 2-tuple of numpy arrays where the ... what?
    """
    if isinstance(matrix, np.ndarray):
        return False
    elif (isinstance(matrix, tuple) and 
          len(matrix) == 2 and
          isinstance(matrix[0], np.ndarray) and
          isinstance(matrix[1], np.ndarray) and
          matrix[0].ndim == 1 and
          matrix[1].ndim == 1):
        num_cells = matrix[0].sum()
        if (3 * num_cells) == len(matrix[1]):
            return True
        else:
            raise Exception('Invalid sparse matrix')
    else:
        raise Exception('Invalid matrix: neither sparse nor 2-tuple dense')

def matrix_from_dense(a):
    """Accepts dense matrix.
        Returns sparse 2-tuple matrix.
    """
    return matrix_from_list_of_dicts(list_of_dicts_from_dense(a))

def matrix_from_list_of_dicts(data):
        """Accepts list of dictionaries id:value.

            Returns a 2-tuple of an N-dimensional
                array of (i, j, value)  triplets flattened
                into a numpy array.
        """
        rows = len(data)
        counts = np.zeros((rows,), dtype=np.int)
        for r in range(rows):
            counts[r] = len(data[r])
        
        num_cells = counts.sum()
        cells = np.zeros((3 * num_cells,), dtype=DTYPE)
        m = 0
        for r in range(rows):
            d = list(sorted(data[r].iteritems()))
            for i in range(counts[r]):
                cells[m] = r
                m += 1
                cells[m] = d[i][0]
                m += 1
                cells[m] = d[i][1]
                m += 1
        assert m == (3 * num_cells)
        return (counts, cells)

def list_of_dicts_from_dense(a):
    """Accepts dense numpy array.
        Returns list of dictionaries, one dict per row.
            (ignoring 0-value cells for sparsity)
    """
    return [dict(d for d in enumerate(a[r]) if d[1] != 0)
                for r in xrange(a.shape[0])]
        
def shape(matrix):
    """Accepts 2-tuple sparse matrix.
        Returns 2-tuple of rows,columns of whole sparse matrix.
    """
    counts, cells = matrix
    N = len(counts)
    if len(cells) == 0:
        return (N,0)
    else:
        M = 1 + max(cells[1::3]) # middle of triplets
    return (N, M)

def iterate(matrix):
    """Accepts 2-tuple sparse matrix.
        Yields sequence of 
            3-tuples of indexes i,j and value at matrix[i,j].
    """
    counts, cells = matrix
    num_cells = counts.sum()
    for i in xrange(0, 3 * num_cells, 3):
        yield cells[i], cells[i+1], cells[i+2]

def to_dense(matrix):
    """Accepts 2-tuple sparse matrix.
        Returns numpy array 2D
    """
    N, M = shape(matrix)
    a = np.zeros((N, M), dtype=DTYPE)
    for i,j,v in iterate(matrix):
        a[i,j] = v
    return a
        
