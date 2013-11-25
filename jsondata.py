"""
    jsondata helps you read data files encoded in json.
    Copyright (C) 2011 Joseph Perla

    GNU Affero General Public License. See <http://www.gnu.org/licenses/>.
"""
import json
import numpy

def save_data(filename, data):
    """Accepts filenamestring and a list of objects, probably dictionaries.
        Writes these to a file with each object pickled using json on each line.
    """
    with open(filename, 'w') as f:
        if isinstance(data, dict):
            data = data.copy()
            for k,v in data.iteritems():
                if isinstance(v, numpy.ndarray):
                    data[k] = v.tolist()
            f.write(json.dumps(data))
        else:
            for i,d in enumerate(data):
                if i != 0:
                    f.write('\n')
                f.write(json.dumps(d))

def read_data(filename):
    """Accepts filename string.
        Reads filename line by line and unpickles from json each line.
        Returns generator of objects.
    """
    with open(filename, 'r') as f:
        for r in f.readlines():
            yield json.loads(r)

read = lambda f: list(read_data(f)) # read_data function is deprecated
save = lambda f,d: save_data(f,d)
