import datetime
import array
import argparse
import logging
from ctypes import Structure, c_int, c_double, sizeof
from os import path

import numpy as np
from scipy import sparse
from tqdm import tqdm

from utils.vocab import load_vocab
from utils.matrices import is_symmetric


logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)

class CREC(Structure):
    """c++ class to read triples (idx, idx, cooc) from GloVe binary file
    """
    _fields_ = [('idx1', c_int),
                ('idx2', c_int),
                ('value', c_double)]


class IncrementalCOOMatrix:
    """class to create scipy.sparse.coo_matrix
    Based on https://maciejkula.github.io/2015/02/22/incremental-construction-of-sparse-matrices/
    """

    def __init__(self, shape, dtype=np.double):
        self.dtype = dtype
        self.shape = shape
        self.rows = array.array('i') # int (np.int32)
        self.cols = array.array('i') # int (np.int32)
        self.data = array.array('I') # unsigned integer

    def append(self, i, j, v):
        m, n = self.shape
        if (i >= m or j >= n):
            raise Exception('Index out of bounds')
        self.rows.append(i)
        self.cols.append(j)
        self.data.append(v)

    def tocoo(self):
        rows = np.frombuffer(self.rows, dtype=np.int32)
        cols = np.frombuffer(self.cols, dtype=np.int32)
        data = np.frombuffer(self.data, dtype=self.dtype)
        return sparse.coo_matrix((data, (rows, cols)), shape=self.shape)


def build_cooc_matrix(vocab_file, cooc_file):
    """
    Build full coocurrence matrix from cooc. data in binary glove file and \
    glove vocab text file
    Row and column indices are numeric indices from vocab_file
    There must be (i,j) for every (j,i) such that C[i,j]=C[j,i]
    """
    str2idx, idx2str, str2count = load_vocab(vocab_file)
    V = max(str2idx.values())  # vocab size (largest word index)
    size_crec = sizeof(CREC)  # crec: structura de coocucrrencia en Glove
    C = IncrementalCOOMatrix((V+1, V+1), dtype=np.uint32)
    K = path.getsize(cooc_file) / size_crec # total de coocurrencias
    pbar = tqdm(total=K)
    # open bin file and store coocs in C
    with open(cooc_file, 'rb') as f:
        # read and overwrite into cr while there is data
        cr = CREC()
        while (f.readinto(cr) == size_crec):
            C.append(cr.idx1, cr.idx2, int(cr.value))
            pbar.update(1)
    pbar.close()
    return C.tocoo().tocsr()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('vocabfile', type=str)
    parser.add_argument('coocfile', type=str)
    parser.add_argument('outfile', type=str)

    args = parser.parse_args()

    logging.info("BUILDING MATRIX")
    C = build_cooc_matrix(args.vocabfile, args.coocfile)

    logging.info("CHECKING SYMMETRY")
    check_symmetry = is_symmetric(C, verbose=True)
    if check_symmetry:
        print("IT IS OK")
    else:
        logging.warning("""\nOJALDRE: the co-occurrence matrix is not symmetric. 
        According to tests, the higher value should be replaced with the lower 
        value. The matrix will be saved nonetheless.\n""")

    logging.info("SAVING MATRIX")
    sparse.save_npz(args.outfile, C)

    logging.info("DONE")


if __name__ == "__main__":
    main()
