import numpy as np


def is_symmetric(M, verbose=True, tolerance=8):
    """
    Test symmetry of sparse matrix
    Param:
        - verbose: prints values and indices of conflicting matrix entries
        - tolerance: decimal tolerance for equality
    """
    diffs_matrix = abs(M - M.T)
    diffs_matrix.data = np.round(diffs_matrix.data, tolerance) # decimales tolerance
    diffs_matrix.eliminate_zeros() # sets 0 as sparse 0
    if diffs_matrix.nnz == 0:
        return True
    if verbose:
        ii_nz, jj_nz = (diffs_matrix).nonzero() # nonzero row/col indices
        print("Conflicting entries:")
        for pair in zip(ii_nz, jj_nz):
            print(f"M[{pair}] = {M[pair]}")
    return False
