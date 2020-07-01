import numpy as np
from scipy.sparse.linalg import spsolve_triangular

def sp_tri_solve_mat(A, B, axis=0, lower=True):
    if len(B.shape) == 1:
        return A.solve(B)

    B = np.moveaxis(B, axis, 0)
    Bshape = B.shape

    # solve along the first axis
    B = B.reshape(Bshape[0], np.prod(Bshape[1:]))
    C = spsolve_triangular(A, B, lower)

    C = C.reshape(Bshape)
    C = np.moveaxis(C, 0, axis)

    return C
