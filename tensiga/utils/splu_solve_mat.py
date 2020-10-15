import numpy as np

def splu_solve_mat(A, B, axis=-1, trans='N'):
    if len(B.shape) == 1:
        return A.solve(B)

    B = np.moveaxis(B, axis, 0)
    Bshape = B.shape

    # solve along the first axis
    B = B.reshape(Bshape[0], np.prod(Bshape[1:]))
    A = A.solve(B, trans)

    A = A.reshape(Bshape)
    A = np.moveaxis(A, 0, axis)

    return A
