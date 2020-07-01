import numpy as np
from numba import jit
from scipy.sparse import csc_matrix

@jit
def kntinsop(i, n, u, p, U):
    """Computes the single knot insertion operator
    :math:`A_{ij}` in the form of a sparse matrix.
    The knot vector remains unchanged!

    The operation is to be evalated as :math:`\\tilde{P}_{ij} = P_{ik} A_{kj}`

    Parameters:
        i (int) : knot span in which :math:`u` lies
        n (int)  : number of basis functions defined on the knot vector
        u (float) : evaluation point
        p (int)  : basis degree
        U (np.array(float)) : knot vector

    Returns:
        (float, (int, int)) : (values, (rows, cols))
    """

    rows, cols, vals = [], [], []

    # ones
    for k in range(0, i+1-p):
        cols.append(k)
        rows.append(k)
        vals.append(1.0)

    # corner cutting
    for k in range(i+1-p, i+1):
        alpha = (u - U[k]) / (U[k+p] - U[k])

        cols.append(k)
        rows.append(k-1)
        vals.append(1.0 - alpha)

        cols.append(k)
        rows.append(k)
        vals.append(alpha)

    # ones
    for k in range(i+1, n+1):
        cols.append(k)
        rows.append(k-1)
        vals.append(1.0)

    return (vals, (rows, cols))
