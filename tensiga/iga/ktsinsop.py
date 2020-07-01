import numpy as np
from scipy.sparse import csc_matrix, eye
from tensiga.iga.fspan import fspan
from tensiga.iga.kntinsop import kntinsop

def ktsinsop(n, u, p, U):
    """Computes the multi-knot insertion operator
    :math:`A_{ij}` in the form of a sparse matrix, which is returned together
    with the new knot vector.

    The operation is to be evalated as :math:`\\tilde{P}_{ij} = P_{ik} A_{kj}`

    Parameters:
        n (int)  : number of basis functions defined on the knot vector
        u (np.array(float)) : evaluation points
        p (int)  : basis degree
        U (np.array(float)) : knot vector

    Returns:
        scipy.sparse.csc_matrix() : operator :math:`A_{ij}`
        np.array(float) : updated knot vector
    """

    A = eye(n, format='csc')
    for ui in u:
        i = fspan(ui, p, U)
        A = A * csc_matrix(kntinsop(i, n, ui, p, U))
        n += 1
        U = np.insert(U, i+1, ui)

    return A, U
