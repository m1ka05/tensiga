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


if __name__ == '__main__':
    from iga.Bspline import Bspline
    from iga.bfunsop import bfunsop
    from iga.fspan import fspan

    dim = 2;
    codim = 3;
    deg = [2, 1]

    kv = [ np.array([0., 0., 0., 1., 1., 1.]),
           np.array([0., 0., 1., 1.])]

    ctrlpts = [np.array([[1.0, 0.6],
                         [1.0, 0.6],
                         [0.0, 0.0]]), # x

               np.array([[0.0, 0.0],
                         [1.0, 0.6],
                         [1.0, 0.6]]), # y

               np.array([[0.0, 0.0],
                         [0.0, 0.0],
                         [0.0, 0.0]])] # z

    # init bspline object
    spline = Bspline(dim, codim, kv, deg, ctrlpts)


    ## knot insertion
    u = np.linspace(0.1, 0.9, 9)
    n = spline.nbfuns[0]
    Au = ktsinsop(n, u, spline.deg[0], spline.kv[0])

    print(Au @ ctrlpts[0])
