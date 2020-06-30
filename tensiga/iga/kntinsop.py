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
    # refine the spline object
    u = np.linspace(0.1, 0.9, 9)
    v = u

    Au, U = ktsinsop(spline.nbfuns[0], u, spline.deg[0], spline.kv[0])
    Av, V = ktsinsop(spline.nbfuns[1], v, spline.deg[1], spline.kv[1])
                                        # compute: P_mn = P^x_ij A^u_im A^v_jn
    Px = spline.ctrlpts[0] @ Av         # P^x_ij A^v_jn = P_in
    Px = np.moveaxis(Px, 0, -1)         # P_in -> P_ni, noop
    Px = Px @ Au                        # P_ni A^u_im = P_nm
    Px = Px.transpose()                 # P_nm -> P_mn, (noop, works in 3d, too)

