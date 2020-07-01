import numpy as np
from tensiga.iga.fspan import fspan
from tensiga.iga.bfuns import bfuns
from scipy.sparse import csc_matrix
from numba.typed import List
from numba import njit

@njit
def bfunsop(u, p, U):
    """Computes values, rows and cols for an operator (a sparse matrix) of the form
    :math:`B_{ij}=B_{i,p}(u_j)`

    For perfomance and flexibility reasons the sparse matrix is to be
    constructed out of the return values,
    e.g. `scipy.sparse.csc_matrix(vals, rows, cols)`

    Parameters:
        u (np.array(float)) : evaluation point(s)
        p (int)  : basis function degree
        U (np.array(float)) : knot vector

    Returns:
        (float, (float, float)) : (values, (rows, cols))
    """
    nkts = U.size
    nbfuns = nkts - p - 1
    npts = u.size
    rows, cols, vals = [], [], []

    for j in range(0, npts):
        span = fspan(u[j], p, U)
        B_i = bfuns(span, u[j], p, U)

        for i in range(0, p+1):
            rows.append(span-p+i)
            cols.append(j)
            vals.append(B_i[i])

    shape = (nbfuns,npts)
    return (np.array(vals), (np.array(rows), np.array(cols))), shape

@njit
def bfunsmat(u, p, U):
    """Computes a matrix of the form :math:`B_{ij}`, where
    :math:`i=0\\ldots p` and for each :math:`j` th column the
    row :math:`i` of the matrix corresponds to the value of
    :math:`(\\mathrm{span}(u_j)-p+i)` th bspline basis function at
    :math:`u_j`.

    Parameters:
        u (np.array(float)) : evaluation point(s)
        p (int)  : basis function degree
        U (np.array(float)) : knot vector

    Returns:
        np.array(float) : matrix :math:`B_{ij}`
    """
    nkts = U.size
    nbfuns = nkts - p - 1
    npts = u.size

    Bij = np.zeros((nbfuns, npts))
    for j in range(0, npts):
        span = fspan(u[j], p, U)
        B_i = bfuns(span, u[j], p, U)

        for i in range(0, p+1):
            Bij[i,j] = B_i[i]

    return Bij


if __name__ == '__main__':
    import matplotlib.pylab as plt
    import scipy.sparse as sps

    u = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    p = 2
    U = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

    u = np.linspace(0,1,10)
    U = np.concatenate((
        np.repeat(0, p),
        np.linspace(0,1,5),
        np.repeat(1, p)), axis=None);

    (vals, (rows, cols)), sz = bfunsop(u,p,U)
    Bij = csc_matrix((vals, (rows, cols)), shape=sz)
    plt.spy(csc_matrix(Bij))
    plt.axis('equal')
    plt.show()
