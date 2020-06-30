import numpy as np
from scipy.sparse import csc_matrix
from numba.typed import List
from numba import njit, prange
from tensiga.iga.fspan import fspan
from tensiga.iga.dbfuns import dbfuns

@njit
def dbfunsop(n, u, p, U):
    """Computes values, rows and cols for an operator (a sparse matrix) of the form
    :math:`dB_{ij}=\\partial_{u_j} B_{i,p}(u_j)`

    For perfomance and flexibility reasons the sparse matrix is to be
    constructed out of the return values,
    e.g. `scipy.sparse.csc_matrix(vals, rows, cols)`

    Parameters:
        n (int)  : derivative order
        u (np.array(float)) : evaluation point(s)
        p (int)  : basis function degree
        U (np.array(float)) : knot vector

    Returns:
        (float, (int, int)) : (values, (rows, cols))
    """
    nkts = U.size
    nbfuns = nkts - p - 1
    npts = u.size

    rows, cols, vals = [], [], []

    for j in range(0, npts):
        span = fspan(u[j], p, U)
        dB_i = dbfuns(span, n, u[j], p, U)

        for i in range(0, p+1):
            rows.append(span-p+i)
            cols.append(j)
            vals.append(dB_i[n, i])

    shape = (nbfuns, npts)
    return (np.array(vals), (np.array(rows), np.array(cols))), shape

@njit
def dbfunsmat(n, u, p, U):
    """Computes a matrix of the form :math:`dB_{ij}`, where
    :math:`i=0\\ldots p` and for each :math:`j` th column the
    row :math:`i` of the matrix corresponds to the value of
    :math:`(\\mathrm{span}(u_j)-p+i)` th bspline basis function
    :math:`n` th derivative at :math:`u_j`.

    Parameters:
        n (int)  : derivative order
        u (np.array(float)) : evaluation point(s)
        p (int)  : basis function degree
        U (np.array(float)) : knot vector

    Returns:
        np.array(float) : matrix :math:`dB_{ij}`
    """
    nkts = U.size
    nbfuns = nkts - p - 1
    npts = u.size

    dBij = np.zeros((p+1, npts))
    for j in range(0, npts):
        span = fspan(u[j], p, U)
        dB_i = dbfuns(span, n, u[j], p, U)

        for i in range(0, p+1):
            dBij[i,j] = dB_i[n, i]

    return dBij

if __name__ == '__main__':
    import matplotlib.pylab as plt
    import scipy.sparse as sps

    n = 0
    u = 5./2
    p = 2
    U = np.array([0,0,0,1,2,3,4,4,5,5,5], dtype=np.float64)

    u = np.array([5./2])
    Bij_csc = csc_matrix(dbfunsop(n, u, p, U))
    Bij_dense = dbfunsmat(n, u, p, U)
    plt.spy(csc_matrix(Bij_csc))
    plt.axis('equal')
    plt.show()
