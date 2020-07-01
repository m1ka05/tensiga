import numpy as np
from scipy.sparse import csc_matrix, find
from scipy.sparse.linalg import spsolve
from tensiga.iga.gpnts import gpnts
from tensiga.iga.bfunsop import bfunsop

def twoscaleop(p, U, q, V):
    m = U.size - 1
    n = m - p - 1

    x = gpnts(q, V)

    Bold_data, Bold_shape = bfunsop(x, p, U)
    Bnew_data, Bnew_shape = bfunsop(x, q, V)


    Bold = csc_matrix(Bold_data, shape=Bold_shape)
    Bnew = csc_matrix(Bnew_data, shape=Bnew_shape)
    for row in range(0, n+1):
        rows, cols, vals = find(Bold[row,:])
        Bold[row,cols] = spsolve(Bnew[cols,:][:,cols].transpose(), vals)

    # clean-up
    tol_mask = np.array(Bold[Bold.nonzero()] < 1e2*np.finfo(np.float).eps)[0]
    rows = Bold.nonzero()[0][tol_mask]
    cols = Bold.nonzero()[1][tol_mask]
    Bold[rows, cols] = 0
    Bold.prune()

    return Bold


if __name__ == '__main__':
    from tensiga.iga.auxkv import auxkv

    p = 2
    U = np.repeat([0.0,1.0,2.0,3.0,4], [3,1,1,2,3])
    Z, M = auxkv(U)
    dp = 2

    q = p + dp
    V = np.repeat(Z, M+dp)

    T = twoscaleop(p,U,q,V)
