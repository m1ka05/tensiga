import numpy as np
from numba import njit, int64, float64, prange

@njit(float64[:](int64, int64, float64, int64, float64[:]))
def dbfun(i, n, u, p, U):
    """Evaluates derivatives up to :math:`n` th order of a single
    bspline basis function :math:`\\frac{\partial^n}{\partial u^n} B_{i,p}(u)`
    at the coordinate :math:`u`

    Parameters:
        i (int) : basis function index
        n (in) : highest derivative order
        u (float) : evaluation point
        p (int)  : basis function degree
        U (np.array(float)) : knot vector

    Returns:
        np.array(float) : scalar values of
        :math:`\\frac{\partial^k}{\partial u^k} B_{i,p}(u)`, :math:`k=0\ldots n`
    """
    m = U.size - 1
    ders = np.zeros(n+1)
    N = np.zeros((p+1, p+1))
    ND = np.zeros(n+1)

    # handle n > p, all derivatives n > p eq zero!
    if(n > p):
        n = p

    if u < U[i] or u >= U[i+p+1]:
        return ders

    for j in range(0, p+1):
        if u >= U[i+j] and u < U[i+j+1]:
            N[j][0] = 1
        else:
            N[j][0] = 0

    for k in range(1, p+1):
        if N[0][k-1] == 0:
            saved = 0
        else:
            saved = ((u-U[i]) * N[0][k-1]) / (U[i+k]-U[i])

        for j in range(0, p-k+1):
            Uleft = U[i+j+1]
            Uright = U[i+j+k+1]
            if N[j+1][k-1] == 0:
                N[j][k] = saved;
                saved = 0
            else:
                temp = N[j+1][k-1] / (Uright - Uleft)
                N[j][k] = saved + (Uright - u) * temp
                saved = (u - Uleft) * temp


    ders[0] = N[0][p]
    for k in prange(1, n+1):
        for j in range(0, k+1):
            ND[j] = N[j][p-k]
        for jj in range(1, k+1):
            if ND[0] == 0:
                saved = 0
            else:
                saved = ND[0] / (U[i+p-k+jj] - U[i])

            for j in range(0, k-jj+1):
                Uleft = U[i+j+1]
                Uright = U[i+j+p-k+jj+1]

                if ND[j+1] == 0:
                    ND[j] = (p-k+jj) * saved
                    saved = 0
                else:
                    temp = ND[j+1] / (Uright - Uleft)
                    ND[j] = (p-k+jj) * (saved - temp)
                    saved = temp


        ders[k] = ND[0]

    return ders


if __name__ == '__main__':
    from iga.fspan import fspan

    n = 3
    u = 5./2
    p = 2
    U = np.array([0,0,0,1,2,3,4,4,5,5,5], dtype=np.float64)

    print(dbfun(2, n, u, p, U))
    print(dbfun(3, n, u, p, U))
    print(dbfun(4, n, u, p, U))

