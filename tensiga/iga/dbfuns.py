import numpy as np
from numba import njit, int64, float64, prange

@njit(float64[:,:](int64, int64, float64, int64, float64[:]))
def dbfuns(i, n, u, p, U):
    """Evaluates derivatives up to :math:`n` th order of all non-zero
    bspline basis functions :math:`\\frac{\partial^n}{\partial u^n} B_{j,p}(u)`
    at the coordinate :math:`u`, :math:`j=i-p\ldots i`,
    :math:`(n+1) \\times (p+1)` values.

    Parameters:
        i (int) : knot span in which :math:`u` lies
        n (in) : highest derivative order
        u (float) : evaluation point
        p (int)  : basis function degree
        U (np.array(float)) : knot vector

    Returns:
        np.array(float) : scalar values of
        :math:`\\frac{\partial^k}{\partial u^k} B_{j,p}(u)`,
        where dB[k,j] = :math:`\\frac{\partial^k}{\partial u^k} B_{j,p}(u)`.
        :math:`k=0\ldots n`
    """
    m = U.size - 1

    ndu = np.zeros((p+1, p+1))
    a = np.zeros((2,p+1))
    ders = np.zeros((n+1,p+1))
    left = np.zeros(p+1)
    right = np.zeros(p+1)

    ndu[0][0] = 1.
    for j in range(1, p+1):
        left[j] = u - U[i+1-j]
        right[j] = U[i+j] - u
        saved = 0.

        for r in range(0, j):
            ndu[j][r] = right[r+1] + left[j-r]
            temp = ndu[r][j-1] / ndu[j][r]
            ndu[r][j] = saved + right[r+1] * temp
            saved  = left[j-r] * temp

        ndu[j][j] = saved;

    for j in range(0, p+1):
        ders[0][j] = ndu[j][p]

    for r in range(0, p+1):
        s1 = 0
        s2 = 1
        a[0][0] = 1
        for k in range(1, n+1):
            d = 0;
            rk = r - k;
            pk = p - k;
            if r >= k:
                a[s2][0] = a[s1][0] / ndu[pk+1][rk]
                #d += a[s2][0] * ndu[rk][pk] # diff in C# code
                d = a[s2][0] * ndu[rk][pk]

            if rk >= -1:
                j1 = 1
            else:
                j1 = -rk

            if (r-1) <= pk:
                j2 = k - 1
            else:
                j2 = p - r

            for j in range(j1, j2+1):
                a[s2][j] = (a[s1][j] - a[s1][j-1]) / ndu[pk+1][rk+j]
                d += a[s2][j] * ndu[rk+j][pk]

            if r <= pk:
                a[s2][k] = -a[s1][k-1] / ndu[pk+1][r]
                d += a[s2][k] * ndu[r][pk];

            ders[k][r] = d;
            j = s1;
            s1 = s2;
            s2 = j;

    r = float(p);
    for k in range(1, n+1):
        for j in range(0, p+1):
            ders[k][j] *= r
        r *= (p - k)

    return ders


if __name__ == '__main__':
    from tensiga.iga.fspan import fspan
    from tensiga.iga.dbfun import *

    n = 3
    u = 5./2
    p = 2
    U = np.array([0,0,0,1,2,3,4,4,5,5,5], dtype=np.float64)
    i = fspan(u, p, U)

    print('Single call:')
    print(dbfuns(i, n, u, p, U))

    print('  ')
    print('One by one computation:')
    dB22 = dbfun(i-p+0, n, u, p, U)
    dB32 = dbfun(i-p+1, n, u, p, U)
    dB42 = dbfun(i-p+2, n, u, p, U)
    print(np.stack([dB22, dB32, dB42], axis=-1))
