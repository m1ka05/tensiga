import numpy as np
from numba import njit, int64, float64

@njit(float64[:](int64, float64, int64, float64[:]), nogil=True)
def bfuns(i, u, p, U):
    """Evaluates all non-zero bspline basis functions :math:`B_{j,p}(u)`
    at :math:`u`

    Parameters:
        i (int) : knot span in which :math:`u` lies
        u (float) : evaluation point
        p (int)  : basis function degree
        U (np.array(float)) : knot vector

    Returns:
        np.array(float) : scalar value for each :math:`B_{j,p}(u) \geq 0`,
        :math:`j=i-p\ldots i` (:math:`p+1` values)
    """

    N = np.zeros(p+1)
    left = np.zeros(p+1)
    right = np.zeros(p+1)

    N[0] = 1.;
    for j in range(1, p+1):
        left[j] = u - U[i+1-j]
        right[j] = U[i+j] - u
        saved = 0.0

        for r in range(0, j):
            temp = N[r] / (right[r+1] + left[j-r])
            N[r] = saved + right[r+1] * temp
            saved = left[j-r] * temp

        N[j] = saved

    return N


if __name__ == '__main__':
    from tensiga.iga.fspan import *
    from tensiga.iga.bfun import *

    u = 5./2;
    p = 2
    U = np.array([0,0,0,1,2,3,4,4,5,5,5], dtype=np.float64)
    i = fspan(u, p, U);

    print('Non-zero basis functions: [B_2,2, B_3,2, B_[4,2]]',
      '\nwith values:', bfuns(i,u,p,U))
    # eqv. to
    #print(bfun(2, u, p, U))
    #print(bfun(3, u, p, U))
    #print(bfun(4, u, p, U))
