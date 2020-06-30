import numpy as np
from numba import njit, int64, float64


@njit(int64(float64, int64, float64[:] ))
def fspan(u, p, U):
    """Computes the span index in :math:`U`, in which :math:`u` lies

    Parameters:
        u (float) : evaluation point :math:`u`
        p (int)  : basis function degree :math:`p`
        U (np.array(float)) : knot vector :math:`U`

    Returns:
        int : span index :math:`i : [U_i, U_{i+1})`
    """
    nkts = U.size

    if u == U[p]:
        return p # = span

    if u == U[nkts - 1 - p]:
        return (nkts - 2 - p) # = span

    if (u > U[p]) and (u < U[nkts - 1 - p]):
        low = 0
        high = nkts - 1
        mid = (low + high) // 2

        while (u < U[mid]) or (u >= U[mid+1]):
            if u < U[mid]:
                high = mid
            else:
                low = mid

            mid = (low + high) // 2

        span = (low + high) // 2
    else:
        pass
        # todo: throw exception

    return span


if __name__ == '__main__':
    u = 2.5
    p = 2
    U = np.array([0,0,0,1,2,3,4,4,5,5,5], dtype=np.float64)
    print(fspan(u, p, U))
