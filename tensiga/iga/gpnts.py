import numpy as np
from numba import njit

@njit
def gpnts(p, U):
    ## COMPUTE GREVILLE NODES
    nkts = U.size
    nbfuns = nkts - p - 1

    G = np.zeros(nbfuns)
    for k in range(0, nbfuns):
        Gk = 0
        for l in range(1, p+1):
            Gk += U[k+l]

        G[k] = Gk / (p)

    return G



if __name__ == '__main__':
    p = 2
    U = np.linspace(0,1,10)
    print(gpnts(p,U))
