import numpy as np
from numba import njit
from tensiga.iga.auxkv import auxkv

#@njit
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

    # in the case of C^-1
    Z, M = auxkv(G)
    if sum(M) != M.size:
        G_update = []
        eps = 1e-12
        for z, m in zip(Z, M):
            if m == 2:
                G_update.append(z - eps)
                G_update.append(z + eps)
            else:
                G_update.append(z)

        G = np.array(G_update)

    return G



if __name__ == '__main__':
    p = 2
    U = np.linspace(0,1,10)
    print(gpnts(p,U))
