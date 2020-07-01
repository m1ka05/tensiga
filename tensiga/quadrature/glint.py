import numpy as np
from numba import njit
from time import time
from tensiga.iga.auxkv import auxkv
from tensiga.iga.Bspline import Bspline
from tensiga.quadrature.afftf import afftf

def glint(kv, p):
    """Univariate, global Gauss-Legendre quadrature rule

    Parameters:
        n (int) : quadrature order n > 0

    Returns:
        ip (np.array(float)) : array integration point coordinates
        W (numpy.array(float)) : weights W[k] corresponding to ip[k]
    """
    dim = len(kv)
    aux, _ = auxkv(kv)
    nelem = len(aux) - 1

    ip, W = proto_rule = np.polynomial.legendre.leggauss(p)

    glob_ip = [None] * ip.size * nelem
    glob_W = [None] * ip.size * nelem
    for idx in range(nelem):
        lbound = aux[idx]
        rbound = aux[idx+1]

        x, dx = afftf(ip, [lbound, rbound])
        glob_ip[(idx*ip.size):(idx*ip.size + ip.size)] = x
        glob_W[(idx*ip.size):(idx*ip.size + ip.size)] = W * dx

    return np.array(glob_ip), np.array(glob_W)

if __name__ == '__main__':
    def f1(x):
        return x**2 + 3*x - 10

    kv = np.array([0, 1, 1, 1, 2, 3, 4, 4, 5, 6, 6, 6])
    kv = np.linspace(0,6,1000)

    tstart = time()
    ip, W = glint(kv, 2)
    print('time:', time()-tstart)

    res = 0
    for (ip_k, W_k) in zip(ip, W):
        res += W_k * f1(ip_k)

    print(res) # exact: 66
