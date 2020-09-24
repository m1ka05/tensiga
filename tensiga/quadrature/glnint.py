import numpy as np
from numba import njit
from tensiga.quadrature.glint import glint
from tensiga.quadrature.Quadrature import Quadrature

def glnint(kv, p, tensor_ip_output=False):
    ipts = [None] * len(kv)
    weights = [None] * len(kv)

    # compute univariate rules for each parametric direction
    for k in range(len(kv)):
        ipts[k], weights[k] = glint(kv[k], p[k])

    # compute the weights tensor W_ijk
    W = weights[0]
    for k in range(1, len(weights)):
        W = np.tensordot(W, weights[k], axes=0)

    if tensor_ip_output == False:
        # use univariate coordinates as output
        ip = ipts
    else:
        # compute the grid of integration points as u_ijk, v_ijk, w_ijk...
        ip_shape = [ip.size for ip in ipts]
        nip = np.prod(ip_shape)
        ip = [None] * len(kv)

        for k in range(len(kv)):
            ip[k] = np.repeat(ipts[k], nip/ipts[k].size).reshape(ip_shape)
            ip[k] = np.moveaxis(ip[k], 0, k)

    return Quadrature(p, ip, weights, W)

if __name__ == '__main__':
    from time import time

    def f1(x, y, z):
        return x**2 + 3*y + 2*x - z

    kv1 = np.linspace(0,  2, 70)
    kv2 = np.linspace(5,  7, 70)
    kv3 = np.linspace(8, 10, 70)

    ## direct approach (useful for basis contrations [no loops then!], etc)
    tstart = time()
    quadrature = glnint([kv1,kv2,kv3], [2,2,2])
    u, v, w = quadrature.ip

    fval = np.empty((u.size, v.size, w.size))
    for i in range(u.size):
        for j in range(v.size):
            for k in range(w.size):
                fval[i,j,k] = f1(u[i], v[j], w[k])

    # compute F_ijk W_ijk = volume
    res = np.sum(fval * quadrature.W) # exact 296/3 ~= 98.6667
    print('Direct:')
    print(res)
    print('time: ', time()-tstart)


    ## vectorized approach (useful for covariance function evals, etc)
    tstart = time()
    quadrature = glnint([kv1,kv2,kv3], [2,2,2], True)
    u, v, w = quadrature.ip
    u = u.reshape(-1)
    v = v.reshape(-1)
    w = w.reshape(-1)

    f1_vec = np.vectorize(f1)
    fval = f1(u,v,w).reshape(quadrature.W.shape)
    res = np.sum(fval*quadrature.W)
    print('Vectorized:')
    print(res)
    print('time: ', time()-tstart)
