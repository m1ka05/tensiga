import numpy as np

def univar_to_multivar(pts):
    # compute the grid of points as u_ijk, v_ijk, w_ijk...
    p_shape = [p.size for p in pts]
    npts = np.prod(p_shape)
    p = [None] * len(pts)

    for k in range(len(pts)):
        p[k] = np.repeat(pts[k], npts/pts[k].size).reshape(p_shape)
        p[k] = np.moveaxis(p[k], 0, k)

    return p
