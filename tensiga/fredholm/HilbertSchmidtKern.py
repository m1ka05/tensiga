import numpy as np
from numba import njit, jit, float64, prange

def _kern_mat_to_tens(G, x_shape, xp_shape):
    return G.view().reshape(x_shape + xp_shape)

def _kern_pts_to_mulidx(args):
    args = [ arg.view().reshape(-1) for arg in args ]
    return np.vstack(args).transpose()

@jit(float64[:,:](float64[:,:], float64[:,:], float64[:]), parallel=True, fastmath=True)
def gaukernop(x, xp, data):
    # in order to do this efficiently, the kernel operator expects
    # two arrays of points in the multiindex notation [[x_0, y_0],[x_1,y_1]]...
    # what is returned is G_IJ which can be then reshaped back into a tensor
    G = np.empty((x.shape[0], xp.shape[0]))
    for k in prange(x.shape[0]):
        d = x[k,:] - xp
        norm = np.sqrt(np.sum(d**2, axis=1))
        G[k,:] = (data[0]**2) * np.exp(-(norm/(data[1]*data[2]))**2 )

    return G

@jit(float64[:,:](float64[:,:], float64[:,:], float64[:], float64[:]), parallel=True, fastmath=True)
def gaukernop_at(x, xp, y, data):
    y = np.ascontiguousarray(y)
    res = np.empty((y.shape[0], 1))
    for k in prange(x.shape[0]):
        d = x[k,:] - xp
        norm = np.sqrt(np.sum(d**2, axis=1))

        res[k] = ((data[0]**2) * np.exp(-(norm/(data[1]*data[2]))**2)) @ y

    return res

@jit(float64[:,:](float64[:,:], float64[:,:], float64[:]), parallel=True, fastmath=True)
def expkernop(x, xp, data):
    # in order to do this efficiently, the kernel operator expects
    # two arrays of points in the multiindex notation [[x_0, y_0],[x_1,y_1]]...
    # what is returned is G_IJ which can be then reshaped back into a tensor
    G = np.empty((x.shape[0], xp.shape[0]))
    for k in prange(x.shape[0]):
        d = x[k,:] - xp
        norm = np.sqrt(np.sum(d**2, axis=1))
        G[k,:] = (data[0]**2) * np.exp(-norm/(data[1]*data[2]))

    return G

@jit(float64[:,:](float64[:,:], float64[:,:], float64[:], float64[:]), parallel=True, fastmath=True)
def expkernop_at(x, xp, y, data):
    y = np.ascontiguousarray(y)
    res = np.empty((y.shape[0], 1))
    for k in prange(x.shape[0]):
        d = x[k,:] - xp
        norm = np.sqrt(np.sum(d**2, axis=1))
        res[k] = ((data[0]**2) * np.exp(-norm/(data[1]*data[2]))) @ y

    return res

##
## https://datascience.blog.wzb.eu/2018/02/02/vectorization-and-parallelization-in-python-with-numpy-and-pandas/
##

@jit(float64[:,:](float64[:,:], float64[:,:], float64[:]), parallel=True, fastmath=True)
def sepexpkernop(x, xp, data):
    G = np.empty((x.shape[0], xp.shape[0]))
    for k in prange(x.shape[0]):
        d = x[k,:] - xp
        norm = np.sum(np.abs(d), axis=1)
        G[k,:] = (data[0]**2) * np.exp(-norm/(data[1]*data[2]))

    return G

@jit(float64[:,:](float64[:,:], float64[:,:], float64[:], float64[:]), parallel=True, fastmath=True)
def sepexpkernop_at(x, xp, y, data):
    y = np.ascontiguousarray(y)
    res = np.empty((y.shape[0], 1))
    for k in prange(x.shape[0]):
        d = x[k,:] - xp
        norm = np.sum(np.abs(d), axis=1)
        res[k] = ((data[0]**2) * np.exp(-norm/(data[1]*data[2]))) @ y

    return res
