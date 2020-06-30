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

def unitkernop(x, xp, data):
    G = np.ones((x.shape[0], xp.shape[0]))
    return G

def unitkernop_at(x, xp, y, data):
    res = np.ones((1,xp.shape[0])) @ res
    return res

if __name__ == '__main__':
    '''
    x = np.array([15,14,13.,12,11,10,9,8,7,6,5,4,3,2,1]).reshape(5,3)
    xp = x.copy()
    np.random.seed(0)
    y = np.random.rand(5,1)
    '''

    '''
    x = np.random.rand(20000,3)
    xp = np.random.rand(20000,3)
    y = np.random.rand(20000,1)
    '''
    x = np.ones((4,3))
    xp = np.ones((4,3))
    y = np.array([1.,2.,3.,4.])

    G = expkernop_at(x, xp, y, np.array([1.,1.,1.]))
