import numpy as np
from numba import njit, jit, float64, prange
import sharedmem as shm
import os

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
    res = np.empty((x.shape[0], 1))
    for k in prange(x.shape[0]):
        d = x[k,:] - xp
        norm = np.sqrt(np.sum(d**2, axis=1))

        res[k] = ((data[0]**2) * np.exp(-(norm/(data[1]*data[2]))**2)) @ y

    return res

def shm_gaukernop_at(x, xp, y, data):
    y = np.ascontiguousarray(y)
    x_shm = shm.copy(x)
    xp_shm = shm.copy(xp)
    y_shm = shm.copy(y)
    data_shm = shm.copy(data)

    nthread = int(os.environ["TENSIGA_NUM_THREADS"])

    result = shm.empty(y.shape, np.float)
    with shm.MapReduce(np=nthread) as pool:
        def row(k):
            d = x_shm[k,:] - xp_shm
            norm = np.sqrt(np.sum(d**2, axis=1))
            return k, ((data_shm[0]**2) * np.exp(-(norm/(data_shm[1]*data_shm[2]))**2)) @ y_shm

        def reduce(k, coeff):
            result[k] = coeff

        r = pool.map(row, np.arange(x_shm.shape[0]), reduce=reduce)

    return result

def shm_chunk_gaukernop_at(x, xp, y, data):
    nthread = int(os.environ["TENSIGA_NUM_THREADS"])

    chunk_size = x.shape[0]//nthread
    last_chunk_size = chunk_size + x.shape[0] % nthread

    indices_start = [ chunk_size*k for k in range(nthread-1) ]
    indices_start.append(chunk_size*(nthread-1))
    indices_start = shm.copy(np.array(indices_start))

    indices_stop = [ chunk_size*(k+1) for k in range(nthread-1) ]
    indices_stop.append(chunk_size*(nthread-1) + last_chunk_size)
    indices_stop = shm.copy(np.array(indices_stop))
    
    y = np.ascontiguousarray(y)
    x = shm.copy(x)
    xp = shm.copy(xp)
    y = shm.copy(y)
    data = shm.copy(data)

    result = shm.empty((y.shape[0],1), np.float)

    with shm.MapReduce(np=nthread) as pool:
        @jit(fastmath=True)
        def row(k):
            xslice = x[slice(indices_start[k], indices_stop[k]),:]
            res = np.empty((xslice.shape[0],1)) 
            for l in range(xslice.shape[0]):
                d = xslice[l,:] - xp
                norm = np.sqrt(np.sum(d**2, axis=1))
                res[l] = ((data[0]**2) * np.exp(-(norm/(data[1]*data[2]))**2)) @ y

            return k, res 

        def reduce(k, coeff):
            result[slice(indices_start[k], indices_stop[k])] = coeff

        r = pool.map(row, np.arange(nthread), reduce=reduce)

    return result

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
    res = np.empty((x.shape[0], 1))
    for k in range(x.shape[0]):
        d = x[k,:] - xp
        norm = np.sqrt(np.sum(d**2, axis=1))
        res[k] = ((data[0]**2) * np.exp(-norm/(data[1]*data[2]))) @ y

    return res

def shm_expkernop_at(x, xp, y, data):
    y = np.ascontiguousarray(y)
    x_shm = shm.copy(x)
    xp_shm = shm.copy(xp)
    y_shm = shm.copy(y)
    data_shm = shm.copy(data)

    nthread = int(os.environ["TENSIGA_NUM_THREADS"])

    result = shm.empty(y.shape, np.float)
    with shm.MapReduce(np=nthread) as pool:
        def row(k):
            d = x_shm[k,:] - xp_shm
            norm = np.sqrt(np.sum(d**2, axis=1))
            return k, ((data_shm[0]**2) * np.exp(-norm/(data_shm[1]*data_shm[2]))) @ y_shm

        def reduce(k, coeff):
            result[k] = coeff

        r = pool.map(row, np.arange(x_shm.shape[0]), reduce=reduce)

    return result

def shm_chunk_expkernop_at(x, xp, y, data):
    nthread = int(os.environ["TENSIGA_NUM_THREADS"])

    chunk_size = x.shape[0]//nthread
    last_chunk_size = chunk_size + x.shape[0] % nthread

    indices_start = [ chunk_size*k for k in range(nthread-1) ]
    indices_start.append(chunk_size*(nthread-1))
    indices_start = shm.copy(np.array(indices_start))

    indices_stop = [ chunk_size*(k+1) for k in range(nthread-1) ]
    indices_stop.append(chunk_size*(nthread-1) + last_chunk_size)
    indices_stop = shm.copy(np.array(indices_stop))
    
    y = np.ascontiguousarray(y)
    x = shm.copy(x)
    xp = shm.copy(xp)
    y = shm.copy(y)
    data = shm.copy(data)

    result = shm.empty((y.shape[0],1), np.float)

    with shm.MapReduce(np=nthread) as pool:
        @jit(fastmath=True)
        def row(k):
            xslice = x[slice(indices_start[k], indices_stop[k]),:]
            res = np.empty((xslice.shape[0],1)) 
            for l in range(xslice.shape[0]):
                d = xslice[l,:] - xp
                norm = np.sqrt(np.sum(d**2, axis=1))
                res[l] = ((data[0]**2) * np.exp(-norm/(data[1]*data[2]))) @ y

            return k, res 

        def reduce(k, coeff):
            result[slice(indices_start[k], indices_stop[k])] = coeff

        r = pool.map(row, np.arange(nthread), reduce=reduce)

    return result

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
