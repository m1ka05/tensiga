import numpy as np
from scipy.sparse import csc_matrix, find
from scipy.sparse.linalg import spsolve
from tensiga.iga.gpnts import gpnts
from tensiga.iga.bfunsop import bfunsop

def delevop(p, U, q, V):
    m = U.size - 1
    n = m - p - 1

    x = gpnts(q, V)

    Bold_data, Bold_shape = bfunsop(x, p, U)
    Bnew_data, Bnew_shape = bfunsop(x, q, V)

    Bold = csc_matrix(Bold_data, shape=Bold_shape).transpose().tocsc()
    Bnew = csc_matrix(Bnew_data, shape=Bnew_shape).transpose().tocsc()
    Bold = spsolve(Bnew, Bold).transpose().tocsc()

    return Bold
