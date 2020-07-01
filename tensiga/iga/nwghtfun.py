import numpy as np

def nwghtfun(spline, u):
    if not isinstance(u, list):
        u = [u]

    B = spline.bfunsops(u)
    W = spline.ctrlpts[-1]

    R = spline._mat_dot_multi_sp(W, B)

    return np.reciprocal(R)
    
