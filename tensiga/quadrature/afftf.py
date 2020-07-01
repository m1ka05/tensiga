import numpy as np
from numba import njit

def afftf(eta, bounds):
    # x(-1) = a
    # x(1) = b
    # x(eta) = (a+b)/2 * eta + (b-a)/2
    # dxdeta = (a+b)/2

    a = bounds[0]
    b = bounds[1]
    x = (b - a) / 2 * eta + (a + b) / 2
    dx = (b - a) / 2

    return x, dx


if __name__ == '__main__':
    d = 2
    print('d:', d)
    eta = np.array([1, 1])
    bounds = np.array([[-1,2],[3,4]]) # x, dx
    x, dx = afftf(d, eta, bounds)
    print('x:\n',x)
    print('dx:\n', dx)
    
    d = 1
    print('d:', d)
    eta = 1
    bounds = np.array([-1,2])
    x, dx = afftf(d, eta, bounds)
    print('x:\n', x)
    print('dx:\n', dx)


    
