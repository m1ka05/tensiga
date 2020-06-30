import numpy as np
from tensiga.iga.Nurbs import Nurbs
from tensiga.iga.Bspline import Bspline
from math import sqrt

def UnitCube(n, p):
    dim = n
    codim = n
    deg = [ p for _ in range(n) ]
    kv = [ np.repeat([0., 1.], deg[k]+1) for k in range(n) ]
    cp_shape = [ deg[k]+1 for k in range(n) ]

    # construct control points for n cube
    cp = [None] * dim
    cp_proto = [ np.linspace(0., 1., cp_shape[k]) for k in range(n) ]

    cp[0] = np.repeat(cp_proto[0], np.prod(cp_shape[0:-1])).reshape(cp_shape)
    for k in range(1, codim):
        cp[k] = np.tile(np.repeat(cp_proto[k], np.prod(cp_shape[k:-1])), np.prod(cp_shape[0:k])).reshape(cp_shape)
    '''
    dim = 2
    codim = 2
    deg = [2, 2]
    kv = [ np.array([0.,0.,0.,1.,1.,1.]), np.array([0.,0.,0.,1.,1.,1.]) ]
    ctrlpts = []
    cp_shape = (3,3)
    x = [ 0., 0., 0., 0.5, 0.5, 0.5, 1., 1., 1.  ]
    y = [ 0., 0.5, 1., 0., 0.5, 1., 0., 0.5, 1 ]
    ctrlpts.append(np.array(x, dtype=np.float).reshape(cp_shape))
    ctrlpts.append(np.array(y, dtype=np.float).reshape(cp_shape))
    '''

    # init primitive spline
    domain = Bspline(dim, codim, kv, deg, cp)
    return domain

def QuarterAnnulus2D(R, r):
    # geometry parameters
    dim = 2;
    codim = 2;
    deg = [2, 1]
    kv = [ np.array([0., 0., 0., 1., 1., 1.]), np.array([0., 0., 1., 1.])]

    ctrlpts = []
    cp_shape = (3, 2)
    x = [ R, r, R, r, 0., 0. ] # numpy ordering
    y = [ 0., 0., R, r, R, r ] #
    w = [ 1., 1., 1./sqrt(2.), 1./sqrt(2.), 1., 1. ]
    ctrlpts.append(np.array(x, dtype=np.float).reshape(cp_shape))
    ctrlpts.append(np.array(y, dtype=np.float).reshape(cp_shape))
    ctrlpts.append(np.array(w, dtype=np.float).reshape(cp_shape))

    # init primitive spline
    domain = Nurbs(dim, codim, kv, deg, ctrlpts)
    return domain

def QuarterAnnulus3D(R, r, L):
    ## define spline data
    dim = 3;
    codim = 3;
    deg = [2, 1, 1]
    kv = [ np.array([0., 0., 0., 1., 1., 1.]),
           np.array([0., 0., 1., 1.]),
           np.array([0., 0., 1., 1.]) ]

    # this is using the numpy ordering
    ctrlpts = []
    cp_shape = (3,2,2)
    x = [ R, R, r, r, R, R, r, r, .0, .0, .0, .0 ]
    y = [ 0, 0., 0., 0., R, R, r, r, R, R, r, r ]
    z = [ 0, L, 0., L, 0., L, 0., L, 0., L, 0., L ]
    w = [ 1., 1., 1., 1., 1./sqrt(2.), 1./sqrt(2.), 1./sqrt(2.), 1./sqrt(2.), 1., 1., 1., 1. ]
    ctrlpts.append(np.array(x, dtype=np.float).reshape(cp_shape))
    ctrlpts.append(np.array(y, dtype=np.float).reshape(cp_shape))
    ctrlpts.append(np.array(z, dtype=np.float).reshape(cp_shape))
    ctrlpts.append(np.array(w, dtype=np.float).reshape(cp_shape))

    ## init bspline object
    domain = Nurbs(dim, codim, kv, deg, ctrlpts)
    return domain

def Halfpipe2D(R, r):
    # geometry parameters
    dim = 2;
    codim = 2;
    deg = [2, 1]
    kv = [ np.array([0., 0., 0., .5, .5, 1., 1., 1.]), np.array([0., 0., 1., 1.])]

    ctrlpts = []
    cp_shape = (5, 2)
    W = 1./sqrt(2.)
    y = [ -R, -r, -R, -r, 0., 0., R, r, R, r ] # numpy ordering
    x = [ 0., 0., R, r, R, r, R, r, 0., 0. ] #
    w = [ 1., 1., W, W, 1, 1, W, W, 1., 1. ]
    ctrlpts.append(np.array(x, dtype=np.float).reshape(cp_shape))
    ctrlpts.append(np.array(y, dtype=np.float).reshape(cp_shape))
    ctrlpts.append(np.array(w, dtype=np.float).reshape(cp_shape))

    # init primitive spline
    domain = Nurbs(dim, codim, kv, deg, ctrlpts)
    return domain

def Halfpipe3D(R, r, L):
    dim = 3;
    codim = 3;
    deg = [2, 2, 2]
    kv = [ np.array([0., 0., 0., 0.5, 0.5, 1., 1., 1.]),
           np.array([0., 0., 0., 1., 1., 1.]),
           np.array([0., 0., 0., 1., 1., 1.]) ]

    ctrlpts = []
    cp_shape = (5,3,3)

    W = sqrt(2)
    #W = 1.

    x=[-R,-R,-R,-(R+r)/2,-(R+r)/2,-(R+r)/2,-r,-r,-r,-R,-R,-R,-(R+r)/2,-(R+r)/2,-(R+r)/2,-r,-r,-r,0,0,0,0,0,0,0,0,0,R,R,R,(R+r)/2,(R+r)/2,(R+r)/2,r,r,r,R,R,R,(R+r)/2,(R+r)/2,(R+r)/2,r,r,r]
    y=[0,L/2,L,0,L/2,L,0,L/2,L,0,L/2,L,0,L/2,L,0,L/2,L,0,L/2,L,0,L/2,L,0,L/2,L,0,L/2,L,0,L/2,L,0,L/2,L,0,L/2,L,0,L/2,L,0,L/2,L]
    z=[0,0,0,0,0,0,0,0,0,R,R,R,(R+r)/2,(R+r)/2,(R+r)/2,r,r,r,R,R,R,(R+r)/2,(R+r)/2,(R+r)/2,r,r,r,R,R,R,(R+r)/2,(R+r)/2,(R+r)/2,r,r,r,0,0,0,0,0,0,0,0,0]
    w=[1.,1.,1.,1.,1.,1.,1.,1.,1.,1./W,1./W,1./W,1./W,1./W,1./W,1./W,1./W,1./W,1,1,1,1,1,1,1,1,1,1./W,1./W,1./W,1./W,1./W,1./W,1./W,1./W,1./W,1.,1.,1.,1.,1.,1.,1.,1.,1.]

    ctrlpts.append(np.array(x, dtype=np.float).reshape(cp_shape))
    ctrlpts.append(np.array(y, dtype=np.float).reshape(cp_shape))
    ctrlpts.append(np.array(z, dtype=np.float).reshape(cp_shape))
    ctrlpts.append(np.array(w, dtype=np.float).reshape(cp_shape))

    ## init bspline object
    domain = Nurbs(dim, codim, kv, deg, ctrlpts)
    return domain
