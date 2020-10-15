import numpy as np
import tensiga
from tensiga.iga.Nurbs import Nurbs
from tensiga.iga.Bspline import Bspline
from math import sqrt
import os

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

def OpenUnitBasis(n, p, N):
    dim = n
    codim = n
    deg = [ p for _ in range(n) ]
    kv = [ np.linspace(0, 1, N) for k in range(n) ]
    cp_shape = [ kv[k].size-p-1 for k in range(n) ]

    # construct control points for n cube
    cp = [None] * dim
    cp_proto = [ np.zeros(cp_shape[k]) for k in range(n) ]

    cp[0] = np.repeat(cp_proto[0], np.prod(cp_shape[0:-1])).reshape(cp_shape)
    for k in range(1, codim):
        cp[k] = np.tile(np.repeat(cp_proto[k], np.prod(cp_shape[k:-1])), np.prod(cp_shape[0:k])).reshape(cp_shape)

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


def Shell3d():
    dim = 3
    codim = 3
    deg = [2, 2, 2]
    kv = [ np.array([0, 0, 0, 1.5707963267949, 1.5707963267949, 3.14159265358979, 3.14159265358979, 4.71238898038469, 4.71238898038469, 6.28318530717959, 6.28318530717959,  6.28318530717959]),
          np.array([-88.6003574854838,-88.6003574854838,-88.6003574854838,-2,-2,-1,-1,0,0,0])+88.6003574854838,
          np.array([0.,0.,0.,1.,1.,1.])]
    kv = [ v/v[-1] for v in kv ]

    module_path = os.path.dirname(tensiga.__file__)
    inner = np.loadtxt(module_path+'/utils/rhino_data/cps_inner.txt')
    center = np.loadtxt(module_path+'/utils/rhino_data/cps_center.txt')
    outer = np.loadtxt(module_path+'/utils/rhino_data/cps_outer.txt')

    x, y, z, w = [], [], [], []
    surfs = [outer, center, inner]

    for surf in surfs:
        # extract weights
        w_surf = surf[:,3]

        # project back
        npts = surf.shape[0]
        surf = surf[:,0:3]/w_surf.reshape(npts, -1)

        x_surf = surf[:,0]
        y_surf = surf[:,1]
        z_surf = surf[:,2]

        x.append(x_surf)
        y.append(y_surf)
        z.append(z_surf)
        w.append(w_surf)

    x = np.ascontiguousarray(np.hstack(x))
    y = np.ascontiguousarray(np.hstack(y))
    z = np.ascontiguousarray(np.hstack(z))
    w = np.ascontiguousarray(np.hstack(w))

    cp_shape = (9,7,3)
    ctrlpts = [np.array(x).reshape(cp_shape, order='F'),
               np.array(y).reshape(cp_shape, order='F'),
               np.array(z).reshape(cp_shape, order='F'),
               np.array(w).reshape(cp_shape, order='F')]

    spline = Nurbs(dim, codim, kv, deg, ctrlpts)
    return spline

# gets C0 bspline mesh of given size for any domain
def interpolation_mesh(domain, ref_nodes, p=1):
    idomain = UnitCube(domain.dim, 1)
    
    for k in range(0, idomain.dim):
        idomain.href(ref_nodes[k], k)

    ep = [ np.unique(kv) for kv in idomain.kv ]
    ctrlpts = [ domain.eval(ep, k) for k in range(domain.dim) ]
    idomain.ctrlpts = ctrlpts

    if int(p) > 1:
        for k in range(idomain.dim):
            idomain.pref(p-1, k)

    return idomain 
