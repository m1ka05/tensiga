import os
thread_env = ["OMP_NUM_THREADS",
  "NUMBA_NUM_THREADS",
  "OPENBLAS_NUM_THREADS"
  "MKL_NUM_THREADS"
  "VECLIB_MAXIMUM_THREADS"
  "NUMEXPR_NUM_THREADS"]
for env in thread_env:
    os.environ[env] = "1"

import numpy as np
import scipy.io
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
import pyvista as pv
import tensiga.utils.varmeshes as varmeshes
from tensiga.quadrature.glnint import glnint
from tensiga.fredholm.HilbertSchmidtKern import gaukernop as cov
from tensiga.fredholm.HilbertSchmidtKern import gaukernop_at as cov_at
from tensiga.fredholm.assembly.Galerkin import Galerkin
from tensiga.fredholm.assembly.Collocation import Collocation
from tensiga.fredholm.assembly.ApproxGalerkin import ApproxGalerkin
from time import time
import csv
import os.path
from os import path
import sys
import pickle as pkl
np.set_printoptions(edgeitems=30, linewidth=100000)

# syntax
# script.py [res-fname] [mesh-idx] [p] [Galerkin | Collocation | ApproxGalerkin]

print(str(sys.argv))

fname = '/home/mika/proj/ibq/tmpres/'+str(sys.argv[1])+'.pkl'

# init geometry
domain = varmeshes.Shell3d()
idomain = varmeshes.Shell3d()

# elevate order of the interpolation space
if int(sys.argv[3]) > 2:
    for k in range(domain.dim):
        domain.pref(int(sys.argv[3])-2, k)

    for k in range(idomain.dim):
        idomain.pref(int(sys.argv[3])-2, k)

# refine solution space
if int(sys.argv[2]) == 0:
    quarter_circum_nkts = 22
    domain.href(np.linspace(0,0.25,quarter_circum_nkts)[1:-1],0)
    domain.href(np.linspace(0.25,0.5,quarter_circum_nkts)[1:-1],0)
    domain.href(np.linspace(0.5,0.75,quarter_circum_nkts)[1:-1],0)
    domain.href(np.linspace(0.75,1,quarter_circum_nkts)[1:-1],0)

    unkts = np.unique(domain.kv[1])
    domain.href(np.linspace(unkts[0],unkts[1], 20)[1:-1],1)
    domain.href(np.linspace(unkts[1],unkts[2], 3)[1:-1],1)
    domain.href(np.linspace(unkts[2],unkts[3], 3)[1:-1],1)

    domain.href(np.linspace(0,1,3)[1:-1],2)
elif int(sys.argv[2]) == 1:
    # more refined solution space
    quarter_circum_nkts = 43
    domain.href(np.linspace(0,0.25,quarter_circum_nkts)[1:-1],0)
    domain.href(np.linspace(0.25,0.5,quarter_circum_nkts)[1:-1],0)
    domain.href(np.linspace(0.5,0.75,quarter_circum_nkts)[1:-1],0)
    domain.href(np.linspace(0.75,1,quarter_circum_nkts)[1:-1],0)

    unkts = np.unique(idomain.kv[1])
    domain.href(np.linspace(unkts[0],unkts[1], 39)[1:-1],1)
    domain.href(np.linspace(unkts[1],unkts[2], 5)[1:-1],1)
    domain.href(np.linspace(unkts[2],unkts[3], 5)[1:-1],1)

    domain.href(np.linspace(0,1,5)[1:-1],2)

# refine interpolation space
circ_nkts = 22
idomain.href(np.linspace(0,0.25,circ_nkts)[1:-1],0)
idomain.href(np.linspace(0.25,0.5,circ_nkts)[1:-1],0)
idomain.href(np.linspace(0.5,0.75,circ_nkts)[1:-1],0)
idomain.href(np.linspace(0.75,1,circ_nkts)[1:-1],0)

unkts = np.unique(idomain.kv[1])
idomain.href(np.linspace(unkts[0],unkts[1], 20)[1:-1],1)
idomain.href(np.linspace(unkts[1],unkts[2], 3)[1:-1],1)
idomain.href(np.linspace(unkts[2],unkts[3], 3)[1:-1],1)

idomain.href(np.linspace(0,1, 3)[1:-1],2)

'''
# eval geometry
ep = [ np.unique(kv) for kv in sdomain.kv ]
x, y, z = [ sdomain.eval(ep, k) for k in range(sdomain.dim) ]

# plot mesh
plotter = pv.Plotter()
mesh = pv.StructuredGrid(x, y, z)
plotter.add_mesh(mesh, show_edges=False)
plotter.show_axes()
plotter.view_xy()
plotter.show()
import pdb; pdb.set_trace()
'''


# number of elements
nelem = np.prod([(np.unique(domain.kv[k]).size-1) for k in range(domain.dim)])
inelem = np.prod([(np.unique(idomain.kv[k]).size-1) for k in range(idomain.dim)])
print('nelem:', nelem)
print(np.prod(domain.nbfuns))
print('inelem:', inelem)

# mesh size
unique_kv = [ np.unique(kv) for kv in domain.kv ]
phy_knot = [ domain.eval(unique_kv, k) for k in range(domain.dim)  ] 
x = np.array([ phy_knot[k][0,0,0] for k in range(domain.dim) ])
xp = np.array([ phy_knot[k][1,1,1] for k in range(domain.dim) ])
h_dom = np.linalg.norm(x-xp)

unique_kv = [ np.unique(kv) for kv in idomain.kv ]
phy_knot = [ idomain.eval(unique_kv, k) for k in range(idomain.dim)  ] 
x = np.array([ phy_knot[k][0,0,0] for k in range(idomain.dim) ])
xp = np.array([ phy_knot[k][1,1,1] for k in range(idomain.dim) ])
h_idom = np.linalg.norm(x-xp)

# compute global quadrature rule
quadrature = glnint(domain.kv, domain.deg+np.array([1,1,1]))

# define data struct for covariance kernel (:sig:, :b:, :L:)
cov_data = np.array([1., 0.5, 176.21])


# initialize method
formation_tstart = time()
if(str(sys.argv[4]) == "ApproxGalerkin"):
    method = ApproxGalerkin(domain, idomain, quadrature, cov_at, cov_data, precond=True)
    print('here!')
    A, B = method.matrix_free()
    solver = eigsh
elif(str(sys.argv[4]) == "ExactApproxGalerkin"):
    method = ApproxGalerkin(domain, idomain, quadrature, cov, cov_data)
    A, B = method.exact_elementwise()
    solver = eigsh
elif(str(sys.argv[4]) == "DirectApproxGalerkin"):
    method = ApproxGalerkin(domain, idomain, quadrature, cov, cov_data)
    A, B = method.direct()
    solver = eigsh
elif(str(sys.argv[4]) == "Galerkin"):
    method = Galerkin(domain, quadrature, cov, cov_data)
    A, B = method._nurbs_direct_elementwise() 
    solver = eigsh
elif(str(sys.argv[4]) == "Collocation"):
    method = Collocation(domain, quadrature, cov, cov_data)
    A, B = method._nurbs_direct_elementwise()
    solver = eigs
formation_time = time() - formation_tstart
ndof = np.prod(domain.nbfuns) 
indof = np.prod(idomain.nbfuns)


# solution
eigensolver_tstart = time()
lambda_h, f_h = solver(A, 20, B) # ordered smallest to highest
eigensolver_time = time() - eigensolver_tstart

# save data for postprocessing
if str(sys.argv[4]) == "ApproxGalerkin":
    data = { "formation_time" : formation_time,
             "eigensolver_time" : eigensolver_time,
             "lambda_h" : lambda_h,
             "ndof" : ndof,
             "indof" : indof,
             "nelem" : nelem,
             "inelem" : inelem,
             "h_dom" : h_dom, 
             "h_idom" : h_idom,
             "f_h" : f_h,
             "domain" : domain,
             "idomain" : idomain,
             }
else:
    data = { "formation_time" : formation_time,
             "eigensolver_time" : eigensolver_time,
             "lambda_h" : lambda_h,
             "ndof" : ndof,
             "indof" : indof,
             "nelem" : nelem,
             "inelem" : inelem,
             "h_dom" : h_dom, 
             "h_idom" : h_idom,
             "f_h" : f_h,
             "domain" : domain,
             "idomain" : idomain,
             "A": A,
             "B": B,
             }

print(formation_time, lambda_h)

with open(fname, "wb") as fh:
    pkl.dump(data, fh)

