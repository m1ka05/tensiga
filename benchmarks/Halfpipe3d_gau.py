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
# script.py [res-fname] [p] [Galerkin | Collocation | ApproxGalerkin]

print(str(sys.argv))

fname = '/home/mika/proj/ibq/tmpres/'+str(sys.argv[1])+'.pkl'

# init geometry (Rahmans mesh)
domain = varmeshes.Halfpipe3D(10, 8, 15)
if int(sys.argv[2]) == -2 or str(sys.argv[3]) == "DirectApproxGalerkin":
    domain.href(np.arange(0, 0.5, 1/38)[1:])
    domain.href(np.arange(0.5, 1, 1/38)[1:])
    domain.href(np.arange(0, 1, 1/4)[1:], 1)
    domain.href(np.arange(0, 1, 1/25)[1:], 2)
else:
    domain.href(np.arange(0, 0.5, 1/32)[1:])
    domain.href(np.arange(0.5, 1, 1/32)[1:])
    domain.href(np.arange(0, 1, 1/8)[1:], 2)





idomain = varmeshes.Halfpipe3D(10, 8, 15)

if int(sys.argv[2]) > 2:
    for k in range(idomain.dim):
        idomain.pref(int(sys.argv[2])-2, k)

    # refine solution space with p

idomain.href(np.arange(0, 0.5, 1/32)[1:])
idomain.href(np.arange(0.5, 1, 1/32)[1:])
idomain.href(np.arange(0, 1, 1/8)[1:], 2)

# eval geometry
ep = [ np.unique(kv) for kv in idomain.kv ]
x, y, z = [ idomain.eval(ep, k) for k in range(idomain.dim) ]

'''
# plot mesh
plotter = pv.Plotter()
mesh = pv.StructuredGrid(x, y, z)
plotter.add_mesh(mesh, show_edges=False)
plotter.show_axes()
plotter.view_xy()
plotter.show()
'''

# number of elements
nelem = np.prod([(np.unique(domain.kv[k]).size-1) for k in range(domain.dim)])
inelem = np.prod([(np.unique(idomain.kv[k]).size-1) for k in range(idomain.dim)])
print('nelem:', nelem)
print(np.prod(domain.nbfuns))
print('inelem:', inelem, np.prod(idomain.nbfuns))

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
print(quadrature.deg)

# define data struct for covariance kernel (:sig:, :b:, :L:)
cov_data = np.array([1., 0.5, 10.])


# initialize method
formation_tstart = time()
if(str(sys.argv[3]) == "ApproxGalerkin"):
    method = ApproxGalerkin(domain, idomain, quadrature, cov_at, cov_data, precond=True)
    A, B = method.matrix_free()
    solver = eigsh
elif(str(sys.argv[3]) == "DirectApproxGalerkin"):
    method = ApproxGalerkin(domain, idomain, quadrature, cov, cov_data)
    A, B = method.direct()
    solver = eigsh
elif(str(sys.argv[3]) == "ExactApproxGalerkin"):
    method = ApproxGalerkin(domain, idomain, quadrature, cov, cov_data)
    A, B = method.exact_kron()
    solver = eigsh
elif(str(sys.argv[3]) == "Galerkin"):
    method = Galerkin(domain, quadrature, cov, cov_data)
    A, B = method._nurbs_direct_elementwise()
    solver = eigsh
elif(str(sys.argv[3]) == "Collocation"):
    method = Collocation(domain, quadrature, cov, cov_data)
    A, B = method._nurbs_direct_elementwise()
    solver = eigs
formation_time = time() - formation_tstart
ndof = np.prod(domain.nbfuns) 
indof = np.prod(idomain.nbfuns)


# load reference
'''
with open('/home/mika/proj/ibq/tmpres/Shell3d_Galerkin_p2_reference.pkl', 'rb') as fh:
    data = pkl.load(fh)
'''

# solution
eigensolver_tstart = time()
lambda_h, f_h = solver(A, 20, B) # ordered smallest to highest
eigensolver_time = time() - eigensolver_tstart

# save data for postprocessing
if str(sys.argv[3]) == "ApproxGalerkin":
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

'''
with open('/home/mika/proj/ibq/tmpres/Galerkin_p2_reference.pkl', 'rb') as fh:
    ref_data = pkl.load(fh)
'''

'''
# eval geometry
ep = [ np.linspace(0,1,100) ] * domain.dim
x, y, z = [ domain.eval(ep, k) for k in range(domain.dim) ]

# plot at evaluation points
plotter = pv.Plotter()
cmap = plt.get_cmap('jet', 2048)
mesh = pv.StructuredGrid(x, y, z)
data = method.eval_ef(ep, np.real(f_h), -1)
plotter.add_mesh(mesh, show_edges=False, scalars=data.transpose(), stitle='  ', cmap=cmap)
plotter.show_axes()
plotter.view_xy()
plotter.show()
'''
