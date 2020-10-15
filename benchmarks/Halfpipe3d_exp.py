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
from tensiga.fredholm.HilbertSchmidtKern import expkernop as cov
from tensiga.fredholm.HilbertSchmidtKern import expkernop_at as cov_at
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
# script.py [res-fname] [mesh-idx] [Galerkin | Collocation | ApproxGalerkin]

print(str(sys.argv))

fname = '/home/mika/proj/ibq/tmpres/'+str(sys.argv[1])+'.pkl'
batch = int(sys.argv[2])

# init geometry (Rahmans mesh)
domain = varmeshes.Halfpipe3D(10, 8, 15)
if batch == -2:
    domain.href(np.arange(0, 0.5, 1/38)[1:])
    domain.href(np.arange(0.5, 1, 1/38)[1:])
    domain.href(np.arange(0, 1, 1/4)[1:], 1)
    domain.href(np.arange(0, 1, 1/25)[1:], 2)
else:
    domain.href(np.arange(0, 0.5, 1/32)[1:])
    domain.href(np.arange(0.5, 1, 1/32)[1:])
    domain.href(np.arange(0, 1, 1/8)[1:], 2)

def ref_factor(ref,k):
    if k == 0:
        return np.int(ref)
    elif k == 1:
        return np.int(np.ceil(ref/(np.pi*((8+10)/2)) * 2))
    elif k == 2:
        return np.int(np.ceil(ref/(np.pi*((8+10)/2)) * 15))

iref = np.array([10, 20, 30, 40, 50, 80, 90, 120, 160, 170])
idomain_nodes = []
# get interpolation space
'''
if batch == -1:
    idomain_nodes = [ np.unique(kv)[1:-1] for kv in domain.kv ]
elif batch == -2:
    idomain_nodes = [ np.unique(kv)[1:-1] for kv in domain.kv ]
else:
    iknots0 = np.hstack(
                (np.linspace(0,0.5,np.int(np.floor(ref_factor(iref[batch], 0)/2)))[1:],
                 np.linspace(0.5,1,np.int(np.floor(ref_factor(iref[batch], 0)/2)))[1:-1]))
    idomain_nodes.append(iknots0)
    for k in range(1,domain.dim):
        irefk = ref_factor(iref[batch],k)
        idomain_nodes.append(np.linspace(0,1,np.int(ref_factor(iref[batch], k)))[1:-1])

idomain = varmeshes.interpolation_mesh(domain,idomain_nodes,2) # C0 linear 3d cube
'''
if batch == -1:
    idomain = varmeshes.Halfpipe3D(10, 8, 15)

    idomain.href(np.arange(0, 0.5, 1/32)[1:])
    idomain.href(np.arange(0, 0.5, 1/32)[1:])

    idomain.href(np.arange(0.5, 1, 1/32)[1:])
    idomain.href(np.arange(0.5, 1, 1/32)[1:])

    idomain.href(np.arange(0, 1, 1/8)[1:], 2)
else:
    idomain = varmeshes.Halfpipe3D(10, 8, 15)
    iknots0 = np.hstack(
                (np.linspace(0,0.5,np.int(np.floor(ref_factor(iref[batch], 0)/2)))[1:-1],
                 np.linspace(0.5,1,np.int(np.floor(ref_factor(iref[batch], 0)/2)))[1:-1]))
    idomain.href(np.repeat(iknots0,2), 0)
    for k in range(1,domain.dim):
        irefk = ref_factor(iref[batch],k)
        idomain.href(np.repeat(np.linspace(0,1,np.int(ref_factor(iref[batch], k)))[1:-1], 2), k)

# eval geometry
sdomain = idomain
ep = [ np.unique(kv) for kv in sdomain.kv ]
x, y, z = [ sdomain.eval(ep, k) for k in range(sdomain.dim) ]

# plot mesh
pv.set_plot_theme('document')
plotter = pv.Plotter()
#plotter = pv.Plotter(off_screen=True)
mesh = pv.StructuredGrid(x, y, z)
plotter.add_mesh(mesh, show_edges=True, color='orange')
plotter.camera_position = [(-36.48360660010704, -15.076744297974928, -9.94012976454021), (-0.11374793768036537, 9.315108552047567, 4.176445472374096), (0.49998951429012756, -0.8478048073913201, 0.17674131991158568)]
cpos = plotter.show(screenshot='mesh.png')
cpos = plotter.show()
print(cpos)
import pdb; pdb.set_trace()
'''
# eval geometry
ep = [ np.unique(kv) for kv in idomain.kv ]
x, y, z = [ idomain.eval(ep, k) for k in range(idomain.dim) ]


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
quadrature = glnint(domain.kv, domain.deg+np.array([3,3,3]))

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
