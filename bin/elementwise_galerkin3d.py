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
from scipy.sparse.linalg import eigsh
from scipy.linalg import eig
import matplotlib.pyplot as plt
import pyvista as pv
import tensiga.utils.varmeshes as varmeshes
from tensiga.quadrature.glnint import glnint
from tensiga.fredholm.HilbertSchmidtKern import expkernop as cov
from tensiga.fredholm.assembly.Galerkin import Galerkin
from time import time

np.set_printoptions(edgeitems=30, linewidth=100000)

# init geometry
domain = varmeshes.Halfpipe3D(10, 8, 15)

# refine the spline object (analog to Rahman2018)
domain.href(np.arange(0, 0.5, 1/32)[1:])
domain.href(np.arange(0.5, 1, 1/32)[1:])
domain.href(np.arange(0, 1, 1/8)[1:], 2)

# eval geometry
ep = [ np.unique(kv) for kv in domain.kv ]
x, y, z = [ domain.eval(ep, k) for k in range(domain.dim) ]

''' # plot mesh
plotter = pv.Plotter()
mesh = pv.StructuredGrid(x, y, z)
plotter.add_mesh(mesh, show_edges=False)
plotter.show_axes()
plotter.view_xy()
plotter.show()

'''

print(domain.nelem())

# compute global quadrature rule
quadrature = glnint(domain.kv, domain.deg+np.array([2,2,2]))

# define data struct for covariance kernel (:sig:, :b:, :L:)
cov_data = np.array([1., 0.5, 10.])

# initialize method
method = Galerkin(domain, quadrature, cov, cov_data)

# formation and assembly
A, B = method._nurbs_direct_elementwise()
Aref, Bref = method._bspline_direct_elementwise()



# solution
neigs = 10
lambda_h, f_h = eigsh(A, neigs, B) # ordered smallest to highest

#method.normalize_ev(f_h)
# eval geometry
ep = [ np.linspace(0,1,100) ] * domain.dim
x, y, z = [ domain.eval(ep, k) for k in range(domain.dim) ]

# plot at evaluation points
plotter = pv.Plotter()
cmap = plt.get_cmap('jet', 2048)
mesh = pv.StructuredGrid(x, y, z)
data = method.eval_ef(ep, f_h, -1)
plotter.add_mesh(mesh, show_edges=False, scalars=data.transpose(), stitle='  ', cmap=cmap)
plotter.show_axes()
plotter.view_xy()
plotter.show()
