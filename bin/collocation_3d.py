import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from scipy.sparse.linalg import eigs
from scipy.linalg import eig
import matplotlib.pyplot as plt
import pyvista as pv
import tensiga.utils.varmeshes as varmeshes
from tensiga.quadrature.glnint import glnint
from tensiga.fredholm.HilbertSchmidtKern import expkernop as cov
from tensiga.fredholm.assembly.Collocation import Collocation
from time import time

np.set_printoptions(edgeitems=30, linewidth=100000)

# init geometry
domain = varmeshes.Halfpipe3D(10, 8, 15)

# refine the spline object
'''
#domain.href(np.linspace(0, 0.5, 14)[1:-1], 0)
#domain.href(np.linspace(0.5, 1, 14)[1:-1], 0)
#domain.href(np.linspace(0, 1, 11)[1:-1], 2)
for k in range(domain.dim):
    domain.href(np.linspace(0,1,3)[1:-1], k)
''' 
domain.href(np.arange(0, 0.5, 1/32)[1:])
domain.href(np.arange(0.5, 1, 1/32)[1:])
domain.href(np.arange(0, 1, 1/16)[1:], 2)
domain.href(np.arange(0, 1, 1/8)[1:], 1)

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
print(np.prod(domain.nbfuns))

# compute global quadrature rule
quadrature = glnint(domain.kv, domain.deg+np.array([2,2,2]))

# define data struct for covariance kernel (:sig:, :b:, :L:)
cov_data = np.array([1., 0.5, 10.])

# initialize method
method = Collocation(domain, quadrature, cov, cov_data)

# formation and assembly
tstart = time()
A, B = method._nurbs_direct_elementwise()
print(time() - tstart)

# solution
neigs = 10
lambda_h, f_h = eigs(A, neigs, B) # ordered smallest to highest
lambda_h = np.real(lambda_h)
f_h = np.real(f_h)
print(lambda_h)
import pdb; pdb.set_trace()


#method.normalize_ev(f_h)
# eval geometry
ep = [ np.linspace(0,1,100) ] * domain.dim
x, y, z = [ domain.eval(ep, k) for k in range(domain.dim) ]

# plot at evaluation points
plotter = pv.Plotter()
cmap = plt.get_cmap('jet', 2048)
mesh = pv.StructuredGrid(x, y, z)
data = method.eval_ef(ep, f_h, 0)
plotter.add_mesh(mesh, show_edges=False, scalars=data.transpose(), stitle='  ', cmap=cmap)
plotter.show_axes()
plotter.view_xy()
plotter.show()
