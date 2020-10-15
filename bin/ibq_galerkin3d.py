import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import tensiga.utils.varmeshes as varmeshes
from tensiga.quadrature.glnint import glnint
from tensiga.fredholm.HilbertSchmidtKern import expkernop as cov
from tensiga.fredholm.assembly.ApproxGalerkin import ApproxGalerkin
from scipy.sparse.linalg import eigsh

# init geometry
domain = varmeshes.Halfpipe3D(10, 8, 15)
idomain = varmeshes.Halfpipe3D(10, 8, 15)

# refine the spline object
for k in range(domain.dim):
    domain.href(np.linspace(0, 0.5, 5)[1:-1], k)
    domain.href(np.linspace(0.5, 1, 5)[1:-1], k)
    idomain.href(np.linspace(0, 0.5, 10)[1:-1], k)
    idomain.href(np.linspace(0.5, 1, 10)[1:-1], k)

# compute global quadrature rule
quadrature = glnint(domain.kv, domain.deg+np.array([1,1,1]))

# define data struct for covariance kernel (:sig:, :b:, :L:)
cov_data = np.array([1., 0.5, 10.])

# initialize method
method = ApproxGalerkin(domain, idomain, quadrature, cov, cov_data)

# formation and assembly
A, B = method.direct()
Aref, Bref = method.exact_kron()

# solution
neigs = 10
lambda_h, f_h = eigsh(A, neigs, B) # ordered smallest to highest
method.normalize_ev(f_h)

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
