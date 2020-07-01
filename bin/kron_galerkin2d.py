import numpy as np
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import pyvista as pv
import tensiga.utils.varmeshes as varmeshes
from tensiga.quadrature.glnint import glnint
from tensiga.fredholm.HilbertSchmidtKern import expkernop as cov
from tensiga.fredholm.assembly.Galerkin import Galerkin

# init geometry
domain = varmeshes.QuarterAnnulus2D(1.,0.6)

# refine the spline object
for k in range(domain.dim):
    domain.href(np.linspace(0,1,30)[1:-1], k)

# compute global quadrature rule
quadrature = glnint(domain.kv, domain.deg+np.array([1,1]))

# define data struct for covariance kernel (:sig:, :b:, :L:)
cov_data = np.array([1., .5, 1.])

# initialize method
method = Galerkin(domain, quadrature, cov, cov_data)

# formation and assembly
A, B = method.direct()

# solution
neigs = 3
lambda_h, f_h = eigsh(A, neigs, B) # ordered smallest to highest
print(lambda_h)
method.normalize_ev(f_h)

## postprocessing
# eval geometry
ep = [ np.linspace(0,1,150) ] * 2
x, y = [ domain.eval(ep, k) for k in range(0,2) ]
z = np.zeros(x.shape)

# plot at evaluation points
cmap = plt.get_cmap('jet', 2048)
plotter = pv.Plotter()
mesh = pv.StructuredGrid(x, y, z)
data = method.eval_ef(ep, f_h, -1)
plotter.add_mesh(mesh, show_edges=False, scalars=data.transpose(), stitle='  ', cmap=cmap)
plotter.show_axes()
plotter.view_xy()
plotter.show()
