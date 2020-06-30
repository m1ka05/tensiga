import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from tensiga.quadrature.glnint import glnint
from tensiga.fredholm.HilbertSchmidtKern import gaukernop_at as cov
from tensiga.fredholm.assembly.ApproxGalerkin import ApproxGalerkin
import tensiga.utils.varmeshes as varmeshes

# init geometry
domain = varmeshes.QuarterAnnulus2D(1.,0.6)
idomain = varmeshes.QuarterAnnulus2D(1.,0.6)

dp = 20
idomain.pref(dp, 0)
idomain.pref(dp, 1)

# refine the spline object
for k in range(domain.dim):
    domain.href(np.linspace(0,1,30)[1:-1], k)
    idomain.href(np.linspace(0,1,5)[1:-1], k)

# compute global quadrature rule
quadrature = glnint(domain.kv, domain.deg+np.array([1,1]))

# define data struct for covariance kernel (:sig:, :b:, :L:)
cov_data = np.array([1., .5, 1.])

# initialize method
method = ApproxGalerkin(domain, idomain, quadrature, cov, cov_data)

# formation and assembly
A, B = method.matrix_free()

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
