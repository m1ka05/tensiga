import numpy as np
from scipy.sparse.linalg import eigsh
from time import time
import pyvista as pv
import matplotlib.pyplot as plt
from tensiga.utils.varmeshes import Shell3d
from tensiga.quadrature.glnint import glnint
from tensiga.fredholm.HilbertSchmidtKern import gaukernop_at as cov
from tensiga.fredholm.assembly.ApproxGalerkin import ApproxGalerkin

# init geometry
domain = Shell3d()
idomain = Shell3d()

# elevate order of the interpolation and solution space
'''
for k in range(domain.dim):
    domain.pref(1, k) # p+dp; dp = 1
    idomain.pref(1, k)
'''

# refine the solution space
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


# refine the interpolation space
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


# compute global quadrature rule
quadrature = glnint(domain.kv, domain.deg+np.array([1,1,1]))

# define data struct for covariance kernel (:sig:, :b:, :L:)
cov_data = np.array([1., 0.5, 176.21])

# initialize method
method = ApproxGalerkin(domain, idomain, quadrature, cov, cov_data, precond=True)

# formation and assembly
formation_tstart = time()
A, B = method.matrix_free()
formation_time = time() - formation_tstart

# ndofs
ndof = np.prod(domain.nbfuns)
indof = np.prod(idomain.nbfuns)
print('ndof, indof: ', ndof, ',', indof)

# nelems
nelem = domain.nelem()
inelem = idomain.nelem()

h_dom = domain.elsz()
h_idom = idomain.elsz()


# solution
neigs = 20
eigensolver_tstart = time()
lambda_h, f_h = eigsh(A, neigs, B) # ordered smallest to highest
eigensolver_time = time() - eigensolver_tstart
niter = method.niter()

# eval geometry
ep = [ np.linspace(0,1,160), np.linspace(0,1,160), np.linspace(0,1,160) ]
x, y, z = [ domain.eval(ep, k) for k in range(domain.dim) ]

# plot
eigenmode_number = -5 # consider the ordering
plotter = pv.Plotter()
mesh = pv.StructuredGrid(x, y, z)
data = method.eval_ef(ep, f_h, eigenmode_number)
cmap = plt.get_cmap('jet', 2048)
plotter.add_mesh(mesh, show_edges=False, scalars=data.transpose(), stitle='  ', cmap=cmap)
plotter.show_axes()
plotter.view_xy()
plotter.export_vtkjs('/home/mika/tmps/vtkjs/shell.js')
plotter.show()
