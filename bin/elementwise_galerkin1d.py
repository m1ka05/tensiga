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
import matplotlib.pyplot as plt
import tensiga.utils.varmeshes as varmeshes
from tensiga.quadrature.glnint import glnint
from tensiga.fredholm.HilbertSchmidtKern import expkernop as cov
from tensiga.fredholm.assembly.Galerkin import Galerkin

# init geometry
domain = varmeshes.UnitCube(1, 2)

# refine the spline object
domain.href(np.linspace(0, 1, 25)[1:-1], 0)

# compute global quadrature rule
quadrature = glnint(domain.kv, domain.deg+np.array([1]))

# define data struct for covariance kernel (:sig:, :b:, :L:)
cov_data = np.array([1., 1., 1.])

# initialize method
method = Galerkin(domain, quadrature, cov, cov_data)

# formation and assembly
A, B = method._bspline_direct_elementwise()

# solution
neigs = 10
lambda_h, f_h = eigsh(A, neigs, B) # ordered smallest to highest
method.normalize_ev(f_h)

# plot
ep = np.linspace(0,1,200)
x = domain.eval(quadrature.ip)

for k in range(1,10):
    data = method.eval_ef(quadrature.ip, f_h, -k)
    plt.plot(x, np.sqrt(lambda_h[-k])*data)

plt.show()
