import numpy as np
from scipy.sparse.linalg import splu
from tensiga.utils.splu_solve_mat import splu_solve_mat
from tensiga.utils.mat_dot_sp import mat_dot_sp


class BsplineInterp:

    def __init__(self, domain, idomain):
        self.domain = domain
        self.idomain = idomain

        self.gp = self.idomain.greville_knts()
        self.gp_phys = [ self.idomain.eval(self.gp, k)
          for k in range(self.idomain.dim) ]

        self.I = self.idomain.bfunsops(self.gp)

    def compute_lu(self):
        # Ik transposed since I = I_i(u_j)!
        self.lu = [ splu(Ik.transpose().tocsc()) for Ik in self.I ]
        return self.lu

    def direct_gkl_eval(self, f):
        self.compute_lu()

        g = splu_solve_mat(self.lu[0], f, 0)
        g = splu_solve_mat(self.lu[0], g, self.idomain.dim)
        for k in range(1, self.idomain.dim):
            g = splu_solve_mat(self.lu[k], g, k)
            g = splu_solve_mat(self.lu[k], g, k+self.idomain.dim)

        return g

    def eval_2d_interpolant(self, g, ep, method):
        Bt = method.idomain.bfunsops(ep)

        f = mat_dot_sp(g, Bt[0], axis=0)
        f = mat_dot_sp(f, Bt[0], axis=self.domain.dim)
        for k in range(1, method.domain.dim):
            f = mat_dot_sp(f, Bt[k], axis=k)
            f = mat_dot_sp(f, Bt[k], axis=k+method.domain.dim)

        return f
