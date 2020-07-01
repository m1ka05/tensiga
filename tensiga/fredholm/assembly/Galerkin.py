import numpy as np
from scipy.sparse import kron
from scipy.sparse.linalg import LinearOperator
from tensiga.iga.Nurbs import Nurbs
from tensiga.iga.Bspline import Bspline
from tensiga.fredholm.HilbertSchmidtKern import _kern_mat_to_tens, _kern_pts_to_mulidx
from tensiga.utils.mat_dot_sp import mat_dot_sp

_LinOpInit_wthf = False
n_it_count = 0

class Galerkin:

    def __init__(self, domain, quadrature, kernel, data, method='kron'):
        ## initialize data
        self.domain = domain
        self.quadrature = quadrature
        self.kernel = kernel
        self.method = method
        self.data = data

        # compute jacobian
        self.J = self.domain.jacobian(self.quadrature.ip)

        # compute integration points in physical domain
        self.xip = [ self.domain.eval(self.quadrature.ip, k)
          for k in range(self.domain.dim) ]

        # compute nurbs weighting function
        if isinstance(self.domain, Nurbs):
            self.R = self.domain.wghtfun(self.quadrature.ip)

        # compute basis operators
        self.B = self.domain.bfunsops(self.quadrature.ip)

    def eval_ef(self, ep, ev, k):
        # ev into tensor of coeffs (each columnt corresponds to an ev)
        nev = ev.shape[1]
        ev = [ ev[:,k].reshape(self.domain.nbfuns) for k in range(nev) ]
        return self.domain.evalwrt(ep, ev[k])

    def normalize_ev(self, ev):
        nev = ev.shape[1]
        factors = [0.] * nev
        for k in range(nev):
            ef = self.eval_ef(self.quadrature.ip, ev, k)
            factors[k] = np.sum(self.quadrature.W * (ef**2))
            ev[:,k] /= np.sqrt(factors[k])

        return factors

    def direct(self):
        if isinstance(self.domain, Nurbs):
            if self.method == 'kron':
                return self._nurbs_direct_kron()
            elif self.method == 'sumfac':
                return self._nurbs_direct_kron()

        elif isinstance(self.domain, Bspline):
            if self.method == 'kron':
                return self._bspline_direct_kron()
            elif self.method == 'sumfac':
                return self._bspline_direct_kron()

    def _nurbs_direct_kron(self):
        G = self.kernel(
          _kern_pts_to_mulidx(self.xip),
          _kern_pts_to_mulidx(self.xip), self.data)
        J = self.J.view().reshape(-1)
        R = self.R.view().reshape(-1)
        W = self.domain.ctrlpts[-1].view()
        W = W.reshape(self.domain.ctrlpts[-1].size,-1)

        # precompute basis matrices
        Bi = self.B[0].multiply(self.quadrature.weights[0])
        for k in range(1, self.domain.dim):
            Bi = kron(Bi, self.B[k].multiply(self.quadrature.weights[k]))
        Bi = Bi.tocsr().multiply(R).multiply(J)
        Bi = Bi.tocsc().multiply(W)

        Bj = self.B[0]
        for k in range(1, self.domain.dim):
            Bj = kron(Bj, self.B[k])
        Bj = Bj.tocsr().multiply(R)
        Bj = Bj.tocsc().multiply(W)

        # assemble A
        A = Bi.tocsr() @ G @ Bi.transpose().tocsc()

        # assemble B
        B = Bi.tocsr() @ Bj.transpose().tocsc()

        return A, B

    def _nurbs_direct_sumfac(self):
        pass

    def _nurbs_matrix_free(self):
        J = self.J.view().reshape(-1)
        R = self.R.view().reshape(-1)
        W = self.domain.ctrlpts[-1].view()
        W = W.reshape(self.domain.ctrlpts[-1].size,-1)

        # precompute basis matrices
        Bi = self.B[0].multiply(self.quadrature.weights[0])
        for k in range(1, self.domain.dim):
            Bi = kron(Bi, self.B[k].multiply(self.quadrature.weights[k]))
        Bi = Bi.tocsr().multiply(R).multiply(J)
        Bi = Bi.tocsc().multiply(W)

        Bj = self.B[0]
        for k in range(1, self.domain.dim):
            Bj = kron(Bj, self.B[k])
        Bj = Bj.tocsr().multiply(R)
        Bj = Bj.tocsc().multiply(W)

        # form A in matrix-free manner
        def mv(v):
            ## This ugly lady fixes the unnecessary 1st call to mv(v)
            global _LinOpInit_wthf
            if _LinOpInit_wthf == False:
                _LinOpInit_wthf = True
                return v
            ##
            global n_it_count
            n_it_count += 1
            print('iter: ', n_it_count)

            y = Bi.transpose().tocsr() @ v
            z = self.kernel(
                  _kern_pts_to_mulidx(self.xip),
                  _kern_pts_to_mulidx(self.xip),
                  y,
                  self.data)
            return Bi @ z


        Ashape = (np.prod(self.domain.nbfuns),)*2
        A = LinearOperator(Ashape, matvec=mv)

        # assemble B
        B = Bi.tocsr() @ Bj.transpose().tocsc()

        return A, B

    def _bspline_direct_kron(self):
        J = self.J.view().reshape(-1)
        G = self.kernel(
          _kern_pts_to_mulidx(self.xip),
          _kern_pts_to_mulidx(self.xip), self.data)

        # precompute basis matrices
        Bi = self.B[0].multiply(self.quadrature.weights[0])
        for k in range(1, self.domain.dim):
            Bi = kron(Bi, self.B[k].multiply(self.quadrature.weights[k]))
        Bi = Bi.tocsr().multiply(J)

        Bj = self.B[0]
        for k in range(1, self.domain.dim):
            Bj = kron(Bj, self.B[k])

        # assemble A
        A = Bi.tocsr() @ G @ Bi.transpose().tocsc()

        # assemble B
        B = Bi.tocsr() @ Bj.transpose().tocsc()

        return A, B

    def _bspline_direct_sumfac(self):
        pass

    def _bspline_matrix_free(self):
        J = self.J.view().reshape(-1)

        # precompute basis matrices
        Bi = self.B[0].multiply(self.quadrature.weights[0])
        for k in range(1, self.domain.dim):
            Bi = kron(Bi, self.B[k].multiply(self.quadrature.weights[k]))
        Bi = Bi.tocsr().multiply(J)

        Bj = self.B[0]
        for k in range(1, self.domain.dim):
            Bj = kron(Bj, self.B[k])

        # form A in matrix-free manner
        def mv(v):
            ## This ugly lady fixes the unnecessary 1st call to mv(v)
            global _LinOpInit_wthf
            if _LinOpInit_wthf == False:
                _LinOpInit_wthf = True
                return v
            ##
            global n_it_count
            n_it_count += 1
            print('iter: ', n_it_count)

            y = Bi.transpose().tocsr() @ v
            z = self.kernel(
                  _kern_pts_to_mulidx(self.xip),
                  _kern_pts_to_mulidx(self.xip),
                  y,
                  self.data)
            return Bi @ z

        Ashape = (np.prod(self.domain.nbfuns),)*2
        A = LinearOperator(Ashape, mv)

        # assemble B
        B = Bi.tocsr() @ Bj.transpose().tocsc()

        return A, B
