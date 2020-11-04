import numpy as np
from scipy.sparse import kron
from scipy.sparse.linalg import LinearOperator
from tensiga.iga.Nurbs import Nurbs
from tensiga.iga.Bspline import Bspline
from tensiga.fredholm.HilbertSchmidtKern import _kern_mat_to_tens, _kern_pts_to_mulidx
from tensiga.utils.mat_dot_sp import mat_dot_sp
from sklearn.utils.extmath import cartesian


_LinOpInit_wthf = False
_nonParaTime = 0.0
_paraTime = 0.0
n_it_count = 0
it_time = 0

class Collocation:

    def __init__(self, domain, quadrature, kernel, data, method='kron'):
        ## initialize data
        self.domain = domain
        self.quadrature = quadrature
        self.kernel = kernel
        self.method = method
        self.data = data

        # compute greville abs
        self.gp = self.domain.greville_knts()
        self.gpp = [ self.domain.eval(self.gp, k)
          for k in range(self.domain.dim) ]

        # compute collocation matrix at greville abs
        self.Bgp = self.domain.bfunsops(self.gp)

        # compute jacobian
        self.J = self.domain.jacobian(self.quadrature.ip)

        # compute integration points in physical domain
        self.xip = [ self.domain.eval(self.quadrature.ip, k)
          for k in range(self.domain.dim) ]

        # compute nurbs weighting function
        if isinstance(self.domain, Nurbs):
            self.R = self.domain.wghtfun(self.quadrature.ip)
            self.Rgp = self.domain.wghtfun(self.gp)

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
          _kern_pts_to_mulidx(self.gpp),
          _kern_pts_to_mulidx(self.xip), self.data)

        J = self.J.view().reshape(-1)
        R = self.R.view().reshape(-1)
        Rgp = self.Rgp.view().reshape(-1)
        W = self.domain.ctrlpts[-1].view()
        W = W.reshape(self.domain.ctrlpts[-1].size,-1)

        # precompute basis matrices
        Bj = self.B[0].multiply(self.quadrature.weights[0])
        for k in range(1, self.domain.dim):
            Bj = kron(Bj, self.B[k].multiply(self.quadrature.weights[k]))
        Bj = Bj.tocsr().multiply(R).multiply(J)
        Bj = Bj.tocsc().multiply(W)

        # compute tensorproduct bsplines collocation matrix at greville abs
        Bi = self.Bgp[0]
        for k in range(1, self.domain.dim):
            Bi = kron(Bi, self.Bgp[k])
        Bi = Bi.tocsr().multiply(Rgp)
        Bi = Bi.tocsc().multiply(W)

        # assemble A
        A = G @ Bj.transpose().tocsc()

        return A, Bi.transpose()

    def _nurbs_direct_elementwise(self):
        # important later for reshapes into elem colloc matrices
        self.B = [ Bk.tocsr() for Bk in self.B ]
        Bel = [None] * self.domain.dim

        Rgp = self.Rgp.view().reshape(-1)

        nip_el = self.quadrature.deg
        gridshape = [ self.domain.nelem(k) for k in range(0, self.domain.dim) ]

        # get elementwise univariate colloc matrices
        for k in range(0, self.domain.dim):
            Bel[k] = [None] * self.domain.nelem(k)
            for el in range(0, self.domain.nelem(k)):
                Bel[k][el] = self.B[k][:, nip_el[k]*el:nip_el[k]*el+nip_el[k]].data.reshape(-1, nip_el[k])

        # get kronecker jacobian on each element as a view of self.J
        Jel = [None] * self.domain.nelem()
        for el in range(0, self.domain.nelem()):
            el_mulidx = np.unravel_index(el, gridshape)
            slices = tuple([slice(nip_el[k]*el_mulidx[k], nip_el[k]*el_mulidx[k]+nip_el[k]) for k in range(0, self.domain.dim)])
            Jel[el] = self.J[slices].view().reshape(-1)

        # get kronecker weighting on each element as a view of self.R
        Rel = [None] * self.domain.nelem()
        for el in range(0, self.domain.nelem()):
            el_mulidx = np.unravel_index(el, gridshape)
            slices = tuple([slice(nip_el[k]*el_mulidx[k], nip_el[k]*el_mulidx[k]+nip_el[k]) for k in range(0, self.domain.dim)])
            Rel[el] = self.R[slices].view().reshape(-1)

        # get kronecker weighting on each element as a view of self.Rgp
        Rgp_el = [None] * self.domain.nelem()
        for el in range(0, self.domain.nelem()):
            el_mulidx = np.unravel_index(el, gridshape)
            slices = tuple([slice(nip_el[k]*el_mulidx[k], nip_el[k]*el_mulidx[k]+nip_el[k]) for k in range(0, self.domain.dim)])
            Rgp_el[el] = self.Rgp[slices].view().reshape(-1)
            
        W = self.domain.ctrlpts[-1].view()
        W = W.reshape(self.domain.ctrlpts[-1].size,-1)
        
        # allocate and initialize global matrices
        A = np.zeros((np.prod(self.domain.nbfuns), np.prod(self.domain.nbfuns)))
        B = np.zeros((np.prod(self.domain.nbfuns), np.prod(self.domain.nbfuns)))

        # compute tensorproduct bsplines collocation matrix at greville abs
        Bi = self.Bgp[0]
        for k in range(1, self.domain.dim):
            Bi = kron(Bi, self.Bgp[k])
        Bi = Bi.tocsr().multiply(Rgp)
        Bi = Bi.tocsc().multiply(W)

        
        # compute element matrices and add contribution to A
        for el in range(0, self.domain.nelem()):
            # get kronecker xips on each element 
            el_mulidx = np.unravel_index(el, gridshape)
            slices = tuple([slice(nip_el[d]*el_mulidx[d], nip_el[d]*el_mulidx[d]+nip_el[d]) for d in range(0, self.domain.dim)])
            el_ip = [ self.quadrature.ip[d][slices[d]] for d in range(0, self.domain.dim) ]
            el_xip = [ self.domain.eval(el_ip, d) for d in range(self.domain.dim) ]

            # define index maps 
            ldofs = [ self.B[d][:,nip_el[d]*el_mulidx[d]].nonzero()[0] for d in range(0, self.domain.dim) ]
            gdofs = np.ravel_multi_index(cartesian(ldofs).transpose(), self.domain.nbfuns)

            # compute the kernel on the element
            Gel = self.kernel(
              _kern_pts_to_mulidx(self.gpp),
              _kern_pts_to_mulidx(el_xip), self.data)
        
            # precompute element basis matrices
            Bj = Bel[0][el_mulidx[0]] * self.quadrature.weights[0][slices[0]]
            for k in range(1, self.domain.dim):
                Bj = np.kron(Bj, Bel[k][el_mulidx[k]] * self.quadrature.weights[k][slices[k]])
            Bj = Bj * Jel[el] * Rel[el] * W[gdofs]

            # assemble A on the element
            Apart = Gel @ Bj.transpose() # ich muss hier zwei mal uber die elemente summieren!!!!
                
            A[:,gdofs] += Apart

        return A, Bi.transpose()

    def _bspline_direct_kron(self):
        G = self.kernel(
          _kern_pts_to_mulidx(self.gpp),
          _kern_pts_to_mulidx(self.xip), self.data)

        J = self.J.view().reshape(-1)

        # precompute basis matrices
        Bj = self.B[0].multiply(self.quadrature.weights[0])
        for k in range(1, self.domain.dim):
            Bj = kron(Bj, self.B[k].multiply(self.quadrature.weights[k]))
        Bj = Bj.tocsr().multiply(J)

        # compute tensorproduct bsplines collocation matrix at greville abs
        Bi = self.Bgp[0]
        for k in range(1, self.domain.dim):
            Bi = kron(Bi, self.Bgp[k])
        Bi = Bi.tocsr()
        Bi = Bi.tocsc()

        # assemble A
        A = G @ Bj.transpose().tocsc()

        return A, Bi.transpose()

    def _bspline_direct_elementwise(self):
        # important later for reshapes into elem colloc matrices
        self.B = [ Bk.tocsr() for Bk in self.B ]
        Bel = [None] * self.domain.dim

        nip_el = self.quadrature.deg
        gridshape = [ self.domain.nelem(k) for k in range(0, self.domain.dim) ]

        # get elementwise univariate colloc matrices
        for k in range(0, self.domain.dim):
            Bel[k] = [None] * self.domain.nelem(k)
            for el in range(0, self.domain.nelem(k)):
                Bel[k][el] = self.B[k][:, nip_el[k]*el:nip_el[k]*el+nip_el[k]].data.reshape(-1, nip_el[k])

        # get kronecker jacobian on each element as a view of self.J
        Jel = [None] * self.domain.nelem()
        for el in range(0, self.domain.nelem()):
            el_mulidx = np.unravel_index(el, gridshape)
            slices = tuple([slice(nip_el[k]*el_mulidx[k], nip_el[k]*el_mulidx[k]+nip_el[k]) for k in range(0, self.domain.dim)])
            Jel[el] = self.J[slices].view().reshape(-1)
        
        # allocate and initialize global matrices
        A = np.zeros((np.prod(self.domain.nbfuns), np.prod(self.domain.nbfuns)))
        B = np.zeros((np.prod(self.domain.nbfuns), np.prod(self.domain.nbfuns)))

        # compute tensorproduct bsplines collocation matrix at greville abs
        Bi = self.Bgp[0]
        for k in range(1, self.domain.dim):
            Bi = kron(Bi, self.Bgp[k])
        Bi = Bi.tocsr()

        # compute element matrices and add contribution to A
        for el in range(0, self.domain.nelem()):
            # get kronecker xips on each element 
            el_mulidx = np.unravel_index(el, gridshape)
            slices = tuple([slice(nip_el[d]*el_mulidx[d], nip_el[d]*el_mulidx[d]+nip_el[d]) for d in range(0, self.domain.dim)])
            el_ip = [ self.quadrature.ip[d][slices[d]] for d in range(0, self.domain.dim) ]
            el_xip = [ self.domain.eval(el_ip, d) for d in range(self.domain.dim) ]

            # define index maps 
            ldofs = [ self.B[d][:,nip_el[d]*el_mulidx[d]].nonzero()[0] for d in range(0, self.domain.dim) ]
            gdofs = np.ravel_multi_index(cartesian(ldofs).transpose(), self.domain.nbfuns)

            # compute the kernel on the element
            Gel = self.kernel(
              _kern_pts_to_mulidx(self.gpp),
              _kern_pts_to_mulidx(el_xip), self.data)
        
            # precompute element basis matrices
            Bj = Bel[0][el_mulidx[0]] * self.quadrature.weights[0][slices[0]]
            for k in range(1, self.domain.dim):
                Bj = np.kron(Bj, Bel[k][el_mulidx[k]] * self.quadrature.weights[k][slices[k]])
            Bj = Bj * Jel[el]

            # assemble A on the element
            Apart = Gel @ Bj.transpose() # ich muss hier zwei mal uber die elemente summieren!!!!
                
            A[:,gdofs] += Apart

        return A, Bi.transpose()

    def _nurbs_matrix_free(self):
        J = self.J.view().reshape(-1)
        R = self.R.view().reshape(-1)
        Rgp = self.Rgp.view().reshape(-1)
        W = self.domain.ctrlpts[-1].view()
        W = W.reshape(self.domain.ctrlpts[-1].size,-1)

        # precompute basis matrices
        Bj = self.B[0].multiply(self.quadrature.weights[0])
        for k in range(1, self.domain.dim):
            Bj = kron(Bj, self.B[k].multiply(self.quadrature.weights[k]))
        Bj = Bj.tocsr().multiply(R).multiply(J)
        Bj = Bj.tocsc().multiply(W)
        BjT = Bj.transpose()


        def mv(v):
            ## This ugly lady fixes the unnecessary 1st call to mv(v)
            global _LinOpInit_wthf
            if _LinOpInit_wthf == False:
                _LinOpInit_wthf = True
                return v
            ##

            global n_it_count
            global it_time
            global _nonParaTime
            global _paraTime

            n_it_count += 1
            print('iter:', n_it_count)
            
            y = BjT * v
            vp = self.kernel(
                  _kern_pts_to_mulidx(self.gpp),
                  _kern_pts_to_mulidx(self.xip),
                  y, self.data)
            return vp


        # compute tensorproduct bsplines collocation matrix at greville abs
        Bi = self.Bgp[0]
        for k in range(1, self.domain.dim):
            Bi = kron(Bi, self.Bgp[k])
        Bi = Bi.tocsr().multiply(Rgp)
        Bi = Bi.tocsc().multiply(W)

        Ashape = (np.prod(self.domain.nbfuns),)*2
        A = LinearOperator(Ashape, matvec=mv)
        return A, Bi.transpose()

