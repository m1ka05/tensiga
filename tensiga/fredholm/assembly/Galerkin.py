import numpy as np
from scipy.sparse import kron
from scipy.sparse.linalg import LinearOperator
from tensiga.iga.Nurbs import Nurbs
from tensiga.iga.Bspline import Bspline
from tensiga.fredholm.HilbertSchmidtKern import _kern_mat_to_tens, _kern_pts_to_mulidx
from tensiga.utils.mat_dot_sp import mat_dot_sp
from sklearn.utils.extmath import cartesian


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

    def _nurbs_direct_elementwise(self):
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

        # get kronecker weighting on each element as a view of self.R
        Rel = [None] * self.domain.nelem()
        for el in range(0, self.domain.nelem()):
            el_mulidx = np.unravel_index(el, gridshape)
            slices = tuple([slice(nip_el[k]*el_mulidx[k], nip_el[k]*el_mulidx[k]+nip_el[k]) for k in range(0, self.domain.dim)])
            Rel[el] = self.R[slices].view().reshape(-1)
            
        W = self.domain.ctrlpts[-1].view()
        W = W.reshape(self.domain.ctrlpts[-1].size,-1)
        
        # allocate and initialize global matrices
        A = np.zeros((np.prod(self.domain.nbfuns), np.prod(self.domain.nbfuns)))
        B = np.zeros((np.prod(self.domain.nbfuns), np.prod(self.domain.nbfuns)))

        # compute element matrices and add contribution to B
        for el in range(0, self.domain.nelem()):
            # get kronecker xips on each element 
            el_mulidx = np.unravel_index(el, gridshape)
            slices = tuple([slice(nip_el[d]*el_mulidx[d], nip_el[d]*el_mulidx[d]+nip_el[d]) for d in range(0, self.domain.dim)])
            el_ip = [ self.quadrature.ip[d][slices[d]] for d in range(0, self.domain.dim) ]
            el_xip = [ self.domain.eval(el_ip, d) for d in range(self.domain.dim) ]

            # define index map
            ldofs = [ self.B[d][:,nip_el[d]*el_mulidx[d]].nonzero()[0] for d in range(0, self.domain.dim) ]
            gdofs = np.ravel_multi_index(cartesian(ldofs).transpose(), self.domain.nbfuns)

            # precompute element basis matrices
            Bi = Bel[0][el_mulidx[0]] * self.quadrature.weights[0][slices[0]]
            for k in range(1, self.domain.dim):
                Bi = np.kron(Bi, Bel[k][el_mulidx[k]] * self.quadrature.weights[k][slices[k]])
            Bi = Bi * Jel[el] * Rel[el] * W[gdofs]

            Bj = Bel[0][el_mulidx[0]]
            for k in range(1, self.domain.dim):
                Bj = np.kron(Bj, Bel[k][el_mulidx[k]])
            Bj = Bj * Rel[el] * W[gdofs]

            # assemble B on the element
            Bpart = Bi @ Bj.transpose()

            B[np.ix_(*(gdofs,)*2)] += Bpart

        # compute element matrices and add contribution to A
        for el1 in range(0, self.domain.nelem()):
            print(el1)
            el1_mulidx = np.unravel_index(el1, gridshape)
            slices1 = tuple([slice(nip_el[d]*el1_mulidx[d], nip_el[d]*el1_mulidx[d]+nip_el[d]) for d in range(0, self.domain.dim)])
            el1_ip = [ self.quadrature.ip[d][slices1[d]] for d in range(0, self.domain.dim) ]
            el1_xip = [ self.domain.eval(el1_ip, d) for d in range(self.domain.dim) ]

            # define index map
            ldofs1 = [ self.B[d][:,nip_el[d]*el1_mulidx[d]].nonzero()[0] for d in range(0, self.domain.dim) ]
            gdofs1 = np.ravel_multi_index(cartesian(ldofs1).transpose(), self.domain.nbfuns)

            # precompute element basis matrices
            Bi = Bel[0][el1_mulidx[0]] * self.quadrature.weights[0][slices1[0]]
            for k in range(1, self.domain.dim):
                Bi = np.kron(Bi, Bel[k][el1_mulidx[k]] * self.quadrature.weights[k][slices1[k]])
            Bi = Bi * Jel[el1] * Rel[el1] * W[gdofs1]

            for el2 in range(0, self.domain.nelem()):
                # get kronecker xips on each element 
                el2_mulidx = np.unravel_index(el2, gridshape)
                slices2 = tuple([slice(nip_el[d]*el2_mulidx[d], nip_el[d]*el2_mulidx[d]+nip_el[d]) for d in range(0, self.domain.dim)])
                el2_ip = [ self.quadrature.ip[d][slices2[d]] for d in range(0, self.domain.dim) ]
                el2_xip = [ self.domain.eval(el2_ip, d) for d in range(self.domain.dim) ]

                # define index map
                ldofs2 = [ self.B[d][:,nip_el[d]*el2_mulidx[d]].nonzero()[0] for d in range(0, self.domain.dim) ]
                gdofs2 = np.ravel_multi_index(cartesian(ldofs2).transpose(), self.domain.nbfuns)

                # compute the kernel on the element
                Gel = self.kernel(
                  _kern_pts_to_mulidx(el2_xip),
                  _kern_pts_to_mulidx(el1_xip), self.data)

                # precompute element basis matrices
                Bj = Bel[0][el2_mulidx[0]] * self.quadrature.weights[0][slices2[0]]
                for k in range(1, self.domain.dim):
                    Bj = np.kron(Bj, Bel[k][el2_mulidx[k]] * self.quadrature.weights[k][slices2[k]])
                Bj = Bj * Jel[el2] * Rel[el2] * W[gdofs2]
               
                # assemble A on the element
                Apart = Bj @ Gel @ Bi.transpose() # ich muss hier zwei mal uber die elemente summieren!!!!

                
                A[np.ix_(*(gdofs2,gdofs1))] += Apart

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

        # compute element matrices and add contribution to B
        for el in range(0, self.domain.nelem()):
            # get kronecker xips on each element 
            el_mulidx = np.unravel_index(el, gridshape)
            slices = tuple([slice(nip_el[d]*el_mulidx[d], nip_el[d]*el_mulidx[d]+nip_el[d]) for d in range(0, self.domain.dim)])
            el_ip = [ self.quadrature.ip[d][slices[d]] for d in range(0, self.domain.dim) ]
            el_xip = [ self.domain.eval(el_ip, d) for d in range(self.domain.dim) ]

            # precompute element basis matrices
            Bi = Bel[0][el_mulidx[0]] * self.quadrature.weights[0][slices[0]]
            for k in range(1, self.domain.dim):
                Bi = np.kron(Bi, Bel[k][el_mulidx[k]] * self.quadrature.weights[k][slices[k]])
            Bi = Bi * Jel[el]

            Bj = Bel[0][el_mulidx[0]]
            for k in range(1, self.domain.dim):
                Bj = np.kron(Bj, Bel[k][el_mulidx[k]])

            # assemble B on the element
            Bpart = Bi @ Bj.transpose()

            ldofs = [ self.B[d][:,nip_el[d]*el_mulidx[d]].nonzero()[0] for d in range(0, self.domain.dim) ]
            gdofs = np.ravel_multi_index(cartesian(ldofs).transpose(), self.domain.nbfuns)

            B[np.ix_(*(gdofs,)*2)] += Bpart

        # compute element matrices and add contribution to A
        for el1 in range(0, self.domain.nelem()):
            el1_mulidx = np.unravel_index(el1, gridshape)
            slices1 = tuple([slice(nip_el[d]*el1_mulidx[d], nip_el[d]*el1_mulidx[d]+nip_el[d]) for d in range(0, self.domain.dim)])
            el1_ip = [ self.quadrature.ip[d][slices1[d]] for d in range(0, self.domain.dim) ]
            el1_xip = [ self.domain.eval(el1_ip, d) for d in range(self.domain.dim) ]

            for el2 in range(0, self.domain.nelem()):
                # get kronecker xips on each element 
                el2_mulidx = np.unravel_index(el2, gridshape)
                slices2 = tuple([slice(nip_el[d]*el2_mulidx[d], nip_el[d]*el2_mulidx[d]+nip_el[d]) for d in range(0, self.domain.dim)])
                el2_ip = [ self.quadrature.ip[d][slices2[d]] for d in range(0, self.domain.dim) ]
                el2_xip = [ self.domain.eval(el2_ip, d) for d in range(self.domain.dim) ]

                # compute the kernel on the element
                Gel = self.kernel(
                  _kern_pts_to_mulidx(el2_xip),
                  _kern_pts_to_mulidx(el1_xip), self.data)
            
                # precompute element basis matrices
                Bi = Bel[0][el1_mulidx[0]] * self.quadrature.weights[0][slices1[0]]
                for k in range(1, self.domain.dim):
                    Bi = np.kron(Bi, Bel[k][el1_mulidx[k]] * self.quadrature.weights[k][slices1[k]])
                Bi = Bi * Jel[el1]

                # precompute element basis matrices
                Bj = Bel[0][el2_mulidx[0]] * self.quadrature.weights[0][slices2[0]]
                for k in range(1, self.domain.dim):
                    Bj = np.kron(Bj, Bel[k][el2_mulidx[k]] * self.quadrature.weights[k][slices2[k]])
                Bj = Bj * Jel[el2]
               
                # assemble A on the element
                Apart = Bj @ Gel @ Bi.transpose() # ich muss hier zwei mal uber die elemente summieren!!!!

                ldofs1 = [ self.B[d][:,nip_el[d]*el1_mulidx[d]].nonzero()[0] for d in range(0, self.domain.dim) ]
                ldofs2 = [ self.B[d][:,nip_el[d]*el2_mulidx[d]].nonzero()[0] for d in range(0, self.domain.dim) ]

                gdofs1 = np.ravel_multi_index(cartesian(ldofs1).transpose(), self.domain.nbfuns)
                gdofs2 = np.ravel_multi_index(cartesian(ldofs2).transpose(), self.domain.nbfuns)
                
                A[np.ix_(*(gdofs2,gdofs1))] += Apart

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
