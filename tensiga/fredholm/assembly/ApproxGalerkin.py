import numpy as np
from scipy.sparse import kron
from scipy.sparse.linalg import LinearOperator, spsolve, splu
from sksparse.cholmod import cholesky
from time import time
from scipy.sparse.linalg import splu, inv
from scipy.sparse.linalg import spsolve_triangular
from tensiga.iga.Nurbs import Nurbs
from tensiga.iga.Bspline import Bspline
from tensiga.iga.BsplineInterp import BsplineInterp
from tensiga.fredholm.HilbertSchmidtKern import _kern_mat_to_tens, _kern_pts_to_mulidx
from tensiga.utils.mat_dot_sp import mat_dot_sp
from tensiga.utils.mat_mul_vec import mat_mul_vec
from tensiga.utils.splu_solve_mat import splu_solve_mat
from tensiga.utils.sp_tri_solve_mat import sp_tri_solve_mat
from sklearn.utils.extmath import cartesian

_LinOpInit_wthf = False
_nonParaTime = 0.0
_paraTime = 0.0
n_it_count = 0
it_time = 0

class ApproxGalerkin:
    def niter(self):
        global n_it_count
        return n_it_count

    def __init__(self, domain, idomain, quadrature, kernel, data, precond=False):
        ## initialize data
        self.domain = domain
        self.idomain = idomain
        self.quadrature = quadrature
        self.kernel = kernel
        self.data = data
        self.precond = precond

        # compute integration points in physical domain
        self.xip = [ self.domain.eval(self.quadrature.ip, k)
          for k in range(self.domain.dim) ]

        # compute basis operators at integration points
        self.B = self.domain.bfunsops(self.quadrature.ip)

        # compute interpolatory basis operators at integration points
        self.Bt = self.idomain.bfunsops(self.quadrature.ip)

        # init Bspline interpolant
        self.interp = BsplineInterp(self.domain, self.idomain)

        # compute jacobian ## this is stupid, need to get rid of it! Eats ram
        #self.J = self.domain.jacobian(self.quadrature.ip)

    def eval_ef(self, ep, ev, k):
        # ev into tensor of coeffs (each columnt corresponds to an ev)
        nev = ev.shape[1]
        ev = [ ev[:,k].reshape(self.domain.nbfuns) for k in range(nev) ]

        # due to pure B-spline basis funs implementation, I need this weighting
        # to get the actual basis funs
        J = np.sqrt(self.domain.jacobian(ep)).view()

        if self.precond == False:
            ef = self.domain.evalwrt(ep, ev[k])/J
        else:
            # precompute univariate mass matrices for B
            Bj = [None] * self.domain.dim
            for j in range(self.domain.dim):
                Bj[j] = self.B[j] @ self.B[j].multiply(self.quadrature.weights[j]).transpose()

            # compute preconditioners
            if self.precond == True:
                L = [cholesky(Bj_j.tocsc(), ordering_method='natural').L().transpose().tocsr()
                  for Bj_j in Bj]

            for j in range(0, self.domain.dim):
                ev[k] = sp_tri_solve_mat(L[j], ev[k], j, lower=False)

            ef = self.domain.evalwrt(ep, ev[k])/J

        return ef

    def normalize_ev(self, ev, k=None):
        if k == None:
            nev = ev.shape[1]
            factors = [0.] * nev
            for k in range(nev):
                ef = self.eval_ef(self.quadrature.ip, ev, k)
                factors[k] = np.sum(self.quadrature.W * (ef**2))
                ev[:,k] /= np.sqrt(factors[k])
        else:
            factors = 0
            ef = self.eval_ef(self.quadrature.ip, ev, k)
            factors = np.sum(self.quadrature.W * (ef**2))
            ev[:,k] /= np.sqrt(factors)

        return factors

    '''
    def l2norm_interp(self, kernel):
        # set up integration
        xip = _kern_pts_to_mulidx(self.xip)
        nxip = xip.shape[0]

        J = self.domain.jacobian(self.quadrature.ip).view().reshape(-1)
        Jsq = np.sqrt(J)

        W = self.quadrature.W.reshape(-1)

        # integrate ref l2 norm, (Gi1*(Ji*J1)**1/2) * Ji * J1 * Wi * W1 + ...
        Gi = np.zeros(nxip)
        for k in range(nxip):
            Gi += (kernel(xip, xip[k].reshape(1,xip.shape[1]),
              self.data).reshape(-1) * Jsq * Jsq[k])**2 * W * J * J[k] * W[k]

        print(np.sum(Gi))
    '''

    def interp_l2_error(self):
        # precompute some stuff
        cov = self.kernel(
          _kern_pts_to_mulidx(self.xip),
          _kern_pts_to_mulidx(self.xip), self.data)
        jac = self.domain.jacobian(self.quadrature.ip).view().reshape(-1)
        sq_jac = np.sqrt(jac)

        ## compute ref norm
        f_ref = cov * sq_jac[:, np.newaxis] * sq_jac[:, np.newaxis].transpose()
        f = (f_ref)**2
        f = _kern_mat_to_tens(f, self.xip[0].shape, self.xip[0].shape)
        f = mat_mul_vec(f, self.quadrature.weights[0], axis=0)
        f = mat_mul_vec(f, self.quadrature.weights[0], axis=self.domain.dim)
        for k in range(1, self.domain.dim):
            f = mat_mul_vec(f, self.quadrature.weights[k], axis=k)
            f = mat_mul_vec(f, self.quadrature.weights[k], axis=k+self.domain.dim)

        # scaling from geo map
        f = f.reshape(jac.shape * 2)
        l2_ref = np.sum(f * jac[:, np.newaxis] * jac[:, np.newaxis].transpose())

        ## interpolant
        # compute interpolant
        cov = self.kernel(
          _kern_pts_to_mulidx(self.interp.gp_phys),
          _kern_pts_to_mulidx(self.interp.gp_phys), self.data)

        sq_jac = np.sqrt(self.domain.jacobian(self.interp.gp)).view().reshape(-1)
        f_j = cov * sq_jac[:, np.newaxis] * sq_jac[:, np.newaxis].transpose()
        f = _kern_mat_to_tens(f_j, self.interp.gp_phys[0].shape,
          self.interp.gp_phys[0].shape)

        g = self.interp.direct_gkl_eval(f)

        # evaluate interpolant
        f_approx = self.interp.eval_2d_interpolant(g, self.quadrature.ip, self)

        ## integrate the square of abs error diff (l2 norm of abs err)
        f_ref = _kern_mat_to_tens(f_ref, self.xip[0].shape, self.xip[0].shape)
        f = np.abs(f_ref-f_approx)**2
        f = mat_mul_vec(f, self.quadrature.weights[0], axis=0)
        f = mat_mul_vec(f, self.quadrature.weights[0], axis=self.domain.dim)
        for k in range(1, self.domain.dim):
            f = mat_mul_vec(f, self.quadrature.weights[k], axis=k)
            f = mat_mul_vec(f, self.quadrature.weights[k], axis=k+self.domain.dim)

        # scaling from geo map
        f = f.reshape(jac.shape * 2)
        l2_err = np.sum(f * jac[:, np.newaxis] * jac[:, np.newaxis].transpose())

        return np.sqrt(l2_err/l2_ref)


    def direct(self):
        # compute kernel at greville abscissa
        A = self.kernel(
          _kern_pts_to_mulidx(self.interp.gp_phys),
          _kern_pts_to_mulidx(self.interp.gp_phys), self.data)

        # compute square root of jacobian dets at greville abscissa
        jac = np.sqrt(self.domain.jacobian(self.interp.gp)).view().reshape(-1)

        # f := Cov(x,y) * (DF(x)*DF(y)) ** 1/2
        A = A * jac[:, np.newaxis] * jac[:, np.newaxis].transpose() # J@Cov@J
        A = _kern_mat_to_tens(A, self.interp.gp_phys[0].shape,
          self.interp.gp_phys[0].shape)

        print('interpolation')
        self.interp.compute_lu()
        A = splu_solve_mat(self.interp.lu[0], A, 0)
        A = splu_solve_mat(self.interp.lu[0], A, self.idomain.dim)
        for k in range(1, self.idomain.dim):
            A = splu_solve_mat(self.interp.lu[k], A, k)
            A = splu_solve_mat(self.interp.lu[k], A, k+self.idomain.dim)

        Mk = [None] * self.domain.dim
        for k in range(self.domain.dim):
            Mk[k] = self.Bt[k].multiply(self.quadrature.weights[k]) @ self.B[k].transpose()

        print('contraction')
        # compute A by tensor contraction
        A = mat_dot_sp(A, Mk[0], 0)
        A = mat_dot_sp(A, Mk[0], self.domain.dim)
        for k in range(1, self.domain.dim):
            A = mat_dot_sp(A, Mk[k], k)
            A = mat_dot_sp(A, Mk[k], k+self.domain.dim)

        # precompute univariate mass matrices for B
        Bk = [None] * self.domain.dim
        for k in range(self.domain.dim):
            Bk[k] = self.B[k] @ self.B[k].multiply(self.quadrature.weights[k]).transpose()

        # compute B or the preconditioners
        if self.precond == True:
            L = [cholesky(Bk_k.tocsc(), ordering_method='natural').L().tocsr()
              for Bk_k in Bk]
            B = None

            print('preconditioning')
            # apply preconditioning
            for k in range(0, self.domain.dim):
                A = sp_tri_solve_mat(L[k], A, k)
                A = sp_tri_solve_mat(L[k], A, k+self.domain.dim)
        else:
            B = Bk[0]
            for k in range(1, self.domain.dim):
                B = kron(B, Bk[k])

        Ashape = (np.prod(self.domain.nbfuns),)*2
        A = A.reshape(Ashape) # reshape from A_{i1..id j1..jd} to A_{IJ}

        return A, B

    def exact_elementwise(self):
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
            Jel[el] = np.sqrt(self.J[slices].view().reshape(-1))
        
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
            Bi = Bi * Jel[el1] # actually its sqrt(J)

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
                Bj = Bel[0][el2_mulidx[0]] * self.quadrature.weights[0][slices2[0]]
                for k in range(1, self.domain.dim):
                    Bj = np.kron(Bj, Bel[k][el2_mulidx[k]] * self.quadrature.weights[k][slices2[k]])
                Bj = Bj * Jel[el2] # actually its sqrt(J)
               
                # assemble A on the element
                Apart = Bj @ Gel @ Bi.transpose()

                ldofs2 = [ self.B[d][:,nip_el[d]*el2_mulidx[d]].nonzero()[0] for d in range(0, self.domain.dim) ]
                gdofs2 = np.ravel_multi_index(cartesian(ldofs2).transpose(), self.domain.nbfuns)

                
                A[np.ix_(*(gdofs2,gdofs1))] += Apart

        return A, B
    
    def exact_kron(self):
        J = self.J.view().reshape(-1)
        G = self.kernel(
          _kern_pts_to_mulidx(self.xip),
          _kern_pts_to_mulidx(self.xip), self.data)

        # precompute basis matrices
        Bi = self.B[0].multiply(self.quadrature.weights[0])
        for k in range(1, self.domain.dim):
            Bi = kron(Bi, self.B[k].multiply(self.quadrature.weights[k]))
        Bi = Bi.tocsr()

        Bj = self.B[0]
        for k in range(1, self.domain.dim):
            Bj = kron(Bj, self.B[k])

        # assemble B
        B = Bi.tocsr() @ Bj.transpose().tocsc()

        # caution: Bi mutated
        Bi = Bi.multiply(np.sqrt(J))
        A = Bi.tocsr() @ G @ Bi.transpose().tocsc()

        return A, B

    '''
    def direct_elementwise(self):
        # compute mass matrices
        Mk = [None] * self.domain.dim
        for k in range(self.domain.dim):
            Mk[k] = self.Bt[k].multiply(self.quadrature.weights[k]) @ self.B[k].transpose()
        
        # do this element, row or columnwise
        # compute kernel at greville abscissa
        gp_phys_mulidx = _kern_pts_to_mulidx(self.interp.gp_phys)
        # compute square root of jacobian dets at greville abscissa
        jac = np.sqrt(self.domain.jacobian(self.interp.gp)).view().reshape(-1)

        self.interp.compute_lu()
        H = np.empty((np.prod(self.idomain.nbfuns), np.prod(self.idomain.nbfuns)))
        for k in range(np.prod(self.idomain.nbfuns)):
            cov = self.kernel(
              gp_phys_mulidx[k, np.newaxis],
              gp_phys_mulidx, self.data)


            # f := Cov(x,y) * (DF(x)*DF(y)) ** 1/2
            f = cov * jac[k, np.newaxis] * jac[:, np.newaxis].transpose() # J@Cov@J
            f = _kern_mat_to_tens(f, (1,1,1), self.interp.gp_phys[0].shape)

            Gr = splu_solve_mat(self.interp.lu[0], f, self.interp.idomain.dim)
            for k in range(1, self.idomain.dim):
                Gr = splu_solve_mat(self.interp.lu[k], Gr, k+self.interp.idomain.dim)

            Ar = mat_dot_sp(Gr, Mk[0], self.domain.dim)
            for k in range(1, self.domain.dim):
                Ar = mat_dot_sp(Ar, Mk[k], k+self.domain.dim)

            import pdb; pdb.set_trace()

        G = splu_solve_mat(self.interp.lu[0], H, 0)
        for k in range(1, self.idomain.dim):
            G = splu_solve_mat(self.interp.lu[k], G, k+self.interp.idomain.dim)

        A = mat_dot_sp(A, Mk[0], 0)
        for k in range(1, self.domain.dim):
            A = mat_dot_sp(A, Mk[k], k+self.domain.dim)

        # precompute univariate mass matrices for B
        Bk = [None] * self.domain.dim
        for k in range(self.domain.dim):
            Bk[k] = self.B[k] @ self.B[k].multiply(self.quadrature.weights[k]).transpose()

        # compute B or the preconditioners
        if self.precond == True:
            L = [cholesky(Bk_k.tocsc(), ordering_method='natural').L().tocsr()
              for Bk_k in Bk]
            B = None

            # apply preconditioning
            for k in range(0, self.domain.dim):
                A = sp_tri_solve_mat(L[k], A, k)
                A = sp_tri_solve_mat(L[k], A, k+self.domain.dim)
        else:
            B = Bk[0]
            for k in range(1, self.domain.dim):
                B = kron(B, Bk[k])

        Ashape = (np.prod(self.domain.nbfuns),)*2
        A = A.reshape(Ashape) # reshape from A_{i1..id j1..jd} to A_{IJ}

        return A, B
    '''

    def matrix_free(self):
        # precompute univariate mass matrices for B
        Bk = [None] * self.domain.dim
        for k in range(self.domain.dim):
            Bk[k] = self.B[k] @ self.B[k].multiply(self.quadrature.weights[k]).transpose()

        # compute B or the preconditioners
        if self.precond == True:
            L = [cholesky(Bk_k.tocsc(), ordering_method='natural').L().tocsr()
              for Bk_k in Bk]
            B = None
        else:
            B = Bk[0]
            for k in range(1, self.domain.dim):
                B = kron(B, Bk[k])

        # precompute stuff for A
        jac = np.sqrt(self.domain.jacobian(self.interp.gp))
        N = self.interp.compute_lu()

        Mk = [None] * self.domain.dim
        for k in range(self.domain.dim):
            Mk[k] = self.Bt[k].multiply(self.quadrature.weights[k]) @ self.B[k].transpose()

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
            it_time = time()
            it_nonParaTime = time()
            print('iter: ', n_it_count)

            # reshape vector of coeffs into a tensor
            v = v.view().reshape(self.domain.nbfuns)

            #print('... stage 1 in progress')
            # stage 1
            if self.precond == True:
                V = sp_tri_solve_mat(L[0].transpose().tocsr(), v, 0, lower=False)
                for k in range(1, self.domain.dim):
                    V = sp_tri_solve_mat(L[k].transpose().tocsr(), V, k, lower=False)
            else:
                V = v.view()

            #print('... stage 2 in progress')
            # stage 2
            X = mat_dot_sp(V, Mk[0].transpose(), 0)
            for k in range(1, self.domain.dim):
                X = mat_dot_sp(X, Mk[k].transpose(), k)

            #print('... stage 3 in progress')
            # stage 3
            Y = splu_solve_mat(N[0], X, 0, 'T')
            for k in range(1, self.domain.dim):
                Y = splu_solve_mat(N[k], Y, k, 'T')

            #print('... stage 4 in progress')
            # stage 4
            Yp = Y * jac

            _nonParaTime += time() - it_nonParaTime 
            it_paraTime = time()

            #print('... stage 5 in progress')
            # stage 5
            Zp = self.kernel(
                  _kern_pts_to_mulidx(self.interp.gp_phys),
                  _kern_pts_to_mulidx(self.interp.gp_phys),
                  Yp.reshape(-1),
                  self.data)
            Zp = Zp.view().reshape(jac.shape)

            _paraTime += time() - it_paraTime
            it_nonParaTime = time()

            #print('... stage 6 in progress')
            # stage 6
            Z = Zp * jac

            #print('... stage 7 in progress')
            # stage 7
            Y = splu_solve_mat(N[0], Z, 0)
            for k in range(1, self.domain.dim):
                Y = splu_solve_mat(N[k], Y, k)

            #print('... stage 8 in progress')
            # stage 8
            V = mat_dot_sp(Y, Mk[0], 0)
            for k in range(1, self.domain.dim):
                V = mat_dot_sp(V, Mk[k], k)

            #print('... stage 9 in progress')
            # stage 9
            if self.precond == True:
                # spsolve_triangualr
                vp = sp_tri_solve_mat(L[0], V, 0)
                for k in range(1, self.domain.dim):
                    vp = sp_tri_solve_mat(L[k], vp, k)
            else:
                vp = V.view()

            _nonParaTime += time() - it_nonParaTime 

            #print('iter time: ', time()-it_time)
            print(_nonParaTime/_paraTime)
            return vp.view().reshape(-1)

        Ashape = (np.prod(self.domain.nbfuns),)*2
        A = LinearOperator(Ashape, matvec=mv)

        return A, B
