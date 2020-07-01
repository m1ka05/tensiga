import numpy as np
from scipy.sparse import csc_matrix
from tensiga.iga.ktsinsop import ktsinsop
from tensiga.iga.auxkv import auxkv
from tensiga.iga.twoscaleop import twoscaleop
from tensiga.iga.delevop import delevop
from tensiga.iga.bfunsop import bfunsop
from tensiga.iga.dbfunsop import dbfunsop
from tensiga.iga.gpnts import gpnts
from tensiga.utils.mat3_dot_csc import mat3_dot_csc
from tensiga.utils.mat_dot_sp import mat_dot_sp


class Bspline:
    """A data container for a Bspline geometric object and an interface to
    applicable geometric modules.

    Attributes:
        dim (int) : dimension of the Bspline object
        codim (int) : dimension of the embedding space
        kv (list(np.array)) : list of knot vectors in each parametric direction
        deg (list(int)) : list of degrees for each parametric direction
        ctrlpts (list(np.array)) : list of control point coordinates for each
          coordinate in the embedding space
        nbfuns (np.array(int)) : number of basis functions in each direction
        nkts (np.array(int)) : knot vector length in each direction
    """

    def __init__(self, dim, codim, kv, deg, ctrlpts):
        self.dim = dim
        self.codim = codim
        self.deg = deg
        self.kv = [ np.array(kv_i, copy=True) for kv_i in kv  ]
        self.nkts = [ kv_i.size for kv_i in self.kv ]
        self.nbfuns = np.array(self.nkts) - np.array(self.deg) - 1
        self.ctrlpts = [ np.array(ctrlpts_i, copy=True) for ctrlpts_i in ctrlpts  ]

        self.__assert_kv_sanity()
        self.__assert_deg_sanity()
        self.__assert_ctrlpts_sanity()

        self.physdim = codim

    def __assert_kv_sanity(self):
        try:
            assert(len(self.kv) == self.dim)
        except AssertionError:
            print('Bspline: number of knot vectors does not match the Bspline dimension')
            exit()

    def __assert_deg_sanity(self):
        try:
            assert(len(self.deg) == self.dim)
        except AssertionError:
            print('Bspline: number of degrees does not match the Bspline dimension')
            exit()

    def __assert_ctrlpts_sanity(self):
        try:
            assert(len(self.ctrlpts) == self.codim)
        except AssertionError:
            print('Bspline: number of control point coordinates does not match the embedding space dimension')
            exit()

        try:
            self.nkts = [ kv_i.size for kv_i in self.kv ]
            self.nbfuns = np.array(self.nkts) - np.array(self.deg) - 1
            for cps in self.ctrlpts:
                assert(list(cps.shape) == self.nbfuns.tolist())
        except AssertionError:
            print('Bspline: provided control points', cps,
              'do not suite the corresponding knot vector!')
            exit()

    def _mat_dot_sp(self, A, B, axis=0):
        return mat_dot_sp(A, B, axis)

    def _mat_dot_multi_sp(self, A, B):
        C = mat_dot_sp(A, B[0], 0)
        for k in range(1, len(B)):
            C = mat_dot_sp(C, B[k], k)
        return C

    def nelem(self, k=None):
        """Computes the number of elements. If ``k = None`` compute total
        number of elements. For ``k=1,...,dim`` computes the univariate elements
        in the direction ``k``.

        Parameters:
            k (int) : parametric direction

        Returns:
            int : number of elements
        """
        if k == None:
            return np.prod([ np.unique(kv).size-1 for kv in self.kv ])
        else:
            return np.unique(self.kv[k]).size-1

    def elsz(self):
        """Computes the element sizes in the physical space. Element size is
        defined as the length of the element diagonal.

        Returns:
            array(float) : element sizes
        """
        unique_kv = [ np.unique(kv) for kv in self.kv ]
        phy_knot = [ self.eval(unique_kv, k) for k in range(self.dim)  ]
        mesh_shape = phy_knot[0].shape

        ##      + -- +  x'                                   ##
        ##     /    .|                                       ##
        ##    + -- + |                                       ##
        ##    |    | +                                       ##
        ##    |    |.                                        ##
        ##    + -- +         |x - x'| = h                    ##
        ##  x                                                ##

        nelem = self.nelem()
        h = [None] * nelem

        for el in range(nelem):
            x_indices = np.unravel_index(el, np.array(mesh_shape)-1)
            xp_indices = tuple([idx+1 for idx in x_indices])

            # x on diagonal
            x = np.array([ phy_knot[k][x_indices] for k in range(self.dim) ])

            # x' on diagonal
            xp = np.array([ phy_knot[k][xp_indices] for k in range(self.dim) ])

            h[el] = np.linalg.norm(x-xp)

        return h

    def bfunsop(self, u, k):
        (vals, (rows, cols)), shape = bfunsop(u, self.deg[k], self.kv[k])
        return csc_matrix((vals, (rows,cols)), shape=shape)

    def dbfunsop(self, u, n, k):
        (vals, (rows, cols)), shape = dbfunsop(n, u, self.deg[k], self.kv[k])
        return csc_matrix((vals, (rows, cols)), shape=shape)

    def bfunsops(self, u):
        return [ self.bfunsop(u[k], k) for k in range(self.dim) ]

    def dbfunsops(self, u, n):
        return [ self.dbfunsop(u[k], n, k) for k in range(self.dim) ]

    def eval(self, u, k=0):
        """ Evaluates the Bspline at ``u`` in the physical space in direction ``k``.

        Parameters:
            u (List(np.array(float))) : list of evaluation points in each
                parametric direction

        Returns:
            np.array(float) : coordinates at each grid point in physical
            direction ``k``.
        """
        return self.evalwrt(u, self.ctrlpts[k])

    def evalwrt(self, u, P):
        """ Evaluates the Bspline at ``u`` with respect to a set of control points ``P``.

        Parameters:
            u (List(np.array(float))) : list of evaluation points in each
                parametric direction
            P (np.array(float)) : control points
        Returns:
            np.array(float) : coordinates at each grid point in physical
            direction ``k``.
        """
        if not isinstance(u, list):
            u = [u]

        B = self.bfunsops(u)
        x = self._mat_dot_multi_sp(P, B)

        return x

    def deval(self, n, k, u):
        raise NotImplementedError('Bspline: arbitrary derivatives\
          not implemented yet!')

    def devalwrt(self, n, u, P):
        raise NotImplementedError('Bspline: arbitrary derivatives\
          not implemented yet!')

    def greville_knt(self, k):
        return gpnts(self.deg[k], self.kv[k])

    def greville_knts(self):
        return [ self.greville_knt(k) for k in range(self.dim) ]

    def href(self, u, k=0):
        if k >= self.dim:
            raise IndexError('Bspline: parametric dir for refinement exceeded!')

        A, U = ktsinsop(self.nbfuns[k], u, self.deg[k], self.kv[k])

        for d in range(self.codim):
            self.ctrlpts[d] = self._mat_dot_sp(self.ctrlpts[d], A, k)

        self.kv[k] = U
        self.nkts[k] = self.kv[k].size
        self.nbfuns[k] = self.nkts[k] - self.deg[k] - 1

    def pref(self, dp, k=0):
        U = self.kv[k]
        p = self.deg[k]

        Z, M = auxkv(U)

        V = np.repeat(Z, M+dp)
        q = p + dp

        A = twoscaleop(p, U, q, V)

        for d in range(self.codim):
            self.ctrlpts[d] = self._mat_dot_sp(self.ctrlpts[d], A, k)

        self.deg[k] = q
        self.kv[k] = V
        self.nkts[k] = self.kv[k].size
        self.nbfuns[k] = self.nkts[k] - self.deg[k] - 1

    def delev(self, dp, k=0):
        U = self.kv[k]
        p = self.deg[k]

        Z, M = auxkv(U)

        M[[0,-1]] += dp
        V = np.repeat(Z, M)
        q = p + dp

        A = delevop(p, U, q, V)

        for d in range(self.codim):
            self.ctrlpts[d] = self._mat_dot_sp(self.ctrlpts[d], A, k)

        self.deg[k] = q
        self.kv[k] = V
        self.nkts[k] = self.kv[k].size
        self.nbfuns[k] = self.nkts[k] - self.deg[k] - 1

    def jacobian(self, u):
        if not isinstance(u, list):
            u = [u]

        if not self.dim == self.physdim:
            raise NotImplementedError('Bspline: non-square Jacobian!')

        B = self.bfunsops(u)
        dB = self.dbfunsops(u, 1)

        if self.dim == 1:
            J = self.ctrlpts[0] @ dB[0]

        elif self.dim == 2:
            dxdu = self._mat_dot_multi_sp(self.ctrlpts[0],[dB[0],  B[1]])
            dxdv = self._mat_dot_multi_sp(self.ctrlpts[0],[ B[0], dB[1]])

            dydu = self._mat_dot_multi_sp(self.ctrlpts[1],[dB[0],  B[1]])
            dydv = self._mat_dot_multi_sp(self.ctrlpts[1],[ B[0], dB[1]])

            # ravel s.t. a_ij -> a_I
            dxdu = dxdu.reshape(-1)
            dxdv = dxdv.reshape(-1)
            dydu = dydu.reshape(-1)
            dydv = dydv.reshape(-1)

            # compute the determinant of the jacobian
            J = (dxdu * dydv) - (dxdv * dydu)

            # reshape J_{I} back to J_ijk
            J = J.reshape(u[0].shape + u[1].shape)

        elif self.dim == 3:
            dxdu = self._mat_dot_multi_sp(self.ctrlpts[0],[dB[0],  B[1],  B[2]])
            dxdv = self._mat_dot_multi_sp(self.ctrlpts[0],[ B[0], dB[1],  B[2]])
            dxdw = self._mat_dot_multi_sp(self.ctrlpts[0],[ B[0],  B[1], dB[2]])

            dydu = self._mat_dot_multi_sp(self.ctrlpts[1],[ dB[0],  B[1],  B[2]])
            dydv = self._mat_dot_multi_sp(self.ctrlpts[1],[  B[0], dB[1],  B[2]])
            dydw = self._mat_dot_multi_sp(self.ctrlpts[1],[  B[0],  B[1], dB[2]])

            dzdu = self._mat_dot_multi_sp(self.ctrlpts[2],[ dB[0],  B[1],  B[2]])
            dzdv = self._mat_dot_multi_sp(self.ctrlpts[2],[  B[0], dB[1],  B[2]])
            dzdw = self._mat_dot_multi_sp(self.ctrlpts[2],[  B[0],  B[1], dB[2]])

            # ravel s.t. a_ij -> a_I
            dxdu = dxdu.reshape(-1)
            dxdv = dxdv.reshape(-1)
            dxdw = dxdw.reshape(-1)

            dydu = dydu.reshape(-1)
            dydv = dydv.reshape(-1)
            dydw = dydw.reshape(-1)

            dzdu = dzdu.reshape(-1)
            dzdv = dzdv.reshape(-1)
            dzdw = dzdw.reshape(-1)

            # compute the determinant of the jacobian
            J = (dxdu * dydv * dzdw) +\
                (dxdv * dydw * dzdu) +\
                (dxdw * dydu * dzdv) -\
                (dxdw * dydv * dzdu) -\
                (dxdv * dydu * dzdw) -\
                (dxdu * dydw * dzdv)

            # reshape J_{I} back to J_ijk
            J = J.reshape(u[0].shape + u[1].shape + u[2].shape)
        else:
            pass # throw an appropriate exception

        return J

if __name__ == '__main__':
    from matplotlib import colors as c
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from tensiga.quadrature.glnint import glnint

    dim = 2;
    codim = 2;
    deg = [2, 1]

    kv = [ np.array([0., 0., 0., 1., 1., 1.]),
           np.array([0., 0., 1., 1.])]

    ctrlpts = [ np.array([[1.0, 0.6],
                          [1.0, 0.6],
                          [0.0, 0.0]]), # x

                np.array([[0.0, 0.0],
                          [1.0, 0.6],
                          [1.0, 0.6]])] # z

    # init primitive spline object
    spline = Bspline(dim, codim, kv, deg, ctrlpts)

    # refine the spline object
    u = np.linspace(0, 1, 20)[1:-1]
    v = np.linspace(0, 1, 6)[1:-1]
    spline.href(u, 0)
    spline.href(v, 1)

    # get the integration rule
    quadrature = glnint(spline.kv, spline.deg+np.array([1,1]))
    xip = spline.eval(quadrature.ip, 0)
    yip = spline.eval(quadrature.ip, 1)

    # compute jacobian
    J = spline.jacobian(quadrature.ip)

    # compute geometry
    ep = spline.kv
    x = spline.eval(ep, 0)
    y = spline.eval(ep, 1)

    # plot
    plt.pcolor(x, y, np.sqrt(x**2 + y**2), edgecolor='black')
    plt.plot(xip, yip, 'ko', markersize=1)
    plt.show()
