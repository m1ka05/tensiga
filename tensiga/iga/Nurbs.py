import numpy as np
from math import sqrt
from tensiga.iga.Bspline import Bspline
from tensiga.iga.nwghtfun import nwghtfun

class Nurbs(Bspline):

    def __init__(self, dim, codim, kv, deg, ctrlpts):
        # weight ctrlpts of the corresponding Bspline
        for k, cp in enumerate(ctrlpts[:-1]):
            ctrlpts[k] = cp * ctrlpts[-1]

        Bspline.__init__(self, dim, codim+1, kv, deg, ctrlpts)
        self.physdim = codim

    def evalwrtwght(self, u, P):
        x = Bspline.evalwrt(self, u, P)
        R = self.wghtfun(u)
        return x * R

    def evalwrt(self, u, P):
        P = P * self.ctrlpts[-1]
        x = self.evalwrtwght(u, P)
        return x

    def eval(self, u, k=0):
        if not isinstance(u, list):
            u = [u]

        return self.evalwrtwght(u, self.ctrlpts[k])

    def wghtfun(self, u):
        return nwghtfun(self, u)

    def jacobian(self, u):
        if not isinstance(u, list):
            u = [u]

        if not self.dim == self.physdim:
            raise NotImplementedError('Bspline: non-square Jacobian!')

        B = self.bfunsops(u)
        dB = self.dbfunsops(u, 1)

        if self.dim == 1:
            Adxdu = self.ctrlpts[0] @ dB[0]
            Adwdu = self.ctrlpts[1] @ dB[0]
            Cx = self.eval(u, 0)
            R = self.pop(u)

            J = ((Adxdu - Adwdu*Cx)*R)

        elif self.dim == 2:
            Adxdu = self._mat_dot_multi_sp(self.ctrlpts[0],[dB[0],  B[1]])
            Adxdv = self._mat_dot_multi_sp(self.ctrlpts[0],[ B[0], dB[1]])

            Adydu = self._mat_dot_multi_sp(self.ctrlpts[1],[dB[0],  B[1]])
            Adydv = self._mat_dot_multi_sp(self.ctrlpts[1],[ B[0], dB[1]])

            Adwdu = self._mat_dot_multi_sp(self.ctrlpts[2],[dB[0],  B[1]])
            Adwdv = self._mat_dot_multi_sp(self.ctrlpts[2],[ B[0], dB[1]])

            Cx = self.eval(u, 0)
            Cy = self.eval(u, 1)
            R = self.wghtfun(u)

            # ravel s.t. a_ij -> a_I
            dxdu = ((Adxdu - Adwdu*Cx)*R).reshape(-1)
            dxdv = ((Adxdv - Adwdv*Cx)*R).reshape(-1)
            dydu = ((Adydu - Adwdu*Cy)*R).reshape(-1)
            dydv = ((Adydv - Adwdv*Cy)*R).reshape(-1)

            # compute the determinant of the jacobian
            J = (dxdu * dydv) - (dxdv * dydu)

            # reshape J_{I} back to J_ijk
            J = J.reshape(u[0].shape + u[1].shape)

        elif self.dim == 3:
            Adxdu = self._mat_dot_multi_sp(self.ctrlpts[0], [dB[0],  B[1],  B[2]])
            Adxdv = self._mat_dot_multi_sp(self.ctrlpts[0], [ B[0], dB[1],  B[2]])
            Adxdw = self._mat_dot_multi_sp(self.ctrlpts[0], [ B[0],  B[1], dB[2]])

            Adydu = self._mat_dot_multi_sp(self.ctrlpts[1], [dB[0],  B[1],  B[2]])
            Adydv = self._mat_dot_multi_sp(self.ctrlpts[1], [ B[0], dB[1],  B[2]])
            Adydw = self._mat_dot_multi_sp(self.ctrlpts[1], [ B[0],  B[1], dB[2]])

            Adzdu = self._mat_dot_multi_sp(self.ctrlpts[2], [dB[0],  B[1],  B[2]])
            Adzdv = self._mat_dot_multi_sp(self.ctrlpts[2], [ B[0], dB[1],  B[2]])
            Adzdw = self._mat_dot_multi_sp(self.ctrlpts[2], [ B[0],  B[1], dB[2]])

            Adwdu = self._mat_dot_multi_sp(self.ctrlpts[3], [dB[0],  B[1],  B[2]])
            Adwdv = self._mat_dot_multi_sp(self.ctrlpts[3], [ B[0], dB[1],  B[2]])
            Adwdw = self._mat_dot_multi_sp(self.ctrlpts[3], [ B[0],  B[1], dB[2]])

            Cx = self.eval(u, 0)
            Cy = self.eval(u, 1)
            Cz = self.eval(u, 2)
            R = self.wghtfun(u)

            # ravel s.t. a_ij -> a_I
            dxdu = ((Adxdu - Adwdu*Cx)*R).reshape(-1)
            dxdv = ((Adxdv - Adwdv*Cx)*R).reshape(-1)
            dxdw = ((Adxdw - Adwdw*Cx)*R).reshape(-1)

            dydu = ((Adydu - Adwdu*Cy)*R).reshape(-1)
            dydv = ((Adydv - Adwdv*Cy)*R).reshape(-1)
            dydw = ((Adydw - Adwdw*Cy)*R).reshape(-1)

            dzdu = ((Adzdu - Adwdu*Cz)*R).reshape(-1)
            dzdv = ((Adzdv - Adwdv*Cz)*R).reshape(-1)
            dzdw = ((Adzdw - Adwdw*Cz)*R).reshape(-1)

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
    # define spline data
    dim = 3;
    codim = 3;
    deg = [2, 1, 1]
    kv = [ np.array([0., 0., 0., 1., 1., 1.]),
           np.array([0., 0., 1., 1.]),
           np.array([0., 0., 1., 1.]) ]

    # this is using the numpy ordering
    ctrlpts = []
    cp_shape = (3,2,2)
    x = [ 1., 1., .6, .6, 1., 1., .6, .6, .0, .0, .0, .0 ]
    y = [ 0., 0., 0., 0., 1., 1., .6, .6, 1., 1., .6, .6 ]
    z = [ 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1. ]
    w = [ 1., 1., 1., 1., 1./sqrt(2.), 1./sqrt(2.), 1./sqrt(2.), 1./sqrt(2.), 1., 1., 1., 1. ]
    ctrlpts.append(np.array(x, dtype=np.float).reshape(cp_shape))
    ctrlpts.append(np.array(y, dtype=np.float).reshape(cp_shape))
    ctrlpts.append(np.array(z, dtype=np.float).reshape(cp_shape))
    ctrlpts.append(np.array(w, dtype=np.float).reshape(cp_shape))

    # init bspline object
    spline = Nurbs(dim, codim, kv, deg, ctrlpts)
