import pytest
from tensiga.iga.fspan import *
from tensiga.iga.bfun import *
from tensiga.iga.bfuns import *
from tensiga.iga.dbfun import *
from tensiga.iga.dbfuns import *

class TestIga:
    def test_fspan(self):
        u = 2.5
        p = 2
        U = np.array([0,0,0,1,2,3,4,4,5,5,5], dtype=np.float64)
        assert fspan(2.5, p, U) == 4
        assert fspan(0., p, U) == 2
        assert fspan(1., p, U) == 3
        assert fspan(4., p, U) == 7
        assert fspan(5., p, U) == 7

    def test_bfun(self):
        U = np.array([0., 0., 0., 1., 2., 3., 4., 4., 5., 5., 5.]);
        p = 2

        assert 1./8 == pytest.approx(bfun(2, 5./2, p, U))
        assert 6./8 == pytest.approx(bfun(3, 5./2, p, U))
        assert 1./8 == pytest.approx(bfun(4, 5./2, p, U))
        assert 1.0 ==  pytest.approx(bfun(0, 0., p, U))
        assert 1.0 ==  pytest.approx(bfun(7, 5., p, U))

    def test_bfuns(self):
        U = np.array([0., 0., 0., 1., 2., 3., 4., 4., 5., 5., 5.]);
        p = 2
        u = 5./2

        i = fspan(u, p, U)
        N = bfuns(i, u, p, U)

        assert 1./8 == pytest.approx(N[0])
        assert 6./8 == pytest.approx(N[1])
        assert 1./8 == pytest.approx(N[2])

    def test_dbfun(self):
        n = 3
        u = 5./2
        p = 2
        U = np.array([0,0,0,1,2,3,4,4,5,5,5], dtype=np.float64)
        i = fspan(u, p, U)
        dN = dbfun(i, n, u, p, U)

        0.125 == pytest.approx(dN[0])
        0.500 == pytest.approx(dN[1])
        1.000 == pytest.approx(dN[2])
        0.000 == pytest.approx(dN[3])

    def test_dbfuns(self):
        ders_ref = np.array([[  0.125,  0.750,  0.125 ],
                             [ -0.500,  0.000,  0.500 ],
                             [  1.000, -2.000,  1.000 ],
                             [  0.000,  0.000,  0.000 ]])

        n = 3
        u = 5./2
        p = 2
        U = np.array([0,0,0,1,2,3,4,4,5,5,5], dtype=np.float64)
        i = fspan(u, p, U)

        ders = dbfuns(i, n, u, p, U)
        print(ders)
        print(ders_ref)
        assert all([a == pytest.approx(b) for a, b in zip(ders_ref.tolist(), ders.tolist())])
