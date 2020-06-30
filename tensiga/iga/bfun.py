import numpy as np
from numba import njit, int64, float64 

@njit(float64(int64, float64, int64, float64[:]))
def bfun(i, u, p, U):
    """Evaluates a single bspline basis function :math:`B_{i,p}(u)`
    at the coordinate :math:`u`

    Parameters:
        i (int) : basis function index
        u (float) : evaluation point
        p (int)  : basis function degree
        U (np.array(float)) : knot vector

    Returns:
        float : scalar value of :math:`B_{i,p}(u)`
    """

    m = U.size - 1
    n = m - p - 1
    Nip = 0

    N = np.zeros(p+2)

    if (i == 0 and u == U[0]) or (i == n and u == U[m]):
        return 1 # = Nip

    if u < U[i] or u >= U[i+p+1]:
        return 0 # = Nip

    for j in range(0, p+1):
        if u >= U[i+j] and u < U[i+j+1]:
            N[j] = 1
        else:
            N[j] = 0

    for k in range(1, p+1):
        if N[0] == 0:
            saved = 0
        else:
            saved = ((u-U[i]) * N[0]) / (U[i+k] - U[i])

        for j in range(0, p-k+1):
            Uleft = U[i+j+1]
            Uright = U[i+j+k+1]

            if N[j+1] == 0:
                N[j] = saved
                saved = 0
            else:
                temp = N[j+1] / (Uright - Uleft)
                N[j] = saved + (Uright - u) * temp
                saved = (u-Uleft) * temp

    return N[0] # = Nip


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy.ma as ma

    u = np.linspace(0, 1, 1000)

    #mpl.rcParams['text.latex.preamble'] = \
    #r"\usepackage[sc]{mathpazo}"

    fig = plt.figure(figsize=(7,6), dpi=150)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    
    ax3.set_xlabel('$\\xi_1$')
    ax1.set_ylabel('$B^1_{i_k,0}(\\xi_1)$')
    ax2.set_ylabel('$B^1_{i_k,1}(\\xi_1)$')
    ax3.set_ylabel('$B^1_{i_k,2}(\\xi_1)$')
    
    ax1.set_xticks([0,0.25,0.5,0.75,1])
    ax2.set_xticks([0,0.25,0.5,0.75,1])
    ax3.set_xticks([0,0.25,0.5,0.75,1])
    
    ax1.grid(True, 'both', linewidth='0.5', alpha=0.35)
    ax2.grid(True, 'both', linewidth='0.5', alpha=0.35)
    ax3.grid(True, 'both', linewidth='0.5', alpha=0.35)
    ax1.minorticks_on()
    ax2.minorticks_on()
    ax3.minorticks_on()

    tol = 0.00075

    p = 0
    U = np.array([0,0.25,0.5,0.75,1])
    m = U.size - 1
    n = m - p - 1
    for i in range(0, n+1):
        B1 = [ bfun(i, x, p, U) for x in u ]
        B1 = ma.array(B1)
        MB1 = ma.masked_where((np.abs(u-0) < tol), B1)
        MB1 = ma.masked_where((np.abs(u-0.25) < tol), MB1)
        MB1 = ma.masked_where((np.abs(u-0.5) < tol), MB1)
        MB1 = ma.masked_where((np.abs(u-0.75) < tol), MB1)
        MB1 = ma.masked_where((np.abs(u-1) < tol), MB1)
        ax1.plot(u, MB1)
    ax1.text(0,0.8,'$B^1_{1,0}$')
    ax1.text(0.25,0.8,'$B^1_{2,0}$')
    ax1.text(0.5,0.8,'$B^1_{3,0}$')
    ax1.text(0.75,0.8,'$B^1_{4,0}$')


    p = 1
    U = np.array([0,0,0.25,0.5,0.75,1,1], dtype=np.float64)
    m = U.size - 1
    n = m - p - 1
    for i in range(0, n+1):
        B2 = [ bfun(i, x, p, U) for x in u ]
        ax2.plot(u, B2)
    offset = 0.025
    ax2.text(0.05,0.85,'$B^1_{1,1}$')
    ax2.text(0.25-offset,0.7,'$B^1_{2,1}$')
    ax2.text(0.5-offset,0.7,'$B^1_{3,1}$')
    ax2.text(0.75-offset,0.7,'$B^1_{4,1}$')
    ax2.text(0.89,0.85,'$B^1_{5,1}$')

    p = 2
    U = np.array([0,0,0.25,0.5,0.5,0.75,1,1], dtype=np.float64)
    m = U.size - 1
    n = m - p - 1
    BSUM = [ 0 for _ in u ]
    for i in range(0, n+1):
        B3 = [ bfun(i, x, p, U) for x in u ]
        BSUM = [ sum(x) for x in zip(B3, BSUM) ]
        B3 = ma.array(B3)
        MB3 = ma.masked_where((np.abs(u-1)) < tol, B3)
        MB3 = ma.masked_where(np.abs(u) < tol, MB3)
        ax3.plot(u, MB3)
    BSUM = ma.array(BSUM)
    MBSUM = ma.masked_where((np.abs(u-1)) < tol, BSUM)
    MBSUM = ma.masked_where(np.abs(u) < tol, MBSUM)
    ax3.plot(u, MBSUM, '--k', linewidth=0.5, alpha=0.6)
    ax3.text(0.195-1*offset,0.725,'$B^1_{1,2}$')
    ax3.text(0.335-offset,0.725,'$B^1_{2,2}$')
    ax3.text(0.505-offset,0.65,'$B^1_{3,2}$')
    ax3.text(0.665-offset,0.725,'$B^1_{4,2}$')
    ax3.text(0.805-offset,0.725,'$B^1_{5,2}$')

    plt.show()
    #fig.savefig("../report/images/bsbfun.pdf",bbox_inches='tight')
