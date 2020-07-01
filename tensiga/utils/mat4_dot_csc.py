import numpy as np
from scipy.sparse import csc_matrix
from tensiga.utils.mat3_dot_csc import mat3_dot_csc

def mat4_dot_csc(A, B):
    """ Contracts a dense `np.array` :math:`A_{ijk}` with a sparse
    `scipy.sparse.csc_matrix()` :math:`B_ij` along the last and first axis
    of :math:`A_ijk` and :math:`B_ij`, respectively.

    Explictly the expression :math:`C_ijl = A_ijk B_kl` is being computed.
    A dense matrix :math:`B_ij` might be also passed as the second argument,
    if the priority is the performance and not the memory, i.e. B is small
    and a temporary variable.
    """

    C = np.zeros((A.shape[0], A.shape[1], A.shape[2], B.shape[1]))
    for k in range(A.shape[2]):
        C[:,:,k,:] = mat3_dot_csc(A[:,:,k,:], B)

    return C


if __name__ == '__main__':
    from time import time
    from scipy.sparse import random
    from tensiga.utils.mat_dot_sp import mat_dot_sp

    A = np.random.rand(40,40,40,300)
    B = csc_matrix(random(300, 200, density=0.1))

    tstart = time()
    C = mat4_dot_csc(A, B)
    C = mat4_dot_csc(A, B)
    C = mat4_dot_csc(A, B)
    tstop = time()
    print(tstop - tstart)

    tstart = time()
    C2 = (A.reshape(40**3, 300) @ B).reshape(40,40,40,200)
    C2 = (A.reshape(40**3, 300) @ B).reshape(40,40,40,200)
    C2 = (A.reshape(40**3, 300) @ B).reshape(40,40,40,200)
    tstop = time()
    print(tstop - tstart)

    tstart = time()
    C3 = mat_dot_sp(A,B)
    C3 = mat_dot_sp(A,B)
    C3 = mat_dot_sp(A,B)
    tstop = time()
    print(tstop - tstart)
