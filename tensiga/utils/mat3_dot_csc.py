import numpy as np
from scipy.sparse import csc_matrix

def mat3_dot_csc(A, B):
    """ Contracts a dense `np.array` :math:`A_{ijk}` with a sparse
    `scipy.sparse.csc_matrix()` :math:`B_ij` along the last and first axis
    of :math:`A_ijk` and :math:`B_ij`, respectively.

    Explictly the expression :math:`C_ijl = A_ijk B_kl` is being computed.
    A dense matrix :math:`B_ij` might be also passed as the second argument,
    if the priority is the performance and not the memory, i.e. B is small
    and a temporary variable.
    """

    C = np.zeros((A.shape[0], A.shape[1], B.shape[1]))
    for k in range(A.shape[1]):
        C[:,k,:] = A[:,k,:] @ B

    '''
    def custom_at_operator(A, B, k):
        return A[:,k,:] @ B

    custom_at_operator_vectorized = np.vectorize(custom_at_operator, otypes=[np.ndarray])
    custom_at_operator_vectorized.excluded.add(0)
    custom_at_operator_vectorized.excluded.add(1)

    out = custom_at_operator_vectorized(A, B, range(A.shape[1]))
    for k in range(A.shape[1]):
        C[:,k,:] = out[k]
    '''
    return C


if __name__ == '__main__':
    from time import time
    from scipy.sparse import random
    from utils.mat_dot_sp import mat_dot_sp

    A = np.random.rand(300,300,300)
    B = csc_matrix(random(300, 1200, density=0.01))

    tstart = time()
    C1 = mat3_dot_csc(A, B)
    C1 = mat3_dot_csc(A, B)
    C1 = mat3_dot_csc(A, B)
    tstop = time()
    print(tstop-tstart)

    tstart = time()
    C2 = (A.reshape(300**2, 300) @ B).reshape(300,300,1200)
    C2 = (A.reshape(300**2, 300) @ B).reshape(300,300,1200)
    C2 = (A.reshape(300**2, 300) @ B).reshape(300,300,1200)
    tstop = time()
    print(tstop-tstart)

    tstart = time()
    C3 = mat_dot_sp(A,B)
    C3 = mat_dot_sp(A,B)
    C3 = mat_dot_sp(A,B)
    tstop = time()
    print(tstop - tstart)
