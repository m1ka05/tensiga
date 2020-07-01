import numpy as np

def mat_dot_sp(A, B, axis=-1):
    if len(A.shape) == 1:
        return A @ B

    A = np.moveaxis(A, axis, -1)
    Ashape = A.shape

    # this is equivalent to contracting A with B along the given axis
    A = A.reshape(np.prod(A.shape[:-1]), A.shape[-1])
    C = A @ B

    C = C.reshape(Ashape[:-1] + B.shape[1:])
    C = np.moveaxis(C, -1, axis)

    return C
