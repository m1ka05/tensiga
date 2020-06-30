import numpy as np

def mat_mul_vec(A, v, axis=-1):
    if len(A.shape) == 1:
        return A @ B

    A = np.moveaxis(A, axis, -1)
    Ashape = A.shape

    # this is equivalent to multiplying element-wise along the given axis
    A = A.reshape(np.prod(A.shape[:-1]), A.shape[-1])
    C = A * v

    C = C.reshape(Ashape)
    C = np.moveaxis(C, -1, axis)

    return C
