import numpy as np

def auxkv(U):
    """Computes the auxiliary knot vector :math:`Z` and the multiplicity vector
    :math:`M` of each breakpoint in :math:`Z`.

    Parameters:
        U (np.array(float)) : knot vector

    Returns:
        np.array(float),  np.array(float) : auxiliary knot vector :math:`Z` and
        multiplicity vector :math:`M`
    """
    return np.unique(U, return_counts=True)


if __name__ == '__main__':
    U = np.array([ 0, 0, 0, 1, 2, 3, 3, 4, 4, 4 ])

    Z, M = auxkv(U)

    print(U)
    print(Z)
    print(M)

