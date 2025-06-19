import numpy as np
from numba import njit

@njit(parallel=True)
def matrix_multiply(A, B):
    C = np.zeros((A.shape[0], B.shape[1]), dtype=np.int32)
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                C[i, j] += A[i, k] * B[k, j]
    return C

if __name__ == "__main__":
    n = 256
    A = np.ones((n, n), dtype=np.int32)
    B = np.full((n, n), 2, dtype=np.int32)
    C = matrix_multiply(A, B)
    print(f"C[0][0] = {C[0, 0]}")
