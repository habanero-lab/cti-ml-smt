import time
import sys
import numpy as np
import os


def get_1d_array(N: int, step: float):
    return np.arange(0, N * step, step, np.float64)


def get_2d_array(NI: int, NJ: int, step: float):
    assert NI == NJ
    A = np.arange(0, (NI * NJ) * step, step, np.float64).reshape(NI, NJ)
    for i in range(0, NI):
        tmp = np.arange(0, i+1, 1, np.float64)
        A[i, 0:i+1] = -(tmp[0:i+1] % NI) / NI + 1
        A[i, i+1:NI] = np.float64(0)
        A[i, i] = np.float64(1)

    # Make the matrix positive semi-definite.
    # not necessary for LU, but using same code as cholesky
    B = np.zeros((N, N), dtype=np.float64)

    B[0:NI, 0:NI] = np.dot(A[0:NI, 0:NI], A[0:NI, 0:NI].T)
    A[0:NI, 0:NI] = B[0:NI, 0:NI]
    return A


if __name__ == "__main__":
    kernel_type = os.getenv('KERNEL_TYPE', 'pydd')
    if kernel_type == 'pydd':
        import kernel
        ludcmp = kernel.ludcmp
    elif kernel_type == 'numba':
        import kernel
        ludcmp = kernel.ludcmp

    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1200

    A = get_2d_array(N, N, 0.01)
    b = get_1d_array(N, 0.02)
    x = get_1d_array(N, 0.03)
    y = get_1d_array(N, 0.04)

    t1 = time.time()
    ludcmp(A, b, x, y, N)
    t2 = time.time()

    ans = x[N//2]
    print('Ans:', ans)
    print("[Elapsed Kernel Time]: ", (t2 - t1))
