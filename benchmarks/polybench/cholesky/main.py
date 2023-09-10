import time
import sys
import numpy as np
import os


def get_2d_array(NI: int, NJ: int, step: float):
    assert NI == NJ
    A = np.zeros((NI, NI), dtype=np.float64)
    for i in range(0, NI):
        tmp = np.arange(0, i+1, 1, np.float64)
        A[i, 0:i+1] = -(tmp[0:i+1] % NI) / NI + 1
        A[i, i+1:NI] = np.float64(0)
        A[i, i] = np.float64(1)

    # Make the matrix positive semi-definite.
    B = np.zeros_like(A)

    for r in range(0, NI):
        B[r, 0:NI] += (A[r, 0:NI] * A[0:NI, 0:NI]).sum(axis=1)

    A[0:NI, 0:NI] = B[0:NI, 0:NI]
    return A


if __name__ == "__main__":
    kernel_type = os.getenv('KERNEL_TYPE', 'pydd')
    if kernel_type == 'pydd':
        import kernel
        cholesky = kernel.cholesky
    elif kernel_type == 'numba':
        import kernel
        cholesky = kernel.cholesky

    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1100

    A = get_2d_array(N, N, 0.01)
    t1 = time.time()
    cholesky(A, N)
    t2 = time.time()

    ans = A[N//2, N//2]
    print('Ans:', ans)
    print("[Elapsed Kernel Time]: ", (t2 - t1))
