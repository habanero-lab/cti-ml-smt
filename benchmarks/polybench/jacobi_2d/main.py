import time
import sys
import numpy as np
import os


def get_2d_array(NI: int, NJ: int, step: float):
    return np.arange(0, (NI * NJ) * step, step, np.float64).reshape(NI, NJ)


if __name__ == "__main__":
    kernel_type = os.getenv('KERNEL_TYPE', 'pydd')
    if kernel_type == 'pydd':
        import kernel
        jacobi_2d = kernel.jacobi_2d
    elif kernel_type == 'numba':
        import kernel
        jacobi_2d = kernel.jacobi_2d

    TSTEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

    A = get_2d_array(N, N, 0.01)
    B = get_2d_array(N, N, 0.02)

    t1 = time.time()
    jacobi_2d(A, B, TSTEPS, N)
    t2 = time.time()

    ans = A[N//2, N//2]
    print('Ans:', ans)
    print("[Elapsed Kernel Time]: ", (t2 - t1))
