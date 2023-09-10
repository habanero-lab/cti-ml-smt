"""
Implementation of the bigClAM algorithm.

Throughout the code, we will use tho following variables

  * F refers to the membership preference matrix. It's in [NUM_PERSONS, NUM_COMMUNITIES]
   so index (p,c) indicates the preference of person p for community c.
  * A refers to the adjency matrix, also named friend matrix or edge set. It's in [NUM_PERSONS, NUM_PERSONS]
    so index (i,j) indicates is 1 when person i and person j are friends.
"""

import numpy as np
import pickle
import time
import os


if __name__ == "__main__":
    kernel_type = os.getenv('KERNEL_TYPE', 'pydd')
    if kernel_type == 'pydd':
        import kernel
    elif kernel_type == 'numba':
        import kernel

    adj = np.load('data/adj.npy')
    p2c = pickle.load(open('data/p2c.pl', 'rb'))

    A = adj.astype(np.int32)
    F = np.load('data/F.npy')

    t1 = time.time()
    F = kernel.train(A, F, 4000)
    t2 = time.time()

    print("[Elapsed Kernel Time]: ", (t2 - t1))

    print(F[0])
    print(F[1])
    print(F[2])
    print(F[-1])
