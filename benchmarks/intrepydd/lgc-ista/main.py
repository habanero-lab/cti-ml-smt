import os
import argparse
import numpy as np
from time import time
from scipy.io import mmread
import kernel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-seeds', type=int, default=50)
    parser.add_argument('--alpha', type=float, default=0.15)
    parser.add_argument('--ista-rho', type=float, default=1e-5)
    parser.add_argument('--ista-iters', type=int, default=50)
    args = parser.parse_args()

    # !! In order to check accuracy, you _must_ use these parameters !!
    assert args.num_seeds == 50
    assert args.alpha == 0.15
    assert args.ista_rho == 1e-5
    assert args.ista_iters == 50

    return args


def ista_pydd(seeds, adj, alpha, rho, iters):
    np_alpha = np.float64(alpha)
    np_rho = np.float64(rho)
    np_iters = np.int32(iters)
    np_seeds = np.array(seeds).astype(np.int32)
    d = np.asarray(adj.sum(axis=-1)).squeeze().astype(np.int32)
    adj_vals = adj.data.astype(np.float64)
    adj_cols = adj.indices
    adj_idxs = adj.indptr
    nrows = adj.shape[0]
    ncols = adj.shape[1]

    out = kernel.func1_unopt(np_seeds, d, np_alpha, np_rho,
                             np_iters, adj_vals, adj_cols, adj_idxs, nrows, ncols)
    return out


if __name__ == "__main__":
    kernel_type = os.getenv('KERNEL_TYPE', 'pydd')
    if kernel_type == 'pydd':
        _ista = ista_pydd
    elif kernel_type == 'numba':
        _ista = ista_pydd

    args = parse_args()

    adj = mmread('data/jhu.mtx').tocsr()

    ista_seeds = list(range(10 * args.num_seeds))

    t = time()
    ista_scores = _ista(ista_seeds, adj, alpha=args.alpha,
                        rho=args.ista_rho, iters=args.ista_iters)
    ista_elapsed = time() - t
    assert ista_scores.shape[0] == adj.shape[0]
    assert ista_scores.shape[1] == len(ista_seeds)
    print("[Elapsed Kernel Time]: ", ista_elapsed)
