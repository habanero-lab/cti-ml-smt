import os
import argparse
import numpy as np
from time import time
from scipy.io import mmread
import kernel


def parallel_pr_nibble_pydd(seeds, adj, alpha, epsilon):
    np_alpha = np.float64(alpha)
    np_epsilon = np.float64(epsilon)
    np_seeds = np.array(seeds).astype(np.int32)
    degrees = np.asarray(adj.sum(axis=-1)).squeeze().astype(np.int32)
    num_nodes = np.int32(adj.shape[0])
    adj_indices = adj.indices
    adj_indptr = adj.indptr

    out = kernel.func0_unopt(np_seeds, degrees, np_alpha,
                             np_epsilon, num_nodes, adj_indices, adj_indptr)
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-seeds', type=int, default=50)
    parser.add_argument('--alpha', type=float, default=0.15)
    parser.add_argument('--pnib-epsilon', type=float, default=1e-6)
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()

    # !! In order to check accuracy, you _must_ use these parameters !!
    assert args.num_seeds == 50
    assert args.alpha == 0.15
    assert args.pnib_epsilon == 1e-6

    return args


if __name__ == "__main__":
    kernel_type = os.getenv('KERNEL_TYPE', 'pydd')
    if kernel_type == 'pydd':
        _parallel_pr_nibble = parallel_pr_nibble_pydd
    elif kernel_type == 'numba':
        _parallel_pr_nibble = parallel_pr_nibble_pydd

    args = parse_args()

    adj = mmread('data/jhu.mtx').tocsr()

    pnib_seeds = np.array(range(args.num_seeds))

    t = time()
    pnib_scores = _parallel_pr_nibble(
        pnib_seeds, adj, args.alpha, args.pnib_epsilon)
    t2 = time()
    assert pnib_scores.shape[0] == adj.shape[0]
    assert pnib_scores.shape[1] == len(pnib_seeds)
    print("[Elapsed Kernel Time]: ", (t2 - t))

    if args.validate:
        from scipy.stats import spearmanr

        PNIB_THRESH = 0.999

        def compute_score(targets, scores):
            assert targets.shape == scores.shape
            n = scores.shape[1]
            return min([spearmanr(scores[:, i], targets[:, i]).correlation for i in range(n)])

        pnib_targets = np.loadtxt('data/pnib_target.txt.gz')
        pnib_score = compute_score(pnib_targets, pnib_scores)

        p = "PASS" if pnib_score > PNIB_THRESH else "FAIL"
        print(f'Pass: {p}, score: {pnib_score}')
