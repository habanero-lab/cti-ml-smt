import os
import argparse
import numpy as np
from time import time
from scipy import sparse
from scipy.spatial.distance import cdist
import kernel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, default='data/cache')
    parser.add_argument('--n-docs', type=int, default=5000)
    parser.add_argument('--query-idx', type=int, default=100)
    parser.add_argument('--lamb', type=float, default=1)
    parser.add_argument('--max_iter', type=int, default=15)
    args = parser.parse_args()

    # !! In order to check accuracy, you _must_ use these parameters !!
    assert args.inpath == 'data/cache'
    assert args.n_docs == 5000
    assert args.query_idx == 100

    return args


def sinkhorn_wmd_pydd(r, c, vecs, lamb, max_iter):
    # I=(r > 0)
    sel = r.squeeze() > 0

    # r=r(I)
    r = r[sel].reshape(-1, 1).astype(np.float64)

    # M=M(I,:)
    M = cdist(vecs[sel], vecs).astype(np.float64)

    # x=ones(lenth(r), size(c,2)) / length(r)
    a_dim = r.shape[0]
    b_nobs = c.shape[1]
    x = np.ones((a_dim, b_nobs)) / a_dim

    # K=exp(-lambda * M)
    K = np.exp(- M * lamb)

    # while x changes: x=diag(1./r)*K*(c.*(1./(K’*(1./x))))

    ret = kernel.func0_unopt(K, M, r, x, max_iter,
                             c.data, c.indices, c.indptr, c.shape[1])
    return ret


if __name__ == "__main__":
    kernel_type = os.getenv('KERNEL_TYPE', 'pydd')
    if kernel_type == 'pydd':
        _sinkhorn_wmd = sinkhorn_wmd_pydd
    elif kernel_type == 'numba':
        _sinkhorn_wmd = sinkhorn_wmd_pydd

    args = parse_args()

    # --
    # IO

    docs = open(args.inpath + '-docs').read().splitlines()
    vecs = np.load(args.inpath + '-vecs.npy')
    mat = sparse.load_npz(args.inpath + '-mat.npz')

    # --
    # Prep

    # Maybe subset docs
    if args.n_docs:
        docs = docs[:args.n_docs]
        mat = mat[:, :args.n_docs]

    # --
    # Run

    # Get query vector
    r = np.asarray(mat[:, args.query_idx].todense()).squeeze()

    t = time()
    scores = _sinkhorn_wmd(r, mat, vecs, lamb=args.lamb,
                           max_iter=args.max_iter)
    elapsed = time() - t

    print("[Elapsed Kernel Time]: ", elapsed)
