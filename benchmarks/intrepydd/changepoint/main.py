"""    
Implementation of "Graph-Based Change Point Detection"
    https://arxiv.org/abs/1209.1625.pdf
"""

import os
import argparse
import numpy as np
import pandas as pd
from time import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--one-d', action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    kernel_type = os.getenv('KERNEL_TYPE', 'pydd')
    if kernel_type == 'pydd':
        import kernel
    elif kernel_type == 'numba':
        import kernel

    args = parse_args()

    # --
    # IO

    edges = pd.read_csv('data/yankee_edges.tsv', header=None, sep='\t').values

    # Dictionary of edges
    g = {}
    for src, trg in edges:
        src = np.int32(src)
        trg = np.int32(trg)
        if src in g:
            g[src].append(trg)
        else:
            g[src] = [trg]

        if trg in g:
            g[trg].append(src)
        else:
            g[trg] = [src]

    # --
    # Run changepoint detection

    for key in g:
        g[key] = np.array(g[key], dtype=np.int32)

    if kernel_type == 'numba':
        from numba.typed.typeddict import Dict
        from numba.core import types
        gg = Dict.empty(types.int32, types.int32[:])
        for k, v in g.items():
            gg[k] = v
        g = gg

    if args.one_d:
        def _scan_stat(g): return kernel.scan_stat(g, 0)
    else:
        _scan_stat = kernel.scan_stat_2d

    print('main.py: computing scan stat')
    t_start = time()
    R = _scan_stat(g)
    t_end = time()
    print("[Elapsed Kernel Time] :", (t_end - t_start))

    print(R)
