"""
    ipnsw/main.py
    
    Note to program performers:
        - Results should match those from program IP-NSW workflow (#16 in December release)
"""

import os
import pickle
import argparse
import numpy as np
from time import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-queries', type=int, default=512)
    parser.add_argument('--v-entry', type=int, default=82026)
    parser.add_argument('--ef', type=int, default=128)
    parser.add_argument('--n-results', type=int, default=10)
    args = parser.parse_args()

    # !! Need these parameters for correctness checking
    assert args.n_queries == 512
    assert args.v_entry == 82026
    assert args.ef == 128
    assert args.n_results == 10

    return args


def flatten(graph):
    key = np.array(list(graph.keys())).astype('int32')
    lst_values = list(graph.values())
    value = np.concatenate(lst_values).astype('int32')
    idx = np.cumsum(np.array([elem.size for elem in lst_values]))
    idx = np.append([0], idx).astype('int32')
    return key, value, idx


if __name__ == "__main__":
    kernel_type = os.getenv('KERNEL_TYPE', 'pydd')
    if kernel_type == 'pydd':
        import kernel
    elif kernel_type == 'numba':
        import kernel
        from numba.typed.typeddict import Dict
        from numba.core import types

    args = parse_args()

    # --
    # IO
    # Load graphs in adjacency list format.
    #
    # Specifically, each element of `graphs` is a dictionary whose key is a vertex ID
    # and whose value is a list of neighbor vertex IDs.

    graphs = pickle.load(open('data/music.graphs.pkl', 'rb'))
    # G0     = graphs[0]
    # Gs     = [graphs[3], graphs[2], graphs[1]]
    Gs = []
    for i in range(3, 0, -1):
        key, value, idx = flatten(graphs[i])
        print('graph', i, '# keys = ', key.size, '/ # values', value.size)

        if kernel_type == 'numba':
            k2i = Dict.empty(
                key_type=types.int32,
                value_type=types.int32
            )
        else:
            k2i = {}

        for j in range(key.shape[0]):
            k2i[key[j]] = j
        Gs.append([k2i, key, value, idx])
    key, value, idx = flatten(graphs[0])
    print('graph', 0, '# keys = ', key.size, '/ # values', value.size)
    k2i = np.empty(1000000, np.int32)
    for j in range(key.shape[0]):
        k2i[key[j]] = j
    G0 = [k2i, key, value, idx]
    data = np.fromfile('data/database_music100.bin',
                       dtype=np.float32).reshape(1_000_000, 100)

    queries = np.fromfile('data/query_music100.bin',
                          dtype=np.float32).reshape(10_000, 100)
    queries = queries[:args.n_queries]

    # --
    # Run

    DIST_FN_COUNTER = np.zeros(1, np.int32)
    t = time()
    scores = kernel.search_knn(
        G0[0], G0[2], G0[3],
        Gs[0][0], Gs[0][2], Gs[0][3],
        Gs[1][0], Gs[1][2], Gs[1][3],
        Gs[2][0], Gs[2][2], Gs[2][3],
        data,
        queries,
        args.v_entry,
        args.ef,
        args.n_results,
        DIST_FN_COUNTER
    )
    elapsed = time() - t
    print("[Elapsed Kernel Time]: ", elapsed)
