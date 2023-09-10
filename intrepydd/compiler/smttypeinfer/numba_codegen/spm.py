import numpy as np
# from numba import prange
# from numba.typed.typedlist import List


def spm_mul_scalar(spm: tuple[np.ndarray, np.ndarray, np.ndarray, int], x):
    data = spm[0]
    ind = spm[1]
    indptr = spm[2]
    ncols = spm[3]

    out_data = data * x
    out_ind = np.copy(ind)
    out_indptr = np.copy(indptr)

    return (out_data, out_ind, out_indptr, ncols)


def spm_mul_dense(spm: tuple[np.ndarray, np.ndarray, np.ndarray, int], arr: np.ndarray):
    data = spm[0]
    ind = spm[1]
    indptr = spm[2]
    ncols = spm[3]
    nrows = len(indptr) - 1
    assert arr.ndim == 2 and \
        nrows == arr.shape[0] and ncols == arr.shape[1]

    out_data = np.empty_like(data)
    out_ind = np.empty_like(ind)
    out_indptr = np.empty_like(indptr)

    out_p = 0
    for r in range(nrows):
        out_indptr[r] = out_p
        for p in range(indptr[r], indptr[r + 1]):
            c = ind[p]
            v = data[p] * arr[r, c]
            if v != 0:
                out_data[out_p] = v
                out_ind[out_p] = c
                out_p += 1
    out_indptr[nrows] = out_p

    return (out_data[:out_p], out_ind[:out_p], out_indptr, ncols)

    # out_data = List([List([np.float64(0) for _ in range(0)])] * nrows)
    # out_ind = List([List([np.int32(0) for _ in range(0)])] * nrows)
    # out_indptr = np.empty_like(indptr)
    # out_indptr[0] = 0
    # for r in prange(nrows):
    #     r_data = List()
    #     r_ind = List()
    #     for p in range(indptr[r], indptr[r + 1]):
    #         c = ind[p]
    #         v = data[p] * arr[r, c]
    #         if v != 0:
    #             r_data.append(v)
    #             r_ind.append(c)
    #     out_data[r] = r_data
    #     out_ind[r] = r_ind
    #     out_indptr[r + 1] = len(r_data)

    # for i in range(1, nrows):
    #     out_indptr[i + 1] += out_indptr[i]

    # od = np.empty_like(data)
    # oi = np.empty_like(ind)
    # for r in prange(nrows):
    #     r_data = out_data[r]
    #     r_ind = out_ind[r]
    #     s = out_indptr[r]
    #     for i in range(len(r_data)):
    #         od[s + i] = r_data[i]
    #         oi[s + i] = r_ind[i]

    # return (od, oi, out_indptr, ncols)


def spm_add_sparse(spm1: tuple[np.ndarray, np.ndarray, np.ndarray, int],
                   spm2: tuple[np.ndarray, np.ndarray, np.ndarray, int]):
    data1 = spm1[0]
    ind1 = spm1[1]
    indptr1 = spm1[2]
    ncols = spm1[3]
    nrows = len(indptr1) - 1
    data2 = spm2[0]
    ind2 = spm2[1]
    indptr2 = spm2[2]
    assert nrows == len(indptr2) - 1 and ncols == spm2[3]

    out_data = np.empty(len(data1) + len(data2), dtype=np.float64)
    out_ind = np.empty(len(data1) + len(data2), dtype=np.int32)
    out_indptr = np.empty_like(indptr1)

    # Assume that cols are monotonic
    out_p = 0
    for r in range(nrows):
        out_indptr[r] = out_p
        p1 = indptr1[r]
        p2 = indptr2[r]
        up1 = indptr1[r + 1]
        up2 = indptr2[r + 1]
        while p1 < up1 or p2 < up2:
            c1 = ind1[p1] if p1 < up1 else ncols
            c2 = ind2[p2] if p2 < up2 else ncols
            if c1 < c2:
                v = data1[p1]
                out_data[out_p] = v
                out_ind[out_p] = c1
                out_p += 1
                p1 += 1
            elif c1 > c2:
                v = data2[p2]
                out_data[out_p] = v
                out_ind[out_p] = c2
                out_p += 1
                p2 += 1
            else:
                v = data1[p1] + data2[p2]
                out_data[out_p] = v
                out_ind[out_p] = c1
                out_p += 1
                p1 += 1
                p2 += 1
    out_indptr[nrows] = out_p

    return (out_data[:out_p], out_ind[:out_p], out_indptr, ncols)


def spmm_dense_ds(arr: np.ndarray, spm: tuple[np.ndarray, np.ndarray, np.ndarray, int]):
    data = spm[0]
    ind = spm[1]
    indptr = spm[2]
    ncols = spm[3]
    nrows = len(indptr) - 1
    assert arr.ndim == 2 and nrows == arr.shape[1]

    ni = arr.shape[0]
    nk = arr.shape[1]
    nj = ncols

    out = np.zeros((ni, nj), dtype=np.float64)

    for i in range(ni):
        for k in range(nk):
            v = arr[i, k]
            for p in range(indptr[k], indptr[k + 1]):
                j = ind[p]
                out[i, j] += v * data[p]

    return out


def spm_diags(arr: np.ndarray):
    n = arr.size
    out_data = arr.astype(np.float64)
    out_ind = np.arange(n, dtype=np.int32)
    out_indptr = np.arange(n + 1, dtype=np.int32)
    return out_data, out_ind, out_indptr, n


def spmm(spm1: tuple[np.ndarray, np.ndarray, np.ndarray, int],
         spm2: tuple[np.ndarray, np.ndarray, np.ndarray, int]):
    data1 = spm1[0]
    ind1 = spm1[1]
    indptr1 = spm1[2]
    ncols1 = spm1[3]
    nrows1 = len(indptr1) - 1
    data2 = spm2[0]
    ind2 = spm2[1]
    indptr2 = spm2[2]
    ncols2 = spm2[3]
    nrows2 = len(indptr2) - 1
    assert ncols1 == nrows2

    ni = nrows1
    nj = ncols2

    out_data = []
    out_ind = []
    out_indptr = [0]

    for i in range(ni):
        t = np.zeros(nj, np.float64)
        for pk in range(indptr1[i], indptr1[i + 1]):
            k = ind1[pk]
            v = data1[pk]
            for pj in range(indptr2[k], indptr2[k + 1]):
                j = ind2[pj]
                t[j] += v * data2[pj]
        for j in range(nj):
            if t[j] != 0:
                out_data.append(t[j])
                out_ind.append(j)
        out_indptr.append(len(out_data))

    return (np.array(out_data, dtype=np.float64),
            np.array(out_ind, dtype=np.int32),
            np.array(out_indptr, dtype=np.int32),
            ncols2)


def spmv(spm: tuple[np.ndarray, np.ndarray, np.ndarray, int], arr: np.ndarray):
    data = spm[0]
    ind = spm[1]
    indptr = spm[2]
    ncols = spm[3]
    nrows = len(indptr) - 1
    assert arr.ndim == 1 and ncols == arr.shape[0]

    out = np.empty(nrows, dtype=np.float64)
    for r in range(nrows):
        t = 0
        for p in range(indptr[r], indptr[r + 1]):
            c = ind[p]
            t += data[p] * arr[c]
        out[r] = t

    return out
