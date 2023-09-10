#ifndef SPARSEMAT_HPP
#define SPARSEMAT_HPP

#include <cstdlib>
#include <iostream>
#include <iterator>
#include <map>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

#include "operation.hpp"
#include "shared/NpArray.hpp"
#include "shared/inline.hpp"

namespace py = pybind11;

template <typename T> struct _compressed_row {
  std::vector<int> cols;
  std::vector<T> vals;
};

namespace pydd {
// Class for sparse data structure
template <class T> class Sparse_matrix {
public:
  int nrows, ncols;
  std::shared_ptr<std::vector<struct _compressed_row<T>>> csr;

  Sparse_matrix() {}

  Sparse_matrix(int nr, int nc) {
    nrows = nr;
    ncols = nc;
    struct _compressed_row<T> tmp;
    csr.reset(new std::vector<struct _compressed_row<T>>(nr, tmp));
  }

  Sparse_matrix(const py::array_t<T> &arr_values,
                const py::array_t<int> &arr_columns,
                const py::array_t<int> &arr_indexes, int ncolumns) {
    auto data_v = arr_values.data();
    auto data_c = arr_columns.data();
    auto data_i = arr_indexes.data();
    int total_v = arr_values.size();
    int total_c = arr_columns.size();
    int total_i = arr_indexes.size();
    if (total_v != total_c || total_v != ((int *)data_i)[total_i - 1]) {
      std::cerr << "[Error] Incorrect CSR format.\n";
      exit(-1);
    }
    nrows = total_i - 1;
    ncols = ncolumns;

    struct _compressed_row<T> tmp;
    csr.reset(new std::vector<struct _compressed_row<T>>(nrows, tmp));
    for (int r = 0; r < nrows; r++) {
      for (int p = ((int *)data_i)[r]; p < ((int *)data_i)[r + 1]; p++) {
        T v = ((T *)data_v)[p];
        int c = ((int *)data_c)[p];
        (*csr)[r].vals.push_back(v);
        (*csr)[r].cols.push_back(c);
      }
    }
  }

  Sparse_matrix(const py::array_t<T> &arr) {
    if (arr.ndim() != 2) {
      std::cerr << "[Error] Dimensionality must be 2 for dense-to-sparse array "
                   "conversion.\n";
      exit(-1);
    }
    auto data = arr.data();
    nrows = arr.shape(0);
    ncols = arr.shape(1);

    struct _compressed_row<T> tmp;
    csr.reset(new std::vector<struct _compressed_row<T>>(nrows, tmp));
    for (int r = 0; r < nrows; r++) {
      for (int c = 0; c < ncols; c++) {
        T v = ((T *)data)[r * ncols + c];
        if (v != 0) {
          (*csr)[r].vals.push_back(v);
          (*csr)[r].cols.push_back(c);
        }
      }
    }
  }
};

template <typename T>
PYDD_INLINE void spm_set_item_unsafe(Sparse_matrix<T> &spm, T v, int r, int c);

// Element-wise operators
template <typename T1, typename T2>
Sparse_matrix<double> spm_binary_scalar(const Sparse_matrix<T1> &spm1, T2 arg,
                                        double (*operation)(T1, T2)) {
  // Question: need or_cond (e.g., add/sub)?
  Sparse_matrix<double> spm0(spm1.nrows, spm1.ncols);
  for (int r = 0; r < spm1.nrows; r++) {
    for (size_t p = 0; p < (*spm1.csr)[r].cols.size(); p++) {
      int c = (*spm1.csr)[r].cols[p];
      T1 v1 = (*spm1.csr)[r].vals[p];
      double v0 = operation(v1, arg);
      if (v0 != 0)
        spm_set_item_unsafe(spm0, v0, r, c);
    }
  }
  return spm0;
}

template <typename T1, typename T2>
Sparse_matrix<double> spm_binary(const Sparse_matrix<T1> &spm1,
                                 const py::array_t<T2> &arr,
                                 double (*operation)(T1, T2)) {
  // Question: need or_cond (e.g., add/sub)?
  if (spm1.nrows != arr.shape(0) || spm1.ncols != arr.shape(1) ||
      arr.ndim() != 2) {
    std::cerr
        << "[Error] Element-wise binary operation assumes same shape arrays.\n";
    exit(-1);
  }
  auto data = arr.data();
  int ncols = arr.shape(1);

  Sparse_matrix<double> spm0(spm1.nrows, spm1.ncols);
  for (int r = 0; r < spm1.nrows; r++) {
    for (size_t p = 0; p < (*spm1.csr)[r].cols.size(); p++) {
      int c = (*spm1.csr)[r].cols[p];
      T1 v1 = (*spm1.csr)[r].vals[p];
      T2 v2 = ((T2 *)data)[r * ncols + c];
      double v0 = operation(v1, v2);
      if (v0 != 0)
        spm_set_item_unsafe(spm0, v0, r, c);
    }
  }
  return spm0;
}

template <typename T1, typename T2>
Sparse_matrix<double> spm_binary(const Sparse_matrix<T1> &spm1,
                                 const Sparse_matrix<T2> &spm2,
                                 double (*operation)(T1, T2), bool or_cond) {
  if (spm1.nrows != spm2.nrows || spm1.ncols != spm2.ncols) {
    std::cerr
        << "[Error] Element-wise binary operation assumes same shape arrays.\n";
    exit(-1);
  }

  Sparse_matrix<double> spm0(spm1.nrows, spm1.ncols);
  for (int r = 0; r < spm1.nrows; r++) {
    int p1 = 0;
    int p2 = 0;
    // Assume that cols are monotonic (TODO: sort cols with vals)
    if (or_cond) {
      int up1 = (*spm1.csr)[r].cols.size();
      int up2 = (*spm2.csr)[r].cols.size();
      while (p1 < up1 || p2 < up2) {
        int c1 = (p1 < up1) ? (*spm1.csr)[r].cols[p1] : spm1.ncols;
        int c2 = (p2 < up2) ? (*spm2.csr)[r].cols[p2] : spm2.ncols;
        if (c1 < c2) {
          T1 v1 = (*spm1.csr)[r].vals[p1];
          if (v1 != 0)
            spm_set_item_unsafe(spm0, v1, r, c1);
          p1++;
        } else if (c2 < c1) {
          T2 v2 = (*spm2.csr)[r].vals[p2];
          if (v2 != 0)
            spm_set_item_unsafe(spm0, v2, r, c2);
          p2++;
        } else {
          T1 v1 = (*spm1.csr)[r].vals[p1];
          T2 v2 = (*spm2.csr)[r].vals[p2];
          double v0 = operation(v1, v2);
          if (v0 != 0)
            spm_set_item_unsafe(spm0, v0, r, c1);
          p1++;
          p2++;
        }
      }
    } else {
      int up1 = (*spm1.csr)[r].cols.size();
      int up2 = (*spm2.csr)[r].cols.size();
      while (p1 < up1 && p2 < up2) {
        int c1 = (*spm1.csr)[r].cols[p1];
        int c2 = (*spm2.csr)[r].cols[p2];
        if (c1 < c2) {
          p1++;
        } else if (c2 < c1) {
          p2++;
        } else {
          T1 v1 = (*spm1.csr)[r].vals[p1];
          T2 v2 = (*spm2.csr)[r].vals[p2];
          double v0 = operation(v1, v2);
          if (v0 != 0)
            spm_set_item_unsafe(spm0, v0, r, c1);
          p1++;
          p2++;
        }
      }
    }
  }
  return spm0;
}

// Reducers
template <typename T>
T spm_reduce(const Sparse_matrix<T> &spm, T (*operation)(T, T), T init_val) {
  T res = init_val;
  for (int r = 0; r < spm.nrows; r++) {
    for (int p = 0; p < (*spm.csr)[r].cols.size(); p++) {
      T v = (*spm.csr)[r].vals[p];
      res = operation(v, res); // Must be this order of args due to nnz
    }
  }
  return res;
}

template <typename T>
std::vector<int> spm_reduce_index(const Sparse_matrix<T> &spm,
                                  bool (*condition)(T, T), T init_val,
                                  const std::vector<int> &init_idx) {
  T res = init_val;
  int res_r = init_idx[0];
  int res_c = init_idx[1];
  for (int r = 0; r < spm.nrows; r++) {
    for (int p = 0; p < (*spm.csr)[r].cols.size(); p++) {
      int c = (*spm.csr)[r].cols[p];
      T v = (*spm.csr)[r].vals[p];
      if (condition(v, res)) {
        res = v;
        res_r = r;
        res_c = c;
      }
    }
  }
  return {res_r, res_c};
}

template <typename T>
bool spm_all_meet_condition(const Sparse_matrix<T> &spm,
                            bool (*condition)(T, T), T arg) {
  for (int r = 0; r < spm.nrows; r++) {
    for (int p = 0; p < (*spm.csr)[r].cols.size(); p++) {
      T v = (*spm.csr)[r].vals[p];
      if (!condition(v, arg))
        return false;
    }
  }
  return true;
}

// Intrepydd functions: Create and output sparse matrix
/* TODO:
template<typename T=double>
Sparse_matrix<T> empty_spm(int nr, int nc, T e=T()) {
  return Sparse_matrix<T>(nr, nc);
}
*/
PYDD_INLINE Sparse_matrix<double> empty_spm(int nr, int nc) {
  return Sparse_matrix<double>(nr, nc);
}

/* TODO:
template<typename T>
Sparse_matrix<T> csr_to_spm(py::array_t<T> &arr_values, py::array_t<int>
&arr_columns, py::array_t<int> &arr_indexes, int ncolumns) { return
Sparse_matrix<T>(arr_values, arr_columns, arr_indexes, ncolumns);
}
*/
PYDD_INLINE Sparse_matrix<double>
csr_to_spm(const py::array_t<double> &arr_values,
           const py::array_t<int> &arr_columns,
           const py::array_t<int> &arr_indexes, int ncolumns) {
  return Sparse_matrix<double>(arr_values, arr_columns, arr_indexes, ncolumns);
}

/* TODO:
template<typename T>
...
*/
PYDD_INLINE Sparse_matrix<double> arr_to_spm(const py::array_t<double> &arr) {
  return Sparse_matrix<double>(arr);
}

/* TODO:
template<typename T>
...
*/
int spm_to_csr(const Sparse_matrix<double> &spm,
               py::array_t<double> &arr_values,
               py::array_t<int> &arr_columns,
               py::array_t<int> &arr_indexes) {
  arr_indexes.resize({spm.nrows + 1}, false);
  auto data_i = arr_indexes.data();
  int total = 0;
  for (int r = 0; r < spm.nrows; r++) {
    ((int *)data_i)[r] = total;
    total += (*spm.csr)[r].cols.size();
  }
  ((int *)data_i)[spm.nrows] = total;

  arr_values.resize({total}, false);
  arr_columns.resize({total}, false);
  auto data_v = arr_values.data();
  auto data_c = arr_columns.data();
  for (int r = 0; r < spm.nrows; r++) {
    int offset = ((int *)data_i)[r];
    for (size_t p = 0; p < (*spm.csr)[r].cols.size(); p++) {
      double v = (*spm.csr)[r].vals[p];
      int c = (*spm.csr)[r].cols[p];
      ((double *)data_v)[offset + p] = v;
      ((int *)data_c)[offset + p] = c;
    }
  }
  return spm.ncols;
}

// Construct a sparse matrix from an array, using the array as diagonal elements
// https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.diags.html
template <typename T>
Sparse_matrix<double> sparse_diags(const py::array_t<T> &a) {
  const int a_size = a.size();
  const auto data_a = a.data();
  Sparse_matrix<double> result(a_size, a_size);

  for (int i = 0; i < a_size; ++i) {
    double array_val = ((T *)data_a)[i];
    spm_set_item_unsafe(result, array_val, i, i);
  }

  return result;
}

template <typename T>
PYDD_INLINE Sparse_matrix<double> spm_diags(const py::array_t<T> &a) {
  return sparse_diags(a);
}

// Intrepydd functions: Set element
template <typename T>
PYDD_INLINE void spm_set_item_unsafe(Sparse_matrix<T> &spm, T v, int r,
                                     int c) {
  (*spm.csr)[r].cols.push_back(c);
  (*spm.csr)[r].vals.push_back(v);
}

template <typename T>
void spm_set_item(Sparse_matrix<T> &spm, T v, int r, int c) {
  if (r >= spm.nrows || c >= spm.ncols) {
    std::cerr << "[Error] Out of bounds for CSR format.\n";
    exit(-1);
  }
  std::vector<int>::iterator it_c = (*spm.csr)[r].cols.begin();
  std::vector<double>::iterator it_v =
      (*spm.csr)[r].vals.begin(); // To be template T
  for (; it_c != (*spm.csr)[r].cols.end(); it_c++, it_v++) {
    if (c == *it_c) {
      // Overwrite existing item [r,c] by v, which can be "0".
      *it_v = v;
      return;
    } else if (c < *it_c && v != 0) { // new item [r,c] must be non-zero
      (*spm.csr)[r].cols.insert(it_c, c);
      (*spm.csr)[r].vals.insert(it_v, v);
      return;
    }
  }
  if (v != 0) // new item [r,c] must be non-zero
    spm_set_item_unsafe(spm, v, r, c);
}

// Intrepydd functions: Observation
template <typename T> int shape(const Sparse_matrix<T> &spm, int i) {
  if (i < 0 || i > 1) {
    std::cerr << "[Error] Out of bounds for dimensionality.\n";
    exit(-1);
  }
  return i == 0 ? spm.nrows : spm.ncols;
}

template <typename T> std::vector<int64_t> shape(const Sparse_matrix<T> &spm) {
  return {spm.nrows, spm.ncols};
}

template <typename T>
PYDD_INLINE int getnnz(const Sparse_matrix<T> &spm, int r) {
  return (*spm.csr)[r].cols.size();
}

template <typename T>
PYDD_INLINE int getcol(const Sparse_matrix<T> &spm, int r, int p) {
  return (*spm.csr)[r].cols[p];
}

template <typename T>
PYDD_INLINE T getval(const Sparse_matrix<T> &spm, int r, int p) {
  return (*spm.csr)[r].vals[p];
}

template <typename T> T spm_first_nonzero(const Sparse_matrix<T> &spm) {
  for (int r = 0; r < spm.nrows; r++) {
    for (int p = 0; p < (*spm.csr)[r].cols.size(); p++) {
      T v = (*spm.csr)[r].vals[p];
      if (v != 0)
        return v;
    }
  }
  return 0;
}

template <typename T>
std::vector<int> spm_first_nonzero_index(const Sparse_matrix<T> &spm) {
  for (int r = 0; r < spm.nrows; r++) {
    for (int p = 0; p < (*spm.csr)[r].cols.size(); p++) {
      T v = (*spm.csr)[r].vals[p];
      if (v != 0) {
        int c = (*spm.csr)[r].cols[p];
        return std::vector<int>{r, c};
      }
    }
  }
  return {};
}

// Intrepydd functions: Element-wise operations
template <typename T1, typename T2>
Sparse_matrix<double> spm_add(const Sparse_matrix<T1> &spm1, T2 arg) {
  return spm_binary_scalar(spm1, arg, ope_add);
}
template <typename T1, typename T2>
Sparse_matrix<double> spm_mul(const Sparse_matrix<T1> &spm1, T2 arg) {
  return spm_binary_scalar(spm1, arg, ope_mul);
}

template <typename T1, typename T2>
Sparse_matrix<double> spm_add(const Sparse_matrix<T1> &spm1,
                              py::array_t<T2> arr) {
  return spm_binary(spm1, arr, ope_add);
}
template <typename T1, typename T2>
Sparse_matrix<double> spm_mul(const Sparse_matrix<T1> &spm1,
                              py::array_t<T2> arr) {

  return spm_binary(spm1, arr, ope_mul);
}

template <typename T1, typename T2>
Sparse_matrix<double> spm_add(const Sparse_matrix<T1> &spm1,
                              const Sparse_matrix<T2> &spm2) {
  return spm_binary(spm1, spm2, ope_add, true);
}
template <typename T1, typename T2>
Sparse_matrix<double> spm_mul(const Sparse_matrix<T1> &spm1,
                              const Sparse_matrix<T2> &spm2) {
  return spm_binary(spm1, spm2, ope_mul, false);
}

// Intrepydd functions: Reduction
//  - Question: Should rename sum by spm_sum and etc (to specify sparsemat.hpp)?
template <typename T> int nnz(const Sparse_matrix<T> &spm) {
  return spm_reduce(spm, ope_nnz, (T)0);
}
template <typename T> T sum(const Sparse_matrix<T> &spm) {
  return spm_reduce(spm, ope_add, (T)0);
}
template <typename T> T prod(const Sparse_matrix<T> &spm) {
  return spm_reduce(spm, ope_mul, (T)1);
}
template <typename T> T min(const Sparse_matrix<T> &spm) {
  return spm_reduce(spm, ope_minimum, spm_first_nonzero(spm));
}
template <typename T> T max(const Sparse_matrix<T> &spm) {
  return spm_reduce(spm, ope_maximum, spm_first_nonzero(spm));
}

template <typename T> std::vector<int> *argmin(const Sparse_matrix<T> &spm) {
  return spm_reduce_index(spm, ope_lt, spm_first_nonzero(spm),
                          spm_first_nonzero_index(spm));
}
template <typename T> std::vector<int> *argmax(const Sparse_matrix<T> &spm) {
  return spm_reduce_index(spm, ope_gt, spm_first_nonzero(spm),
                          spm_first_nonzero_index(spm));
}

template <typename T> bool any(const Sparse_matrix<T> &spm) {
  // true if any element is nonzero (false if all element is zero)
  return !spm_all_meet_condition(spm, ope_eq, (T)0);
}
template <typename T>
bool all(const Sparse_matrix<T> &spm,
         bool elems_only) { // Question: elems_only for all reductions?
  // true if all elements are nonzero
  if (!elems_only && nnz(spm) < spm.nrows * spm.ncols)
    return false;
  return spm_all_meet_condition(spm, ope_neq, (T)0);
}
template <typename T> bool all(const Sparse_matrix<T> &spm) {
  return all(spm, true);
}
template <typename T> bool allclose(const Sparse_matrix<T> &spm, T eps) {
  // true if all elements are at most eps in absolute value
  return spm_all_meet_condition(spm, ope_close, std::abs(eps));
}

// Intrepydd functions: Sparse Matrix Vector Multiply
template <typename T1, typename T2>
py::array_t<double> spmv(const Sparse_matrix<T1> &spm,
                         const py::array_t<T2> &arr) {
  if (arr.ndim() != 1 || spm.ncols != arr.shape(0)) {
    std::cerr << "[Error] sparse matrix-vector multiply assumes compatible "
                 "1-D/2-D arrays.\n";
    exit(-1);
  }
  auto data = arr.data();

  py::array_t<double> arr0 = empty(spm.nrows, double());
  auto data0 = arr0.mutable_data();
  for (int i = 0; i < spm.nrows; i++) {
    double tmp = 0;
    for (size_t p = 0; p < (*spm.csr)[i].cols.size(); p++) {
      int j = (*spm.csr)[i].cols[p];
      T1 v = (*spm.csr)[i].vals[p];
      tmp += v * ((T2 *)data)[j];
    }
    ((double *)data0)[i] = tmp;
  }
  return arr0;
}

template <typename T1, typename T2>
int _spmv(const Sparse_matrix<T1> &spm, const py::array_t<T2> &arr,
          py::array_t<double> &out) {
  if (arr.ndim() != 1 || spm.ncols != arr.shape(0)) {
    std::cerr << "[Error] sparse matrix-vector multiply assumes compatible "
                 "1-D/2-D arrays.\n";
    exit(-1);
  }
  auto data = arr.data();
  auto data0 = out.mutable_data();
  for (int i = 0; i < spm.nrows; i++) {
    double tmp = 0;
    for (int p = 0; p < (*spm.csr)[i].cols.size(); p++) {
      int j = (*spm.csr)[i].cols[p];
      T1 v = (*spm.csr)[i].vals[p];
      tmp += v * ((T2 *)data)[j];
    }
    ((double *)data0)[i] = tmp;
  }
  return 0;
}

// Intrepydd functions: Sparse Matrix Matrix Multiply
template <typename T1, typename T2>
Sparse_matrix<double> spmm(const Sparse_matrix<T1> &spm1,
                           const Sparse_matrix<T2> &spm2) {
  if (spm1.ncols != spm2.nrows) {
    std::cerr << "[Error] sparse matrix-matrix multiply assumes compatible 2-D "
                 "arrays.\n";
    exit(-1);
  }
  int ni = spm1.nrows;
  int nj = spm2.ncols;

  Sparse_matrix<double> spm0 = empty_spm(ni, nj);
  for (int i = 0; i < ni; i++) {
    std::vector<double> vect0(nj, 0);
    for (size_t p_k = 0; p_k < (*spm1.csr)[i].cols.size(); p_k++) {
      int k = (*spm1.csr)[i].cols[p_k];
      T1 v1 = (*spm1.csr)[i].vals[p_k];
      for (size_t p_j = 0; p_j < (*spm2.csr)[k].cols.size(); p_j++) {
        int j = (*spm2.csr)[k].cols[p_j];
        T2 v2 = (*spm2.csr)[k].vals[p_j];
        vect0[j] += v1 * v2;
      }
    }
    for (int j = 0; j < nj; j++) {
      if (vect0[j] != 0)
        spm_set_item_unsafe(spm0, vect0[j], i, j);
    }
  }
  return spm0;
}

template <typename T1, typename T2>
py::array_t<double> spmm_dense(const Sparse_matrix<T1> &spm1,
                               const py::array_t<T2> &arr2) {
  if (spm1.ncols != arr2.shape(0) || arr2.ndim() != 2) {
    std::cerr << "[Error] sparse matrix-matrix multiply assumes compatible 2-D "
                 "arrays.\n";
    exit(-1);
  }
  auto data2 = arr2.data();
  int ni = spm1.nrows;
  int nj = arr2.shape(1);

  py::array_t<double> arr0 = empty({ni, nj}, double());
  auto data0 = arr0.mutable_data();
  for (int i = 0; i < ni; i++) {
    for (int j = 0; j < nj; j++) {
      ((double *)data0)[i * nj + j] = 0;
    }
    for (int p_k = 0; p_k < (*spm1.csr)[i].cols.size(); p_k++) {
      int k = (*spm1.csr)[i].cols[p_k];
      T1 v1 = (*spm1.csr)[i].vals[p_k];
      for (int j = 0; j < nj; j++) {
        T2 v2 = ((T2 *)data2)[k * nj + j];
        ((double *)data0)[i * nj + j] += v1 * v2;
      }
    }
  }
  return arr0;
}

template <typename T1, typename T2>
void spmm_dense(const Sparse_matrix<T1> &spm1, const py::array_t<T2> &arr2,
                py::array_t<double> &out) {
  if (spm1.ncols != arr2.shape(0) || arr2.ndim() != 2) {
    std::cerr << "[Error] sparse matrix-matrix multiply assumes compatible 2-D "
                 "arrays.\n";
    exit(-1);
  }
  auto data2 = arr2.data();
  int ni = spm1.nrows;
  int nj = arr2.shape(1);

  auto data0 = out.mutable_data();
  for (int i = 0; i < ni; i++) {
    for (int j = 0; j < nj; j++) {
      ((double *)data0)[i * nj + j] = 0;
    }
    for (int p_k = 0; p_k < (*spm1.csr)[i].cols.size(); p_k++) {
      int k = (*spm1.csr)[i].cols[p_k];
      T1 v1 = (*spm1.csr)[i].vals[p_k];
      for (int j = 0; j < nj; j++) {
        T2 v2 = ((T2 *)data2)[k * nj + j];
        ((double *)data0)[i * nj + j] += v1 * v2;
      }
    }
  }
}

template <typename T1, typename T2>
py::array_t<double> spmm_dense(const py::array_t<T1> &arr1,
                               const Sparse_matrix<T2> &spm2) {
  if (arr1.shape(1) != spm2.nrows || arr1.ndim() != 2) {
    std::cerr << "[Error] sparse matrix-matrix multiply assumes compatible 2-D "
                 "arrays.\n";
    exit(-1);
  }
  auto data1 = arr1.data();
  int ni = arr1.shape(0);
  int nk = arr1.shape(1);
  int nj = spm2.ncols;

  py::array_t<double> arr0 = empty({ni, nj}, double());
  auto data0 = arr0.mutable_data();
  for (int i = 0; i < ni; i++) {
    for (int j = 0; j < nj; j++) {
      ((double *)data0)[i * nj + j] = 0;
    }
    for (int k = 0; k < nk; k++) {
      T1 v1 = ((T1 *)data1)[i * nk + k];
      for (size_t p_j = 0; p_j < (*spm2.csr)[k].cols.size(); p_j++) {
        int j = (*spm2.csr)[k].cols[p_j];
        T2 v2 = (*spm2.csr)[k].vals[p_j];
        ((double *)data0)[i * nj + j] += v1 * v2;
      }
    }
  }
  return arr0;
}

template <typename T1, typename T2>
void spmm_dense(const py::array_t<T1> &arr1, const Sparse_matrix<T2> &spm2,
                py::array_t<double> &out) {
  if (arr1.shape(1) != spm2.nrows || arr1.ndim() != 2) {
    std::cerr << "[Error] sparse matrix-matrix multiply assumes compatible 2-D "
                 "arrays.\n";
    exit(-1);
  }
  auto data1 = arr1.data();
  int ni = arr1.shape(0);
  int nk = arr1.shape(1);
  int nj = spm2.ncols;

  auto data0 = out.mutable_data();
  for (int i = 0; i < ni; i++) {
    for (int j = 0; j < nj; j++) {
      ((double *)data0)[i * nj + j] = 0;
    }
    for (int k = 0; k < nk; k++) {
      T1 v1 = ((T1 *)data1)[i * nk + k];
      for (int p_j = 0; p_j < (*spm2.csr)[k].cols.size(); p_j++) {
        int j = (*spm2.csr)[k].cols[p_j];
        T2 v2 = (*spm2.csr)[k].vals[p_j];
        ((double *)data0)[i * nj + j] += v1 * v2;
      }
    }
  }
}

template <typename T1, typename T2>
py::array_t<double> spmm_dense(const Sparse_matrix<T1> &spm1,
                               const Sparse_matrix<T2> &spm2) {
  if (spm1.ncols != spm2.nrows) {
    std::cerr << "[Error] sparse matrix-matrix multiply assumes compatible 2-D "
                 "arrays.\n";
    exit(-1);
  }
  int ni = spm1.nrows;
  int nj = spm2.ncols;

  py::array_t<double> arr0 = empty({ni, nj}, double());
  auto data0 = arr0.mutable_data();
  for (int i = 0; i < ni; i++) {
    for (int j = 0; j < nj; j++) {
      ((double *)data0)[i * nj + j] = 0;
    }
    for (int p_k = 0; p_k < (*spm1.csr)[i].cols.size(); p_k++) {
      int k = (*spm1.csr)[i].cols[p_k];
      T1 v1 = (*spm1.csr)[i].vals[p_k];
      for (int p_j = 0; p_j < (*spm2.csr)[k].cols.size(); p_j++) {
        int j = (*spm2.csr)[k].cols[p_j];
        T2 v2 = (*spm2.csr)[k].vals[p_j];
        ((double *)data0)[i * nj + j] += v1 * v2;
      }
    }
  }
  return arr0;
}

template <typename T>
Sparse_matrix<T> spm_mask(const Sparse_matrix<T> &spm,
                          const py::array_t<bool> &mask) {
  if (spm.nrows != mask.shape(0)) {
    std::cerr << "[Error] spm_mask assumes compatible 2-D array and mask"
              << std::endl;
  }

  const bool *mask_data = mask.data();

  // compute number of rows in new sparse array
  int nrows = 0;
  for (int i = 0; i < mask.shape(0); i++)
    if (mask_data[i])
      nrows++;

  // the number of columns remains the same when the mask is applied
  Sparse_matrix<T> ret(nrows, spm.ncols);

  // iterate over the mask and copy of the row to the new sparse matrix if the
  // mask is true.
  for (int i = 0, r = 0; i < mask.shape(0); i++) {
    if (((bool *)mask_data)[i]) {

      // copy over all the columns of the row
      for (int nc = 0; nc < (*spm.csr)[i].vals.size(); nc++) {
        (*ret.csr)[r].vals.push_back((*spm.csr)[i].vals[nc]);
        (*ret.csr)[r].cols.push_back((*spm.csr)[i].cols[nc]);
      }
      r++;
    }
  }

  return ret;
}
} // namespace pydd

#endif // SPARSEMAT_HPP
