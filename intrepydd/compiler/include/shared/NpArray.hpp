#ifndef NPARRAY_HPP
#define NPARRAY_HPP

#include <cassert>
#include <cstdlib>
#include <ctime>
#include <pybind11/numpy.h>

#include "inline.hpp"
#include "rt.hpp"

namespace py = pybind11;

namespace pydd {

template <typename T> int64_t len(const py::array_t<T> &arr) {
  return arr.shape(0);
}

template <typename T> int64_t shape(const py::array_t<T> &arr, int i) {
  return arr.shape(i);
}

template <typename T> std::vector<int64_t> shape(const py::array_t<T> &arr) {
  std::vector<int64_t> s;
  auto sp = arr.shape();
  for (int i = 0; i < arr.ndim(); ++i) {
    s.push_back(sp[i]);
  }
  return s;
}

template <typename T> int64_t ndim(const py::array_t<T> &arr) {
  return arr.ndim();
}

template <typename T, typename... Ix>
PYDD_INLINE const T &getitem(const py::array_t<T> &arr, Ix... index) {
  return arr.at(index...);
}

template <typename T1, typename T2, typename... Ix>
PYDD_INLINE void setitem(py::array_t<T1> &arr, T2 v, Ix... index) {
  arr.mutable_at(index...) = v;
}

template <typename T>
PYDD_INLINE const T &getitem_1d(const py::array_t<T> &arr, int64_t i) {
  return arr.data()[i];
}

template <typename T1, typename T2>
PYDD_INLINE void setitem_1d(py::array_t<T1> &arr, T2 v, int64_t i) {
  arr.mutable_data()[i] = v;
}

template <typename T>
PYDD_INLINE const T &getitem_1d(const T *data, int64_t i) {
  return data[i];
}

template <typename T1, typename T2>
PYDD_INLINE void setitem_1d(T1 *data, T2 v, int64_t i) {
  data[i] = v;
}

template <typename T>
PYDD_INLINE const T &getitem_2d(const py::array_t<T> &arr, int64_t i,
                                int64_t j) {
  int64_t s1 = arr.shape(1);
  return arr.data()[i * s1 + j];
}

template <typename T1, typename T2>
PYDD_INLINE void setitem_2d(py::array_t<T1> &arr, T2 v, int64_t i, int64_t j) {
  int64_t s1 = arr.shape(1);
  arr.mutable_data()[i * s1 + j] = v;
}

template <typename T>
PYDD_INLINE const T &getitem_2d(const T *data, int64_t s1, int64_t i,
                                int64_t j) {
  return data[i * s1 + j];
}

template <typename T1, typename T2>
PYDD_INLINE void setitem_2d(T1 *data, int64_t s1, T2 v, int64_t i, int64_t j) {
  data[i * s1 + j] = v;
}

template <typename T>
PYDD_INLINE const T &getitem_3d(const py::array_t<T> &arr, int64_t i, int64_t j,
                                int64_t k) {
  int64_t s1 = arr.shape(1);
  int64_t s2 = arr.shape(2);
  return arr.data()[i * s1 * s2 + j * s2 + k];
}

template <typename T1, typename T2>
PYDD_INLINE void setitem_3d(py::array_t<T1> &arr, T2 v, int64_t i, int64_t j,
                            int64_t k) {
  int64_t s1 = arr.shape(1);
  int64_t s2 = arr.shape(2);
  arr.mutable_data()[i * s1 * s2 + j * s2 + k] = v;
}

template <typename T>
PYDD_INLINE const T &getitem_3d(const T *data, int64_t s1, int64_t s2,
                                int64_t i, int64_t j, int64_t k) {
  return data[i * s1 * s2 + j * s2 + k];
}

template <typename T1, typename T2>
PYDD_INLINE void setitem_3d(T1 *data, int64_t s1, int64_t s2, T2 v, int64_t i,
                            int64_t j, int64_t k) {
  data[i * s1 * s2 + j * s2 + k] = v;
}

template <typename T, typename U>
py::array_t<T> sum_rows(const py::array_t<T> &arr, const py::array_t<U> &rows) {
  int64_t s1 = arr.shape(1);
  py::array_t<T> ret = py::array_t<T>(s1);
  T *ret_data = ret.mutable_data();
  for (int i = 0; i < s1; ++i) {
    ret_data[i] = 0;
  }
  const T *arr_data = arr.data();
  assert(rows.ndim() == 1);
  const int *rows_data = rows.data();
  for (int r = 0; r < rows.shape(0); r++) {
    for (int i = 0; i < s1; ++i) {
      ret_data[i] += arr_data[i + s1 * rows_data[r]];
    }
  }
  return ret;
}

template <typename T> void plus_eq(py::array_t<T> &A, const py::array_t<T> &B) {
  assert(A.ndim() == B.ndim());
  assert(A.size() == B.size());
  T *A_data = A.mutable_data();
  const T *B_data = B.data();
  for (int i = 0; i < A.size(); ++i) {
    A_data[i] += B_data[i];
  }
}

template <typename T>
void minus_eq(py::array_t<T> &A, const py::array_t<T> &B) {
  assert(A.ndim() == B.ndim());
  assert(A.size() == B.size());
  T *A_data = A.mutable_data();
  const T *B_data = B.data();
  for (int i = 0; i < A.size(); ++i) {
    A_data[i] -= B_data[i];
  }
}

template <typename T>
py::array_t<T> get_row(const py::array_t<T> &arr, int64_t i) {
  const T *arr_data = arr.data();
  const T *start = arr_data + arr.shape(1) * i;
  /* Note: the last argument specifies parent object that owns the data */
  return py::array_t<T>(arr.shape(1), start, arr);
}

template <typename T>
py::array_t<T> get_col(const py::array_t<T> &arr, int64_t i) {
  const T *arr_data = arr.data();
  const T *start = arr_data + i;
  return py::array_t<T>({arr.shape(0)}, {arr.shape(1)}, start, arr);
}

template <typename T> void set_row(py::array_t<T> &arr, int64_t i, const T &v) {
  T *arr_data = arr.mutable_data();
  T *start = arr_data + arr.shape(1) * i;
  for (int j = 0; j < arr.shape(1); ++j) {
    start[j] = v;
  }
}

template <typename T>
void set_row(py::array_t<T> &arr, int64_t i, const py::array_t<T> &values) {
  T *arr_data = arr.mutable_data();
  T *arr_start = arr_data + arr.shape(1) * i;
  const T *value_start = values.data();
  for (int j = 0; j < arr.shape(1); ++j) {
    arr_start[j] = value_start[j];
  }
}

template <typename T> void set_col(py::array_t<T> &arr, int64_t i, const T &v) {
  T *arr_data = arr.mutable_data();
  int s1 = arr.shape(1);
  for (int j = 0; j < arr.shape(0); ++j) {
    arr_data[i + j * s1] = v;
  }
}

template <typename T> PYDD_INLINE void fill(py::array_t<T> &arr, const T &v) {
  auto d = arr.mutable_data();
  std::fill_n(d, arr.size(), v);
}

template <typename T1, typename T2>
PYDD_INLINE void fill(py::array_t<T1> &arr, const T2 &v) {
  auto d = arr.mutable_data();
  std::fill_n(d, arr.size(), static_cast<T1>(v));
}

template <typename T = double>
py::array_t<T> numpy_random_rand(std::initializer_list<int64_t> s, T e = T()) {
  py::array_t<T> arr(s);
  auto d = arr.mutable_data();
  // #pragma omp parallel for
  for (int i = 0; i < arr.size(); ++i) {
    double x = (double)std::rand() / RAND_MAX;
    *d = x;
    ++d;
  }
  return arr;
}

template <typename T = double>
py::array_t<T> arange(std::initializer_list<int64_t> s, T e = T()) {
  py::array_t<T> arr(s);
  auto d = arr.mutable_data();
  for (int i = 0; i < arr.size(); ++i) {
    *d = i;
    ++d;
  }
  return arr;
}

template <typename T = double>
py::array_t<T> empty(std::initializer_list<int64_t> s, T e = T()) {
  py::array_t<T> arr(s);
  return arr;
}

template <typename T = double>
py::array_t<T> empty(const std::vector<int> &s, T e = T()) {
  py::array_t<T> arr(s);
  return arr;
}

template <typename T = double>
py::array_t<T> empty(const std::vector<int64_t> &s, T e = T()) {
  py::array_t<T> arr(s);
  return arr;
}

template <typename T = double> py::array_t<T> empty(int64_t s, T e = T()) {
  py::array_t<T> arr(s);
  return arr;
}

template <typename T = double>
py::array_t<T> empty_2d(const std::vector<int64_t> &s, T e = T()) {
  py::array_t<T> arr(s);
  return arr;
}

template <typename T> py::array_t<T> empty_like(const py::array_t<T> &proto) {
  auto ndim = proto.ndim();
  if (ndim == 1)
    return py::array_t<T>(proto.shape(0));
  if (ndim == 2)
    return py::array_t<T>({proto.shape(0), proto.shape(1)});
  auto shape = proto.shape();
  return py::array_t<T>(std::vector<py::ssize_t>(shape, shape + ndim));
}

template <typename T = double> py::array_t<T> copy(const py::array_t<T> &arr) {
  py::array_t<T> arr1 = empty_like(arr);
  auto d = arr.data();
  auto d1 = arr1.mutable_data();
  std::copy(d, d + arr.size(), d1);
  return arr1;
}

template <typename T = double>
py::array_t<T> zeros(std::initializer_list<int64_t> s, T e = T()) {
  py::array_t<T> arr = empty(s, e);
  fill(arr, 0);
  return arr;
}

template <typename T = double>
py::array_t<T> zeros(const std::vector<int> &s, T e = T()) {
  py::array_t<T> arr = empty(s, e);
  fill(arr, 0);
  return arr;
}

template <typename T = double>
py::array_t<T> zeros_2d(const std::vector<int64_t> &s, T e = T()) {
  py::array_t<T> arr = empty_2d(s, e);
  fill(arr, 0);
  return arr;
}

template <typename T = double>
py::array_t<T> zeros(const std::vector<int64_t> &s, T e = T()) {
  py::array_t<T> arr = empty(s, e);
  fill(arr, 0);
  return arr;
}

template <typename T = double> py::array_t<T> zeros(int64_t s, T e = T()) {
  py::array_t<T> arr = empty(s, e);
  fill(arr, 0);
  return arr;
}

int rand() { return std::rand(); }

int rand(int min, int max) { return pydd::rand() % (max - min + 1) + min; }

} // namespace pydd

#endif // NPARRAY_HPP
