#ifndef REDUCTION_HPP
#define REDUCTION_HPP

#include <cstdlib>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "elemwise.hpp"
#include "operation.hpp"

namespace py = pybind11;

namespace pydd {

// Reducers
template <typename T>
T reduce(const py::array_t<T> &arr, T (*operation)(T, T), T init_val) {
  auto data = arr.data();
  int total = arr.size();

  T res = init_val;
  for (int i = 0; i < total; i++) {
    res = operation(((T *)data)[i], res);
  }
  return res;
}

template <typename T>
std::vector<int64_t> convert_to_muti_indexes(const py::array_t<T> &arr,
                                             int index) {
  int ndim = arr.ndim();
  int unit = arr.strides(ndim - 1);
  std::vector<int64_t> index_md(ndim, 0);

  int remained = index;
  for (int d = 0; d < ndim - 1; d++) {
    int step = arr.strides(d) / unit;
    int i = remained / step;
    index_md[d] = i;
    remained -= i * step;
  }
  index_md[ndim - 1] = remained;

  return index_md;
}

template <typename T>
std::vector<int64_t> reduce_index(const py::array_t<T> &arr,
                                  bool (*condition)(T, T), T init_val) {
  auto data = arr.data();
  int total = arr.size();

  int index = 0;
  T res = init_val;
  for (int i = 0; i < total; i++) {
    T val = ((T *)data)[i];
    if (condition(val, res)) {
      res = val;
      index = i;
    }
  }
  return convert_to_muti_indexes(arr, index);
}

template <typename T>
bool all_meet_condition(const py::array_t<T> &arr, bool (*condition)(T, T),
                        T arg) {
  auto data = arr.data();
  int total = arr.size();

  for (int i = 0; i < total; i++) {
    if (!condition(((T *)data)[i], arg))
      return false;
  }
  return true;
}

// Intrepydd functions
template <typename T> T sum(const py::array_t<T> &arr) {
  return reduce(arr, ope_add, (T)0);
}

template <typename T>
py::array_t<T> sum(const py::array_t<T> &arr, int axis, int version = 0) {
  int ndim = arr.ndim();
  std::vector<int64_t> newshape;
  auto input = arr.data();
  py::array_t<T> out;

  if (abs(axis) >= ndim) {
    std::cerr << "[Error] 'axis' entry is out of bounds.\n";
    exit(-1);
  } else if (axis < 0) {
    axis = ndim + axis;
    assert(axis >= 0 && axis < ndim);
  }
  for (int i = 0; i < ndim; i++) {
    if (i == axis) {
      continue;
    }
    newshape.push_back(arr.shape(i));
  }

  out = empty(newshape, T());
  auto data = out.mutable_data();
  for (int t = 0; t < out.size(); t++) {
    ((T *)data)[t] = (T)0;
  }

  if (ndim == 1) {
    data[0] = reduce(arr, ope_add, (T)0);
    return out;
  } else if (ndim == 2) {
    int ni = arr.shape(0);
    int nj = arr.shape(1);

    switch (version) {
    case 0:
      if (axis == 0) {
        for (int i = 0; i < ni; i++) {
          for (int j = 0; j < nj; j++) {
            ((T *)data)[j] += ((T *)input)[i * nj + j];
          }
        }
      } else if (axis == 1) {
        for (int i = 0; i < ni; i++) {
          for (int j = 0; j < nj; j++) {
            ((T *)data)[i] += ((T *)input)[i * nj + j];
          }
        }
      }
      break;
    case 1:
      if (axis == 0) {
        int par = 4; // Note: this is specific to sinkhorn with 4 cores
        int chunk = nj / par;
#pragma omp parallel for
        for (int p = 0; p < par; p++) {
          for (int i = 0; i < ni; i++) {
            for (int j = p * chunk; j < (p + 1) * chunk; j++) {
              ((T *)data)[j] += ((T *)input)[i * nj + j];
            }
          }
        }
      } else if (axis == 1) {
#pragma omp parallel for
        for (int i = 0; i < ni; i++) {
          for (int j = 0; j < nj; j++) {
            ((T *)data)[i] += ((T *)input)[i * nj + j];
          }
        }
      }
      break;
    default:
      std::cerr << "[Error] Unsupported version of binary.\n";
      exit(-1);
    }
    return out;
  } else if (ndim == 3) {
    for (int t = 0; t < arr.size(); t++) {
      int dim0 = t / (arr.shape(2) * arr.shape(1)); // axis = 0
      int dim1 = (t / arr.shape(2)) % arr.shape(1); // axis = 1
      int dim2 = t % arr.shape(2);                  // axis = 2
      if (axis == 0) {
        ((T *)data)[dim1 * newshape[1] + dim2] += ((T *)input)[t];
      } else if (axis == 1) {
        ((T *)data)[dim0 * newshape[1] + dim2] += ((T *)input)[t];
      } else if (axis == 2) {
        ((T *)data)[dim0 * newshape[1] + dim1] += ((T *)input)[t];
      }
    }
    return out;
  } else {
    std::cerr << "[Error] only 1,2,3-dim arrays are supported.\n";
    exit(-1);
  }
}

template <typename T> T prod(const py::array_t<T> &arr) {
  return reduce(arr, ope_mul, (T)1);
}
template <typename T> T min(const py::array_t<T> &arr) {
  auto data = arr.data();
  return reduce(arr, ope_minimum, *((T *)data));
}
template <typename T> T max(const py::array_t<T> &arr) {
  auto data = arr.data();
  return reduce(arr, ope_maximum, *((T *)data));
}

template <typename T> std::vector<int64_t> *argmin(const py::array_t<T> &arr) {
  auto data = arr.data();
  return reduce_index(arr, ope_lt, *((T *)data));
}
template <typename T> std::vector<int64_t> *argmax(const py::array_t<T> &arr) {
  auto data = arr.data();
  return reduce_index(arr, ope_gt, *((T *)data));
}

template <typename T> bool any(const py::array_t<T> &arr) {
  // true if any element is nonzero (false if all element is zero)
  return !all_meet_condition(arr, ope_eq, (T)0);
}
template <typename T> bool all(const py::array_t<T> &arr) {
  // true if all elements are nonzero
  return all_meet_condition(arr, ope_neq, (T)0);
}
template <typename T> bool allclose(const py::array_t<T> &arr, T eps) {
  // true if all elements are at most eps in absolute value
  return all_meet_condition(arr, ope_close, std::abs(eps));
}

template <typename T> py::array_t<int> where(const py::array_t<T> &condition) {
  auto data = condition.data();
  int size = condition.size();
  int ndim = condition.ndim();

  py::array_t<int> out;

  if (ndim == 1) {
    int total = 0;
    for (int i = 0; i < size; i++) {
      if ((bool)data[i]) {
        total++;
      }
    }
    int dummy = 0;

    out = empty(total, dummy);
    auto outdata = out.data();
    int idx = 0;
    for (int i = 0; i < size; i++) {
      if ((bool)data[i]) {
        ((int *)outdata)[idx++] = (int)i;
      }
    }
    assert(total - 1 == idx);
    return out;
  }

  std::cerr << "[Error] only 1-dim arrays are supported.\n";
  exit(-1);
}

} // namespace pydd

#endif // REDUCTION_HPP
