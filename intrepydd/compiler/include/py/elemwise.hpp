#ifndef ELEMWISE_HPP
#define ELEMWISE_HPP

#include <cstdlib>
#include <iterator>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

#include "py/operation.hpp"
#include "py/util.hpp"
#include "shared/NpArray.hpp"
#include "shared/inline.hpp"

// #define ELEMWISE_DEBUG

namespace pydd {

// Operators

template <typename T, typename R> PYDD_INLINE R unary(T x, R (*operation)(T)) {
  return operation(x);
}

template <typename T, typename R>
py::array_t<R> unary(const py::array_t<T> &arr, R (*operation)(T)) {
  auto data = arr.data();
  int total = arr.size();
  py::array_t<R> arr0(pydd::shape(arr));
  auto data0 = arr0.mutable_data();
  for (int i = 0; i < total; i++) {
    data0[i] = operation(data[i]);
  }
  return arr0;
}

template <typename T1, typename T2, typename R>
PYDD_INLINE R binary(T1 x, T2 y, R (*operation)(T1, T2)) {
  return operation(x, y);
}

template <typename T1, typename T2, typename R>
py::array_t<R> binary(const py::array_t<T1> &arr, T2 arg,
                      R (*operation)(T1, T2)) {
  auto data = arr.data();
  int total = arr.size();
  py::array_t<R> arr0(pydd::shape(arr));
  auto data0 = arr0.mutable_data();
  for (int i = 0; i < total; i++) {
    data0[i] = operation(data[i], arg);
  }
  return arr0;
}

template <typename T1, typename T2, typename R>
py::array_t<R> binary(T1 arg, const py::array_t<T2> &arr,
                      R (*operation)(T1, T2)) {
  auto data = arr.data();
  int total = arr.size();
  py::array_t<R> arr0(pydd::shape(arr));
  auto data0 = arr0.mutable_data();
  for (int i = 0; i < total; i++) {
    data0[i] = operation(arg, data[i]);
  }
  return arr0;
}

template <typename T1, typename T2>
bool is_same_shape(const py::array_t<T1> &arr1, const py::array_t<T2> &arr2) {
  int ndim = arr1.ndim();
  if (arr2.ndim() != ndim)
    return false;
  for (int d = 0; d < ndim; d++) {
    if (arr1.shape(d) != arr2.shape(d))
      return false;
  }
  return true;
}

template <typename T1, typename T2>
bool is_consistent_shape(const py::array_t<T1> &arr1,
                         const py::array_t<T2> &arr2, std::vector<int> &step1,
                         std::vector<int> &step2, std::vector<int> &shape) {
  int diff0 = arr1.ndim() - arr2.ndim();
  int diff = diff0;
  int ndim = (diff >= 0) ? arr1.ndim() : arr2.ndim();

  for (int d = 0; d < ndim; d++) {
    if (diff < 0) {
      step1.push_back(0);
      step2.push_back(1);
      shape.push_back(arr2.shape(d));
      diff++;
    } else if (diff > 0) {
      step1.push_back(1);
      step2.push_back(0);
      shape.push_back(arr1.shape(d));
      diff--;
    } else {
      int d1 = (diff0 >= 0) ? d : d + diff0;
      int d2 = (diff0 <= 0) ? d : d - diff0;
      if (arr1.shape(d1) == arr2.shape(d2)) {
        step1.push_back(1);
        step2.push_back(1);
        shape.push_back(arr1.shape(d1));
      } else if (arr1.shape(d1) == 1) {
        step1.push_back(0);
        step2.push_back(1);
        shape.push_back(arr2.shape(d2));
      } else if (arr2.shape(d2) == 1) {
        step1.push_back(1);
        step2.push_back(0);
        shape.push_back(arr1.shape(d1));
      } else {
        return false;
      }
    }
  }
  return true;
}

template <typename T1, typename T2, typename R>
auto binary(const py::array_t<T1> &arr1, const py::array_t<T2> &arr2,
            R (*operation)(T1, T2)) {
  bool is_same = is_same_shape(arr1, arr2);
  std::vector<int> step1, step2;
  std::vector<int> shape;
  if (!is_same && !is_consistent_shape(arr1, arr2, step1, step2, shape)) {
    std::cerr << "[Error] Element-wise binary operation assumes consistent "
                 "shape arrays.\n";
    exit(-1);
  }
  auto data1 = arr1.data();
  auto data2 = arr2.data();

  if (is_same) {
    int total = arr1.size();
    auto arr0 = py::array_t<R>(pydd::shape(arr1));
    auto data0 = arr0.mutable_data();
    for (int i = 0; i < total; i++) {
      data0[i] = operation(data1[i], data2[i]);
    }
    return arr0;
  } else {
    int ndim = shape.size();
    if (ndim > 5) {
      std::cerr << "[Error] Unsupported dimensionality.\n";
      exit(-1);
    }

    auto arr0 = py::array_t<R>(shape);
    auto data0 = arr0.mutable_data();

    if (ndim < 5) {
      step1.insert(step1.begin(), 5 - ndim, 1);
      step2.insert(step2.begin(), 5 - ndim, 1);
      shape.insert(shape.begin(), 5 - ndim, 1);
    }
    std::vector<int> stride0(5, 1), stride1(5, 1), stride2(5, 1);
    for (int d = 3; d >= 0; d--) {
      stride0[d] = shape[d + 1] * stride0[d + 1];
      stride1[d] = (step1[d + 1] == 1 ? shape[d + 1] : 1) * stride1[d + 1];
      stride2[d] = (step2[d + 1] == 1 ? shape[d + 1] : 1) * stride2[d + 1];
    }

    for (int i0_0 = 0; i0_0 < shape[0]; i0_0++) {
      int os0_0 = i0_0 * stride0[0];
      int os1_0 = (step1[0] == 1) ? i0_0 * stride1[0] : 0;
      int os2_0 = (step2[0] == 1) ? i0_0 * stride2[0] : 0;
      for (int i0_1 = 0; i0_1 < shape[1]; i0_1++) {
        int os0_1 = os0_0 + i0_1 * stride0[1];
        int os1_1 = os1_0 + ((step1[1] == 1) ? i0_1 * stride1[1] : 0);
        int os2_1 = os2_0 + ((step2[1] == 1) ? i0_1 * stride2[1] : 0);
        for (int i0_2 = 0; i0_2 < shape[2]; i0_2++) {
          int os0_2 = os0_1 + i0_2 * stride0[2];
          int os1_2 = os1_1 + ((step1[2] == 1) ? i0_2 * stride1[2] : 0);
          int os2_2 = os2_1 + ((step2[2] == 1) ? i0_2 * stride2[2] : 0);
          for (int i0_3 = 0; i0_3 < shape[3]; i0_3++) {
            int os0_3 = os0_2 + i0_3 * stride0[3];
            int os1_3 = os1_2 + ((step1[3] == 1) ? i0_3 * stride1[3] : 0);
            int os2_3 = os2_2 + ((step2[3] == 1) ? i0_3 * stride2[3] : 0);
            if (step1[4] == 1 && step2[4] == 1) {
              for (int i0_4 = 0; i0_4 < shape[4]; i0_4++)
                data0[os0_3 + i0_4] =
                    operation(data1[os1_3 + i0_4], data2[os2_3 + i0_4]);
            } else if (step1[4] == 1) {
              for (int i0_4 = 0; i0_4 < shape[4]; i0_4++)
                data0[os0_3 + i0_4] =
                    operation(data1[os1_3 + i0_4], data2[os2_3]);
            } else if (step2[4] == 1) {
              for (int i0_4 = 0; i0_4 < shape[4]; i0_4++)
                data0[os0_3 + i0_4] =
                    operation(data1[os1_3], data2[os2_3 + i0_4]);
            } else {
              assert(0);
            }
          }
        }
      }
    }
    return arr0;
  }
}

// Unary

#define DEFINE_UNARY_WITH_OPE(name, ope)                                       \
  template <typename T> PYDD_INLINE auto name(T x) {                           \
    return unary(x, ope<T>);                                                   \
  }                                                                            \
  template <typename T> PYDD_INLINE auto name(const py::array_t<T> &x) {       \
    return unary(x, ope<T>);                                                   \
  }
#define DEFINE_UNARY(name) DEFINE_UNARY_WITH_OPE(name, ope_##name)

DEFINE_UNARY(minus)
DEFINE_UNARY(abs)
DEFINE_UNARY(logical_not)
DEFINE_UNARY_WITH_OPE(elemwise_not, ope_logical_not)
DEFINE_UNARY(isnan)
DEFINE_UNARY(isinf)
DEFINE_UNARY(sqrt)
DEFINE_UNARY(exp)
DEFINE_UNARY(log)
DEFINE_UNARY(cos)
DEFINE_UNARY(sin)
DEFINE_UNARY(tan)
DEFINE_UNARY(acos)
DEFINE_UNARY(asin)
DEFINE_UNARY(atan)

#undef DEFINE_UNARY_WITH_OPE
#undef DEFINE_UNARY

// Binary
#define DEFINE_BINARY_WITH_OPE(name, ope)                                      \
  template <typename T1, typename T2> PYDD_INLINE auto name(T1 x1, T2 x2) {    \
    return binary(x1, x2, ope<T1, T2>);                                        \
  }                                                                            \
  template <typename T1, typename T2>                                          \
  PYDD_INLINE auto name(const py::array_t<T1> &x1, T2 x2) {                    \
    return binary(x1, x2, ope<T1, T2>);                                        \
  }                                                                            \
  template <typename T1, typename T2>                                          \
  PYDD_INLINE auto name(T1 x1, const py::array_t<T2> &x2) {                    \
    return binary(x1, x2, ope<T1, T2>);                                        \
  }                                                                            \
  template <typename T1, typename T2>                                          \
  PYDD_INLINE auto name(const py::array_t<T1> &x1,                             \
                        const py::array_t<T2> &x2) {                           \
    return binary(x1, x2, ope<T1, T2>);                                        \
  }
#define DEFINE_BINARY_WITH_OPE_AND_OPERATOR(name, ope, op)                     \
  DEFINE_BINARY_WITH_OPE(name, ope)                                            \
  template <typename T1, typename T2>                                          \
  PYDD_INLINE auto operator op(const py::array_t<T1> &x1, T2 x2) {             \
    return binary(x1, x2, ope<T1, T2>);                                        \
  }                                                                            \
  template <typename T1, typename T2>                                          \
  PYDD_INLINE auto operator op(T1 x1, const py::array_t<T2> &x2) {             \
    return binary(x1, x2, ope<T1, T2>);                                        \
  }                                                                            \
  template <typename T1, typename T2>                                          \
  PYDD_INLINE auto operator op(const py::array_t<T1> &x1,                      \
                               const py::array_t<T2> &x2) {                    \
    return binary(x1, x2, ope<T1, T2>);                                        \
  }
#define DEFINE_BINARY(name) DEFINE_BINARY_WITH_OPE(name, ope_##name)
#define DEFINE_BINARY_WITH_OPERATOR(name, op)                                  \
  DEFINE_BINARY_WITH_OPE_AND_OPERATOR(name, ope_##name, op)

DEFINE_BINARY_WITH_OPERATOR(add, +)
DEFINE_BINARY_WITH_OPERATOR(sub, -)
DEFINE_BINARY_WITH_OPERATOR(mul, *)
DEFINE_BINARY_WITH_OPERATOR(div, /)
DEFINE_BINARY(floor_div)
DEFINE_BINARY(pow)
DEFINE_BINARY(log)
DEFINE_BINARY(maximum)

DEFINE_BINARY_WITH_OPERATOR(eq, ==)
DEFINE_BINARY_WITH_OPERATOR(neq, !=)
DEFINE_BINARY_WITH_OPERATOR(lt, <)
DEFINE_BINARY_WITH_OPERATOR(gt, >)
DEFINE_BINARY_WITH_OPERATOR(le, <=)
DEFINE_BINARY_WITH_OPERATOR(ge, >=)
DEFINE_BINARY_WITH_OPERATOR(logical_and, &&)
DEFINE_BINARY_WITH_OPERATOR(logical_or, ||)
DEFINE_BINARY(logical_xor)

#undef DEFINE_BINARY_WITH_OPE
#undef DEFINE_BINARY_WITH_OPE_AND_OPERATOR
#undef DEFINE_BINARY
#undef DEFINE_BINARY_WITH_OPERATOR

template <typename... Args>
PYDD_INLINE std::vector<int64_t> compatibility_check(const py::array &first,
                                                     const Args &...args) {
  std::vector<std::reference_wrapper<const py::array>> arrays{{first, args...}};

#ifdef ELEMWISE_DEBUG
  std::cout << "[Verbose] compatibility_check by variadic arguments\n";
#endif

  int count = arrays.size();

  if (count == 0)
    return {};

  int ndim = 0;
  for (int i = 0; i < count; i++) {
    const auto &array = arrays[i].get();
    int nd = array.ndim();
#ifdef ELEMWISE_DEBUG
    std::cout << "[Verbose] array at " << array << " with ndim = " << nd
              << std::endl;
#endif
    if (nd > ndim)
      ndim = nd;
  }

  std::vector<int64_t> shape(ndim);

  for (int d = 0; d < ndim; d++) {
    int size = 1;
    for (int i = 0; i < count; i++) {
      const auto &array = arrays[i].get();
      int nd = array.ndim();
      int idx = d + nd - ndim;
      if (idx >= 0) {
        int sz = array.shape(idx);
#ifdef ELEMWISE_DEBUG
        std::cout << "[Verbose] dim = " << d << ", i = " << i << ", sz = " << sz
                  << ", size = " << size << std::endl;
#endif
        if (size == 1) {
          size = sz;
        } else if (sz != 1 && sz != size) {

          std::string msg = "[Error] Element-wise binary operation assumes "
                            "consistent shape arrays.\n";
          std::cerr << msg;
          for (int j = 0; j < count; ++j) {
            const auto &array = arrays[j].get();
            std::cerr << array.shape(0) << std::endl;
          }
          throw std::length_error(msg);
          // exit(-1);
        }
      }
    }
    shape[d] = size;
  }

#ifdef ELEMWISE_DEBUG
  std::cout << "[Verbose] Arguments satisfied compatibility for element-wise "
               "operation\n";
#endif

  return shape;
}

} // namespace pydd

#endif // ELEMWISE_HPP
