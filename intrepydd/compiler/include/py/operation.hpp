#ifndef OPERATION_HPP
#define OPERATION_HPP

#include <cmath>

#include "shared/inline.hpp"

namespace pydd {

template <typename T> PYDD_INLINE typename std::decay_t<T> ope_minus(T x) {
  return -x;
}
template <typename T> PYDD_INLINE typename std::decay_t<T> ope_abs(T x) {
  return std::abs(x);
}
template <typename T> PYDD_INLINE bool ope_logical_not(T x) { return !x; }
template <typename T> PYDD_INLINE bool ope_isnan(T x) { return std::isnan(x); }
template <typename T> PYDD_INLINE bool ope_isinf(T x) { return std::isinf(x); }
template <typename T> PYDD_INLINE double ope_sqrt(T x) { return std::sqrt(x); }
template <typename T> PYDD_INLINE double ope_exp(T x) { return std::exp(x); }
template <typename T> PYDD_INLINE double ope_log(T x) { return std::log(x); }
template <typename T> PYDD_INLINE double ope_cos(T x) { return std::cos(x); }
template <typename T> PYDD_INLINE double ope_sin(T x) { return std::sin(x); }
template <typename T> PYDD_INLINE double ope_tan(T x) { return std::tan(x); }
template <typename T> PYDD_INLINE double ope_acos(T x) { return std::acos(x); }
template <typename T> PYDD_INLINE double ope_asin(T x) { return std::asin(x); }
template <typename T> PYDD_INLINE double ope_atan(T x) { return std::atan(x); }

template <typename T> PYDD_INLINE typename std::decay_t<T> ope_nnz(T x, T y) {
  return x != 0 ? y + 1 : y;
}

template<typename T> struct identity { using type = T; };

#define DTYPE_COERCION_RULE(x, y, z)                                           \
  std::conditional_t <                                                         \
      std::is_same<_T1, x>::value && std::is_same<_T2, y>::value,              \
      identity<z>,

template <class T1, class T2> struct dtype_coercion {
  typedef typename std::decay_t<T1> _T1;
  typedef typename std::decay_t<T2> _T2;
  typedef typename
  DTYPE_COERCION_RULE(bool, bool, bool)
  DTYPE_COERCION_RULE(bool, int32_t, int32_t)
  DTYPE_COERCION_RULE(bool, int64_t, int64_t)
  DTYPE_COERCION_RULE(bool, float, float)
  DTYPE_COERCION_RULE(bool, double, double)
  DTYPE_COERCION_RULE(int32_t, bool, int32_t)
  DTYPE_COERCION_RULE(int32_t, int32_t, int32_t)
  DTYPE_COERCION_RULE(int32_t, int64_t, int64_t)
  DTYPE_COERCION_RULE(int32_t, float, double)
  DTYPE_COERCION_RULE(int32_t, double, double)
  DTYPE_COERCION_RULE(int64_t, bool, int64_t)
  DTYPE_COERCION_RULE(int64_t, int32_t, int64_t)
  DTYPE_COERCION_RULE(int64_t, int64_t, int64_t)
  DTYPE_COERCION_RULE(int64_t, float, double)
  DTYPE_COERCION_RULE(int64_t, double, double)
  DTYPE_COERCION_RULE(float, bool, float)
  DTYPE_COERCION_RULE(float, int32_t, double)
  DTYPE_COERCION_RULE(float, int64_t, double)
  DTYPE_COERCION_RULE(float, float, float)
  DTYPE_COERCION_RULE(float, double, double)
  DTYPE_COERCION_RULE(double, bool, double)
  DTYPE_COERCION_RULE(double, int32_t, double)
  DTYPE_COERCION_RULE(double, int64_t, double)
  DTYPE_COERCION_RULE(double, float, double)
  DTYPE_COERCION_RULE(double, double, double)
  void>>>>>>>>>>>>>>>>>>>>>>>>>::type type;
};

#undef DTYPE_COERCION_RULE

#define DEFINE_BINARY_OPE(name, expr)                                          \
  template <typename T1, typename T2>                                          \
  PYDD_INLINE auto ope_##name(T1 x, T2 y)                                      \
      ->typename dtype_coercion<T1, T2>::type {                                \
    return expr;                                                               \
  }

DEFINE_BINARY_OPE(add, x + y)
DEFINE_BINARY_OPE(sub, x - y)
DEFINE_BINARY_OPE(mul, (x * y))
DEFINE_BINARY_OPE(minimum, x < y ? x : y)
DEFINE_BINARY_OPE(maximum, x > y ? x : y)

// Floor division (x // y)
DEFINE_BINARY_OPE(floor_div, std::floor(static_cast<double>(x) / y))

#undef DEFINE_BINARY_OPE

// True division (x / y)
template <typename T1, typename T2> PYDD_INLINE double ope_div(T1 x, T2 y) {
  return static_cast<double>(x) / y;
}

template <typename T1, typename T2>
PYDD_INLINE double ope_pow(T1 base, T2 exp) {
  return std::pow(base, exp);
}
template <typename T1, typename T2> PYDD_INLINE double ope_log(T1 x, T2 base) {
  return std::log(x) / std::log(base);
}

#define DEFINE_BOOL_BINARY_OPE(name, expr)                                     \
  template <typename T1, typename T2>                                          \
  PYDD_INLINE bool ope_##name(T1 x, T2 y) {                                    \
    return expr;                                                               \
  }

DEFINE_BOOL_BINARY_OPE(eq, x == y)
DEFINE_BOOL_BINARY_OPE(neq, x != y)
DEFINE_BOOL_BINARY_OPE(lt, x < y)
DEFINE_BOOL_BINARY_OPE(gt, x > y)
DEFINE_BOOL_BINARY_OPE(le, x <= y)
DEFINE_BOOL_BINARY_OPE(ge, x >= y)
DEFINE_BOOL_BINARY_OPE(logical_and, x &&y)
DEFINE_BOOL_BINARY_OPE(logical_or, x || y)
DEFINE_BOOL_BINARY_OPE(logical_xor, !x != !y)
DEFINE_BOOL_BINARY_OPE(close, std::abs(x) <= y)

#undef DEFINE_BOOL_BINARY_OPE

} // namespace pydd

#endif // OPERATION_HPP
