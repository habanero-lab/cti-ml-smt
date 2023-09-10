#ifndef CPPRT_H
#define CPPRT_H

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <vector>

#include <pybind11/pybind11.h>

using namespace pybind11::literals;
namespace py = pybind11;

namespace pydd {

template <typename T> void print(std::vector<T> *v) {
  for (T e : *v) {
    py::print(e, "end"_a = " ");
  }
}

template <typename T> void print(T e) { py::print(e); }

template <typename T, typename... Args> void print(T first, Args... args) {
  py::print(first);
  print(args...);
}

} // namespace pydd

#endif
