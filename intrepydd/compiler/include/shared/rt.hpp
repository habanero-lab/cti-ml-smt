#ifndef CPPRT_H
#define CPPRT_H

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "inline.hpp"

namespace pydd {

std::vector<const char *> callstack;

void append_to_call_stack(const char *funcname) {
  callstack.push_back(funcname);
}

void print_call_stack() {
  std::cout << "=========== Pydd call stack =========" << std::endl;
  for (auto f : callstack) {
    std::cout << f << std::endl;
  }
}

void _debug_print(int n) {
  std::cout << "[DEBUG] location stamp: " << n << std::endl;
}

template <typename T> int64_t len(const T &container) {
  return container.size();
}

template <typename T> void print(const T &e) { std::cout << e << std::endl; }

template <typename T, typename... Args>
void print(const T &first, const Args &...args) {
  print(first);
  print(args...);
}

template <typename T> void print(const std::vector<T> &v) {
  for (const T &e : v) {
    print(e, " ");
  }
  print('\n');
}

template <typename T>
const T &getitem(const std::vector<T> &container, int64_t i) {
  return container.at(i);
}

// template <typename K, typename V>
int getitem(typename std::map<int, int>::iterator &it) { return it->first; }

template <typename T>
void setitem(std::vector<T> &container, const T &v, int64_t i) {
  container[i] = v;
}

template <typename K, typename V>
void setitem(std::map<K, V> &container, const V &v, const K &i) {
  container[i] = v;
}

template <typename K, typename V>
const V &getitem(const std::map<K, V> &container, const K &i) {
  return container[i];
}

template <typename K, typename V>
void setitem(std::unordered_map<K, V> &container, const V &v, const K &i) {
  container[i] = v;
}

template <typename K, typename V>
const V &getitem(const std::unordered_map<K, V> &container, const K &i) {
  return container.at(i);
}

std::vector<int> range(int64_t len) {
  std::vector<int> ret;
  ret.resize(len);
  for (int64_t i = 0; i < len; ++i)
    ret[i] = i;
  return ret;
}

template <typename T> void append(std::vector<T> &vec, const T &v) {
  vec.push_back(v);
}

int int32() { return 0; }

int64_t int64() { return 0; }

float float32() { return 0; }

double float64() { return 0; }

bool boolean() { return 0; }

template <typename T> int int32(T x) { return static_cast<int>(x); }

template <typename T> int64_t int64(T x) { return static_cast<int64_t>(x); }

template <typename T> float float32(T x) { return static_cast<float>(x); }

template <typename T> double float64(T x) { return static_cast<double>(x); }

int stoi(const std::string &s) { return std::atoi(s.c_str()); }

int hextoi(const std::string &s) {
  unsigned int x;
  std::stringstream ss;
  ss << std::hex << s;
  ss >> x;
  return x;
}

int strtol(const std::string &s, int base) {
  return std::strtol(s.c_str(), NULL, base);
}

int randint(int low, int high) { return std::rand() % (high - low + 1) + low; }

// PYDD_INLINE function to swap two numbers
PYDD_INLINE void swap(char *x, char *y) {
  char t = *x;
  *x = *y;
  *y = t;
}

// function to reverse buffer[i..j]
PYDD_INLINE char *reverse(char *buffer, int i, int j) {
  while (i < j)
    swap(&buffer[i++], &buffer[j--]);

  return buffer;
}

// Iterative function to implement itoa() function in C
char *itoa(int value, char *buffer, int base) {
  // invalid input
  if (base < 2 || base > 32)
    return buffer;

  // consider absolute value of number
  int n = abs(value);
  // int n = value;

  int i = 0;
  while (n) {
    // int r = n % base;
    int r;
    if (base == 16) {
      r = n - (n >> 4) * 16;
    } else {
      r = n % base;
    }

    if (r >= 10)
      buffer[i++] = 65 + (r - 10);
    else
      buffer[i++] = 48 + r;

    // n = n / base;
    if (base == 16) {
      n = n >> 4;
    } else {
      n = n / base;
    }
  }

  // if number is 0
  if (i == 0)
    buffer[i++] = '0';

  // If base is 10 and value is negative, the resulting string
  // is preceded with a minus sign (-)
  // With any other base, value is always considered unsigned
  if (value < 0 && base == 10)
    buffer[i++] = '-';

  buffer[i] = '\0'; // null terminate string

  // reverse the string and return it
  return reverse(buffer, 0, i - 1);
}

PYDD_INLINE std::string hex(int n) {
  char hex[17] = {0};
  itoa(n, hex, 16);
  return std::string(hex);
}

class Timer {
public:
  Timer() : beg_(clock_::now()) {}
  void reset() { beg_ = clock_::now(); }
  double elapsed() const {
    return std::chrono::duration_cast<second_>(clock_::now() - beg_).count();
  }

private:
  typedef std::chrono::high_resolution_clock clock_;
  typedef std::chrono::duration<double, std::ratio<1>> second_;
  std::chrono::time_point<clock_> beg_;
};

Timer glb_timer;

double time() { return glb_timer.elapsed(); }

} // namespace pydd

#endif
