#ifndef HEAPQ_HPP
#define HEAPQ_HPP

#include <algorithm>
#include <cstdlib>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace pydd {

/*
  Note:
  We need min heap for ipnsw while std::make_heap is max heap.
  So, key is multiplied by -1 to support min heap...
 */

template <typename K, typename V> class Heap {

private:
  std::vector<std::pair<K, V>> _heap_data;

public:
  Heap() {}

  Heap(K key, V data) {
    this->_heap_data.push_back(std::make_pair(-key, data));
  }

  void push(K key, V data) {
    this->_heap_data.push_back(std::make_pair(-key, data));
    std::push_heap(this->_heap_data.begin(), this->_heap_data.end());
  }

  std::pair<K, V> pop() {
    std::pop_heap(this->_heap_data.begin(), this->_heap_data.end());
    auto res = this->_heap_data.back();
    this->_heap_data.pop_back();
    res.first = -res.first;
    return res;
  }

  K peek_key() const { return -this->_heap_data.front().first; }

  V peek_val() const { return this->_heap_data.front().second; }

  int get_size() const { return _heap_data.size(); }

  const std::vector<std::pair<K, V>> &data() const { return _heap_data; }
};

template <typename K, typename V> Heap<K, V> heapinit(K key, V data) {
  return Heap<K, V>(key, data);
}

template <typename K, typename V> Heap<K, V> heapinit_empty() {
  return Heap<K, V>();
}

template <typename K, typename V>
void heappush(Heap<K, V> &tgt_heap, K key, V data) {
  tgt_heap.push(key, data);
}

template <typename K, typename V> void heappop(Heap<K, V> &tgt_heap) {
  tgt_heap.pop();
}

template <typename K, typename V> K heappeek_key(const Heap<K, V> &tgt_heap) {
  return tgt_heap.peek_key();
}

template <typename K, typename V> V heappeek_val(const Heap<K, V> &tgt_heap) {
  return tgt_heap.peek_val();
}

template <typename K, typename V> int heapsize(const Heap<K, V> &tgt_heap) {
  return tgt_heap.get_size();
}

template <typename K, typename V> int len(const Heap<K, V> &tgt_heap) {
  return tgt_heap.get_size();
}

// template <typename K, typename V> K getitem(const Heap<K, V> &tgt_heap, int
// i) {
//   return tgt_heap.data()[i];
// }

template <typename K, typename V>
K heap_get_key(const Heap<K, V> &tgt_heap, int i) {
  return (-tgt_heap.data()[i].first);
}

} // namespace pydd

#endif // HEAPQ_HPP
