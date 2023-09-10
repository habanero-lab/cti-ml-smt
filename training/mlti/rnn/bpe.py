"""Byte pair encoding (BPE) for word segmentation."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Optional
import tqdm


class BPE:

    def __init__(self, subword_vocab: list[str], pairs: list[tuple[str, str]]):
        self.vocab = {'': 0}  # Reserved for unknown characters
        for i, w in enumerate(subword_vocab):
            self.vocab[w] = i + 1

        self._pairs = pairs
        self._pair_map: dict[str, dict[str, int]] = {}
        for i, (first, second) in enumerate(pairs):
            self._pair_map.setdefault(first, {})[second] = i

        self._cache: dict[str, list[str]] = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_cache']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._cache = {}

    def segment(self, word: str) -> list[str]:
        if word in self._cache:
            return self._cache[word]

        subwords = list(word)

        while len(subwords) > 1:
            min_pair_idx = float('inf')
            positions = []
            prev = subwords[0]
            for i in range(1, len(subwords)):
                now = subwords[i]
                second_idx_map = self._pair_map.get(prev, None)
                if second_idx_map is not None:
                    pair_idx = second_idx_map.get(now, None)
                    if pair_idx is not None:
                        if pair_idx < min_pair_idx:
                            min_pair_idx = pair_idx
                            positions = [i]
                        elif pair_idx == min_pair_idx:
                            # Make sure this location is not invalidated by the previous merge
                            if i - positions[len(positions) - 1] > 1:
                                positions.append(i)
                prev = now

            if not positions:
                break

            pair_str = subwords[positions[0] - 1] + subwords[positions[0]]
            new_subwords = []
            now = 0
            for pos in positions:
                new_subwords += subwords[now:pos - 1]
                new_subwords.append(pair_str)
                now = pos + 1
            new_subwords += subwords[now:]
            subwords = new_subwords

        self._cache[word] = subwords

        return subwords

    def segment_to_ids(self, word):
        return [self.vocab.get(subword, 0) for subword in self.segment(word)]


@dataclass
class _HeapElem:
    count: int
    subword_pair: tuple[str, str]
    heap_pos: int = -1

    def __lt__(self, other: '_HeapElem'):
        return self.count < other.count

    def __gt__(self, other: '_HeapElem'):
        return self.count > other.count


class _Heap:

    def __init__(self, elems: Optional[list[_HeapElem]] = None):
        self._elems: list[_HeapElem] = []
        if elems:
            self._elems[:] = elems

        for i, e in enumerate(self._elems):
            e.heap_pos = i
        for i in range(len(self) // 2 - 1, -1, -1):
            self.bubble_down(i)

    def __len__(self):
        return len(self._elems)

    def _swap(self, i: int, j: int):
        if i == j:
            return
        e_i = self._elems[i]
        e_j = self._elems[j]
        e_i.heap_pos = j
        e_j.heap_pos = i
        self._elems[i] = e_j
        self._elems[j] = e_i

    def bubble_down(self, i: int):
        n = len(self)
        while True:
            largest = i
            l = i * 2 + 1
            r = i * 2 + 2
            if l < n and self._elems[l] > self._elems[largest]:
                largest = l
            if r < n and self._elems[r] > self._elems[largest]:
                largest = r
            if largest != i:
                self._swap(largest, i)
                i = largest
            else:
                break

    def bubble_up(self, i: int):
        while i != 0:
            parent = (i - 1) // 2
            if self._elems[parent] < self._elems[i]:
                self._swap(parent, i)
                i = parent
            else:
                break

    def push(self, elem: _HeapElem):
        i = len(self)
        elem.heap_pos = i
        self._elems.append(elem)
        self.bubble_up(i)

    def pop(self):
        return self.remove(0)

    def top(self):
        return self._elems[0]

    def remove(self, i: int):
        if i == len(self) - 1:
            return self._elems.pop()
        e = self._elems[i]
        last = self._elems.pop()
        self._elems[i] = last
        last.heap_pos = i
        self.bubble_up(i)
        self.bubble_down(i)


def train_bpe(raw_vocab: dict[str, int], max_merges: int, min_freq: int):
    subword_set: set[str] = set()

    vocab_list: list[tuple[list[str], int]] = []
    for word, count in raw_vocab.items():
        subwords = list(word)
        subword_set.update(subwords)
        vocab_list.append((subwords, count))

    pair_heap, pair_to_heap_elem, pair_word_map = _init_pairs(vocab_list)

    merged_pairs: list[tuple[str, str]] = []

    for _ in tqdm.trange(max_merges):
        if len(pair_heap) == 0:
            break

        top = pair_heap.top()

        if top.count < min_freq:
            break

        merged_pairs.append(top.subword_pair)

        _merge_and_update(top.subword_pair, vocab_list, pair_heap,
                          pair_to_heap_elem, pair_word_map)

    subword_vocab: list[str] = list(subword_set)
    for first, second in merged_pairs:
        subword_vocab.append(first + second)

    return BPE(subword_vocab, merged_pairs)


def _init_pairs(vocab_list: list[tuple[list[str], int]]):
    pair_heap_elems: list[_HeapElem] = []
    pair_to_heap_elem: dict[tuple[str, str], _HeapElem] = {}
    pair_word_map: DefaultDict[ \
        tuple[str, str], \
        DefaultDict[int, int] \
    ] = defaultdict(lambda: defaultdict(int))
    for i, (word, count) in enumerate(vocab_list):
        prev_char = word[0]
        for char in word[1:]:
            pair = prev_char, char
            if pair in pair_to_heap_elem:
                pair_to_heap_elem[pair].count += count
            else:
                elem = _HeapElem(count, pair)
                pair_to_heap_elem[pair] = elem
                pair_heap_elems.append(elem)
            pair_word_map[pair][i] += 1
            prev_char = char

    pair_heap = _Heap(pair_heap_elems)

    return pair_heap, pair_to_heap_elem, pair_word_map


def _merge_and_update(pair: tuple[str, str], \
                      vocab_list: list[tuple[list[str], int]], \
                      pair_heap: _Heap, \
                      pair_to_heap_elem: dict[tuple[str, str], _HeapElem], \
                      pair_word_map: DefaultDict[tuple[str, str],
                                                 DefaultDict[int, int]]):
    first, second = pair
    pair_str = first + second

    invalidated_pairs = Counter()
    new_pairs = Counter()

    word_indices = list(pair_word_map[pair].keys())
    for word_idx in word_indices:
        word, count = vocab_list[word_idx]
        new_word = []
        i = 0
        n = len(word)
        while i < n:
            # Find a matching pair
            now = word[i]
            if not i + 1 < n or now != first:
                new_word.append(now)
                i += 1
                continue

            nxt = word[i + 1]
            if nxt != second:
                new_word.append(now)
                i += 1
                continue

            new_word.append(pair_str)

            # Invalidate overlapping pairs
            iv_pairs = [pair]
            if i > 0:
                invalidated_pair = word[i - 1], now
                iv_pairs.append(invalidated_pair)
            if i + 2 < n:
                if not (i + 3 < n and word[i + 2] == first
                        and word[i + 3] == second):
                    invalidated_pair = nxt, word[i + 2]
                    iv_pairs.append(invalidated_pair)
            for invalidated_pair in iv_pairs:
                invalidated_pairs[invalidated_pair] += count
                word_occurrence_map = pair_word_map[invalidated_pair]
                occur_count = word_occurrence_map[word_idx]
                occur_count -= 1
                if occur_count > 0:
                    word_occurrence_map[word_idx] = occur_count
                else:
                    del word_occurrence_map[word_idx]
                    if len(word_occurrence_map) == 0:
                        del pair_word_map[invalidated_pair]

            i += 2

        vocab_list[word_idx] = new_word, count

        # Discover new pairs
        for i, now in enumerate(new_word):
            if now != pair_str:
                continue
            if i > 0:
                prev = new_word[i - 1]
                new_pairs[prev, now] += count
                pair_word_map[prev, now][word_idx] += 1
            if i + 1 < len(new_word) and new_word[i + 1] != pair_str:
                nxt = new_word[i + 1]
                new_pairs[now, nxt] += count
                pair_word_map[now, nxt][word_idx] += 1

    # Update invalidated pairs in the heap
    for iv_pair, count in invalidated_pairs.items():
        elem = pair_to_heap_elem[iv_pair]
        elem.count -= count
        if elem.count > 0:
            pair_heap.bubble_down(elem.heap_pos)
        else:
            del pair_to_heap_elem[iv_pair]
            pair_heap.remove(elem.heap_pos)

    # Add new pairs to the heap
    for new_pair, count in new_pairs.items():
        elem = _HeapElem(count, new_pair)
        pair_to_heap_elem[new_pair] = elem
        pair_heap.push(elem)
