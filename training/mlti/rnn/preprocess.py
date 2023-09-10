from typing import NamedTuple, Optional, Sequence
from pathlib import Path
import pickle
import os
from collections import Counter
import numpy as np
import logging
import multiprocessing as mp
import tqdm
import traceback
import ast
import asttokens
from ..common.pydd_types import dataset_type_to_seq, type_vocab, annotation_type_to_seq
from .name_split import split_name
from .bpe import train_bpe, BPE
from .tokenize import tokenize, TokenKind


class Vocab(NamedTuple):
    subname_bpe: BPE
    nonname_vocab: dict[str, int]


def preprocess_dataset(raw_dataset_path: Path,
                       bpe_max_merges: int,
                       bpe_min_freq: int,
                       nonname_min_freq: int,
                       split_ratios: Sequence[float],
                       vocab_path: Path,
                       dataset_path: Path,
                       seed_sequence: np.random.SeedSequence,
                       num_processes: Optional[int] = None):

    logging.info('Loading the raw dataset')

    raw_data = []
    with open(raw_dataset_path, 'rb') as f:
        while True:
            try:
                src, arg_types = pickle.load(f)
            except EOFError:
                break
            raw_data.append((src, arg_types))

    assert len(split_ratios) == 3
    r = np.array(split_ratios) / sum(split_ratios)
    train_size = int(len(raw_data) * r[0])
    val_size = int(len(raw_data) * r[1])
    test_size = int(len(raw_data) * r[2])
    if len(raw_data) > train_size + val_size + test_size:
        train_size += len(raw_data) - (train_size + val_size + test_size)

    rng = np.random.default_rng(seed_sequence)
    rng.shuffle(raw_data)
    train_raw_data = raw_data[:train_size]
    val_raw_data = raw_data[train_size:train_size + val_size]
    test_raw_data = raw_data[train_size + val_size:]

    logging.info('Generating vocab')

    if num_processes is None:
        num_processes = os.cpu_count() or 1

    input_queue = mp.Queue()
    output_queue = mp.Queue()
    workers = [
        mp.Process(target=_collect_vocab_worker_fn,
                   args=(input_queue, output_queue),
                   daemon=True) for _ in range(num_processes)
    ]
    for worker in workers:
        worker.start()

    for x in train_raw_data:
        input_queue.put(x)
    input_queue.put(None)

    subname_counter = Counter()
    nonname_counter = Counter()
    type_counter = Counter()
    for _ in tqdm.trange(len(train_raw_data) + num_processes):
        kind, ret = output_queue.get()
        if kind == 0:
            continue
        else:
            subname_counter.update(ret[0])
            nonname_counter.update(ret[1])
            type_counter.update(ret[2])

    for worker in workers:
        worker.join()
        worker.close()
    input_queue.close()
    output_queue.close()

    subname_bpe = train_bpe(subname_counter, bpe_max_merges, bpe_min_freq)
    nonname_vocab = _generate_vocab(nonname_counter, nonname_min_freq)
    vocab = Vocab(subname_bpe, nonname_vocab)

    freq_type_seqs = [
        k for k, _ in sorted(
            type_counter.items(), key=lambda x: x[1], reverse=True)
    ]

    vocab_path.parent.mkdir(exist_ok=True)
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
        pickle.dump(freq_type_seqs, f)

    logging.info('Generating training set')

    train_data = _generate_dataset(train_raw_data, vocab, num_processes)

    logging.info('Generating validation set')
    val_data = _generate_dataset(val_raw_data, vocab, num_processes)

    logging.info('Generating test set')
    test_data = _generate_dataset(test_raw_data, vocab, num_processes)

    dataset_path.parent.mkdir(exist_ok=True)
    with open(dataset_path, 'wb') as f:
        pickle.dump((train_data, val_data, test_data), f)


def _generate_vocab(counter: dict[str, int], min_freq: int):
    vocab_count_list = sorted(counter.items(),
                              key=lambda kv: kv[1],
                              reverse=True)
    total = sum(map(lambda wc: wc[1], vocab_count_list))
    in_vocab = 0
    vocab = {'': 0}  # Reserved for unknown words
    for i, (word, count) in enumerate(vocab_count_list):
        if count < min_freq:
            break
        vocab[word] = i + 1
        in_vocab += count

    logging.info('Vocab: %d / %d, coverage: %.4f (%d / %d)', len(vocab),
                 len(counter), in_vocab / total, in_vocab, total)

    return vocab


def _collect_vocab_worker_fn(input_queue: mp.Queue, output_queue: mp.Queue):
    subname_counter = Counter()
    nonname_counter = Counter()
    type_counter = Counter()
    while True:
        task = input_queue.get()
        if task is None:
            input_queue.put(None)
            break
        src, arg_types = task
        _collect_vocab_fn(src, arg_types, subname_counter, nonname_counter,
                          type_counter)
        output_queue.put((0, None))
    output_queue.put((1, (subname_counter, nonname_counter, type_counter)))


def _collect_vocab_fn(src: str,
                      arg_types: dict[str, Optional[tuple]],
                      subname_counter: Counter,
                      nonname_counter: Counter,
                      type_counter: Counter):
    arg_type_seqs: dict[str, list[str]] = {}
    for arg_name, typ in arg_types.items():
        seq = dataset_type_to_seq(typ)
        if seq is None:
            continue
        arg_type_seqs[arg_name] = seq

    if not arg_type_seqs:
        return None

    try:
        tokens = tokenize(src)
    except SyntaxError:
        traceback.print_exc()
        return None

    for t in tokens:
        if t.kind == TokenKind.Name:
            subname_counter.update(split_name(t.string))
        else:
            token_str = f'{t.kind.value}{t.string}'
            nonname_counter[token_str] += 1
    for seq in arg_type_seqs.values():
        type_counter[tuple(seq)] += 1


def _generate_dataset(raw_data, vocab: Vocab, num_processes: int):
    input_queue = mp.Queue()
    output_queue = mp.Queue()
    workers = [
        mp.Process(target=_preprocess_dataset_worker_fn,
                   args=(vocab, input_queue, output_queue),
                   daemon=True) for _ in range(num_processes)
    ]
    for worker in workers:
        worker.start()

    for x in raw_data:
        input_queue.put(x)
    input_queue.put(None)

    data = []
    for _ in tqdm.trange(len(raw_data)):
        ret = output_queue.get()
        if ret is None:
            continue
        data.append(ret)

    for worker in workers:
        worker.join()
        worker.close()
    input_queue.close()
    output_queue.close()

    return data


def _preprocess_dataset_worker_fn(vocab: Vocab,
                                  input_queue: mp.Queue,
                                  output_queue: mp.Queue):
    while True:
        task = input_queue.get()
        if task is None:
            input_queue.put(None)
            break

        src, arg_types = task
        ret = _preprocess_dataset_fn(src, arg_types, vocab)
        output_queue.put(ret)


def _preprocess_dataset_fn(src: str,
                           arg_types: dict[str, Optional[tuple]],
                           vocab: Vocab):
    arg_type_seqs: dict[str, list[str]] = {}
    for arg_name, typ in arg_types.items():
        seq = dataset_type_to_seq(typ)
        if seq is None:
            continue
        arg_type_seqs[arg_name] = seq

    if not arg_type_seqs:
        return None

    try:
        tokens = tokenize(src)
    except SyntaxError:
        traceback.print_exc()
        return None

    name_arrs: list[np.ndarray] = []
    name_to_name_id: dict[str, int] = {}
    tokens_arr = []
    for t in tokens:
        if t.kind == TokenKind.Name:
            name = t.string
            if name in name_to_name_id:
                name_id = name_to_name_id[name]
            else:
                seg_ids = []
                for subname in split_name(name):
                    seg_ids += vocab.subname_bpe.segment_to_ids(subname)
                name_arr = np.array(seg_ids, dtype=np.int64)
                name_id = len(name_arrs)
                name_arrs.append(name_arr)
                name_to_name_id[name] = name_id
            tokens_arr.append((0, name_id))
        else:
            token_str = f'{t.kind.value}{t.string}'
            nonname_id = vocab.nonname_vocab.get(token_str, 0)
            tokens_arr.append((1, nonname_id))
    tokens_arr = np.array(tokens_arr, dtype=np.int64)

    type_seqs_arr = []
    infer_name_ids = []
    for arg_name, seq in arg_type_seqs.items():
        seq_arr = np.array([type_vocab[s] for s in seq], dtype=np.int64)
        type_seqs_arr.append(seq_arr)
        infer_name_ids.append(name_to_name_id[arg_name])
    infer_name_ids = np.array(infer_name_ids, dtype=np.int64)

    return tokens_arr, name_arrs, infer_name_ids, type_seqs_arr


def preprocess_pydd_src(src: str, vocab: Vocab):
    src = src.replace('pfor', 'for')

    func_names: list[str] = []
    func_param_to_infer_id: list[dict[str, int]] = []
    batch: list[tuple[np.ndarray, list[np.ndarray],
                      np.ndarray, list[Optional[np.ndarray]]]] = []

    ast_tokens = asttokens.ASTTokens(src, parse=True)
    for fdef in ast_tokens.tree.body:
        if not isinstance(fdef, ast.FunctionDef):
            continue
        func_names.append(fdef.name)
        func_src = ast_tokens.get_text(fdef)
        (tokens_arr, name_arrs, infer_name_ids, type_seqs_arr,
         param_to_infer_id) = _preprocess_pydd_func_src(func_src, vocab)
        func_param_to_infer_id.append(param_to_infer_id)
        batch.append((tokens_arr, name_arrs, infer_name_ids, type_seqs_arr))

    return batch, func_param_to_infer_id, func_names


def _preprocess_pydd_func_src(src: str, vocab: Vocab):
    tree = ast.parse(src)
    assert len(tree.body) == 1
    fdef = tree.body[0]
    assert isinstance(fdef, ast.FunctionDef)
    arg_type_seqs: dict[str, Optional[list[str]]] = {}
    for arg in fdef.args.args:
        arg_name = arg.arg
        if arg.annotation is None:
            type_seq = None
        else:
            type_seq = annotation_type_to_seq(arg.annotation)
            if type_seq is None:
                logging.info(
                    f'Unsupported arg type for {fdef.name}.{arg_name}: '
                    f'{ast.unparse(arg.annotation)}')
        arg_type_seqs[arg_name] = type_seq

    tokens = tokenize(src)

    name_arrs: list[np.ndarray] = []
    name_to_name_id: dict[str, int] = {}
    tokens_arr = []
    for t in tokens:
        if t.kind == TokenKind.Name:
            name = t.string
            if name in name_to_name_id:
                name_id = name_to_name_id[name]
            else:
                seg_ids = []
                for subname in split_name(name):
                    seg_ids += vocab.subname_bpe.segment_to_ids(subname)
                name_arr = np.array(seg_ids, dtype=np.int64)
                name_id = len(name_arrs)
                name_arrs.append(name_arr)
                name_to_name_id[name] = name_id
            tokens_arr.append((0, name_id))
        else:
            token_str = f'{t.kind.value}{t.string}'
            nonname_id = vocab.nonname_vocab.get(token_str, 0)
            tokens_arr.append((1, nonname_id))
    tokens_arr = np.array(tokens_arr, dtype=np.int64)

    type_seqs_arr = []
    infer_name_ids = []
    param_to_infer_id: dict[str, int] = {}
    for arg_name, seq in arg_type_seqs.items():
        if seq is not None:
            seq_arr = np.array([type_vocab[s] for s in seq], dtype=np.int64)
        else:
            seq_arr = None
        type_seqs_arr.append(seq_arr)
        infer_name_ids.append(name_to_name_id[arg_name])
        param_to_infer_id[arg_name] = len(param_to_infer_id)
    infer_name_ids = np.array(infer_name_ids, dtype=np.int64)

    return tokens_arr, name_arrs, infer_name_ids, type_seqs_arr, param_to_infer_id
