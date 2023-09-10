from typing import Optional, Sequence
import ast
from pathlib import Path
import pickle
import os
import numpy as np
import logging
import multiprocessing as mp
import tqdm
import transformers
import asttokens
import asttokens.util
import strip_hints
import token
import autopep8
import traceback
from ..common.pydd_types import dataset_type_to_seq, type_vocab, annotation_type_to_seq


def preprocess_dataset(raw_dataset_path: Path,
                       encoder_name: str,
                       strip_hints_and_comments: bool,
                       split_ratios: Sequence[float],
                       dataset_path: Path,
                       seed_sequence: np.random.SeedSequence,
                       num_processes: Optional[int] = None):

    logging.info('Loading the raw dataset')

    raw_data: list[tuple[str, dict[str, Optional[tuple]]]] = []
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

    if num_processes is None:
        num_processes = os.cpu_count() or 1

    logging.info('Generating training set')
    train_data = _generate_dataset(
        train_raw_data, encoder_name, strip_hints_and_comments, num_processes)

    logging.info('Generating validation set')
    val_data = _generate_dataset(
        val_raw_data, encoder_name, strip_hints_and_comments, num_processes)

    logging.info('Generating test set')
    test_data = _generate_dataset(
        test_raw_data, encoder_name, strip_hints_and_comments, num_processes)

    dataset_path.parent.mkdir(exist_ok=True)
    with open(dataset_path, 'wb') as f:
        pickle.dump((train_data, val_data, test_data), f)


def strip_type_hints(src: str):
    return strip_hints.strip_string_to_string(src, to_empty=True, strip_nl=True)


def strip_comments(src: str):
    ast_tokens = asttokens.ASTTokens(src, parse=True)
    replacements = []
    for t in ast_tokens.tokens:
        if t.type == token.COMMENT or t.type == token.TYPE_COMMENT:
            replacements.append((t.startpos, t.endpos, ''))
    for node in asttokens.util.walk(ast_tokens.tree):
        if isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Str):
                startpos, endpos = ast_tokens.get_text_range(node)
                replacements.append((startpos, endpos, ''))
    return asttokens.util.replace(src, replacements)


def _generate_dataset(raw_data, encoder_name: str, strip_hints_and_comments: bool,
                      num_processes: int):
    input_queue = mp.Queue()
    output_queue = mp.Queue()
    workers = [
        mp.Process(target=_preprocess_dataset_worker_fn,
                   args=(encoder_name, strip_hints_and_comments,
                         input_queue, output_queue),
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
    input_queue.close()

    return data


def _preprocess_dataset_worker_fn(encoder_name: str, strip_hints_and_comments: bool,
                                  input_queue: mp.Queue, output_queue: mp.Queue):
    tokenizer = transformers.AutoTokenizer.from_pretrained(encoder_name)

    while True:
        task = input_queue.get()
        if task is None:
            input_queue.put(None)
            break

        src, arg_types = task
        ret = _preprocess_dataset_fn(
            src, arg_types, strip_hints_and_comments, tokenizer)
        output_queue.put(ret)


def _preprocess_dataset_fn(src: str,
                           arg_types: dict[str, Optional[tuple]],
                           strip_hints_and_comments: bool,
                           tokenizer: transformers.PreTrainedTokenizerBase):
    arg_type_seqs: dict[str, list[str]] = {}
    for arg_name, typ in arg_types.items():
        seq = dataset_type_to_seq(typ)
        if seq is None:
            continue
        arg_type_seqs[arg_name] = seq

    if not arg_type_seqs:
        return None

    try:
        if strip_hints_and_comments:
            src = autopep8.fix_code(strip_type_hints(strip_comments(src)))
        ast_tokens = asttokens.ASTTokens(src, parse=True)
    except SyntaxError:
        traceback.print_exc()
        return None

    tree = ast_tokens.tree
    fdef = tree.body[0]
    assert isinstance(fdef, ast.FunctionDef)

    arg_offsets: dict[str, int] = {}
    for arg in fdef.args.args:
        arg_name = arg.arg
        startpos, endpos = ast_tokens.get_text_range(arg)
        assert endpos > 0
        arg_offsets[arg_name] = startpos

    batch_encoding = tokenizer(src, truncation=True)
    encoding = batch_encoding[0]

    type_seqs_arr = []
    token_ids = []
    for arg_name, seq in arg_type_seqs.items():
        seq_arr = np.array([type_vocab[s] for s in seq], dtype=np.int64)
        type_seqs_arr.append(seq_arr)
        offset = arg_offsets[arg_name]
        token_id = encoding.char_to_token(offset)
        if token_id is None:  # Truncated
            return None
        token_ids.append(token_id)
    token_ids = np.array(token_ids, dtype=np.int64)

    return batch_encoding, token_ids, type_seqs_arr


def preprocess_pydd_src(src: str,
                        tokenizer: transformers.PreTrainedTokenizerBase,
                        strip_hints_and_comments: bool = True):
    src = src.replace('pfor', 'for')

    func_names: list[str] = []
    func_param_to_infer_id: list[dict[str, int]] = []
    batch: list[tuple[transformers.BatchEncoding,
                      np.ndarray, list[Optional[np.ndarray]]]] = []

    ast_tokens = asttokens.ASTTokens(src, parse=True)
    for fdef in ast_tokens.tree.body:
        if not isinstance(fdef, ast.FunctionDef):
            continue
        func_names.append(fdef.name)
        func_src = ast_tokens.get_text(fdef)
        batch_encoding, token_ids, type_seqs_arr, param_to_infer_id = \
            _preprocess_pydd_func_src(
                func_src, tokenizer, strip_hints_and_comments)
        func_param_to_infer_id.append(param_to_infer_id)
        batch.append((batch_encoding, token_ids, type_seqs_arr))

    return batch, func_param_to_infer_id, func_names


def _preprocess_pydd_func_src(src: str,
                              tokenizer: transformers.PreTrainedTokenizerBase,
                              strip_hints_and_comments: bool):
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

    if strip_hints_and_comments:
        src = autopep8.fix_code(strip_type_hints(strip_comments(src)))
    ast_tokens = asttokens.ASTTokens(src, parse=True)
    fdef = ast_tokens.tree.body[0]

    arg_offsets: dict[str, int] = {}
    for arg in fdef.args.args:
        arg_name = arg.arg
        startpos, endpos = ast_tokens.get_text_range(arg)
        assert endpos > 0
        arg_offsets[arg_name] = startpos

    batch_encoding = tokenizer(src, truncation=True)
    encoding = batch_encoding[0]

    type_seqs_arr: list[Optional[np.ndarray]] = []
    token_ids = []
    param_to_infer_id: dict[str, int] = {}
    for arg_name, seq in arg_type_seqs.items():
        param_to_infer_id[arg_name] = len(token_ids)
        if seq is not None:
            seq_arr = np.array([type_vocab[s] for s in seq], dtype=np.int64)
        else:
            seq_arr = None
        type_seqs_arr.append(seq_arr)
        offset = arg_offsets[arg_name]
        token_id = encoding.char_to_token(offset)
        assert token_id is not None
        token_ids.append(token_id)

    token_ids = np.array(token_ids, dtype=np.int64)

    return batch_encoding, token_ids, type_seqs_arr, param_to_infer_id
