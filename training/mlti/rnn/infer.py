import numpy as np
from pathlib import Path
import torch
import pickle
from .dataset import MltiDataset
from .model import MltiModel
from ..common.pydd_types import type_tokens
from .preprocess import preprocess_pydd_src
from ..common import eval_pydd, compute_ml_combs


def infer(paths: list[Path], vocab_path: Path, model_path: Path,
          top_k: int, beam_size: int, max_type_length: int):
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    model: MltiModel = torch.load(model_path, map_location='cpu')
    model.eval()

    data = []
    info = []

    for path in paths:
        batch, func_param_to_infer_id, func_names = \
            preprocess_pydd_src(path.read_text(), vocab)

        func_type_seqs = []
        for tokens_arr, name_arrs, infer_name_ids, type_seqs_arr in batch:
            tokens_arr = torch.from_numpy(tokens_arr)
            name_arrs = [torch.from_numpy(name_arr) for name_arr in name_arrs]
            infer_name_ids = torch.from_numpy(infer_name_ids)
            data.append((tokens_arr, name_arrs, infer_name_ids, []))
            func_type_seqs.append(type_seqs_arr)

        info.append((path, func_names, func_param_to_infer_id, func_type_seqs))

    (tokens, names, subname_indices, infer_name_ids, _) = MltiDataset.collate(data)

    with torch.inference_mode():
        preds = model.infer(tokens, names, subname_indices, infer_name_ids,
                            beam_size, max_type_length)

    eval_pydd(info, preds, top_k)


def infer_freq(paths: list[Path], vocab_path: Path, top_k: int):
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
        freq_type_seqs: list[tuple[str]] = pickle.load(f)[:top_k]

    data = []
    info = []

    for path in paths:
        batch, func_param_to_infer_id, func_names = \
            preprocess_pydd_src(path.read_text(), vocab)

        func_type_seqs = []
        for tokens_arr, name_arrs, infer_name_ids, type_seqs_arr in batch:
            tokens_arr = torch.from_numpy(tokens_arr)
            name_arrs = [torch.from_numpy(name_arr) for name_arr in name_arrs]
            infer_name_ids = torch.from_numpy(infer_name_ids)
            data.append((tokens_arr, name_arrs, infer_name_ids, []))
            func_type_seqs.append(type_seqs_arr)

        info.append((path, func_names, func_param_to_infer_id, func_type_seqs))

    correct_all = []
    is_simple = []
    function_has_correct_all = []
    for file_path, func_names, func_param_to_infer_id, func_type_seqs in info:
        kernel_preds = []

        print(file_path.parent.name)

        for func_name, param_to_infer_id, type_seqs in \
                zip(func_names, func_param_to_infer_id, func_type_seqs):
            function_has_correct = True
            for param, infer_id in param_to_infer_id.items():
                type_seq_arr = type_seqs[infer_id]
                if type_seq_arr is None:
                    continue
                type_seq = tuple([type_tokens[x] for x in type_seq_arr])

                is_simple.append(len(type_seq) == 1)

                kernel_preds.append(freq_type_seqs)

                correct = []
                has_correct = False
                for pred_seq in freq_type_seqs:
                    if pred_seq == type_seq:
                        correct.append(1)
                        has_correct = True
                    else:
                        correct.append(0)
                correct_all.append(correct)
                function_has_correct = function_has_correct and has_correct

                # if not has_correct:
                #     print(type_seq)
                #     print('\t', pred_seqs)
                # print(f'file: {file_path}\tfunc: {func_name}\tparam: {param}')
                # print(f'    label: {type_seq}')
                # print(f'    preds:')
                # for pred_seq in freq_type_seqs:
                #     if pred_seq == type_seq:
                #         mark = '*'
                #     else:
                #         mark = ' '
                #     print(f'    {mark}   {pred_seq}')
                # print()
            function_has_correct_all.append(function_has_correct)

        compute_ml_combs(kernel_preds)

    correct_all = np.asarray(correct_all, dtype=np.float32)
    correct_all = np.cumsum(correct_all, axis=1)
    print(f'all    ({correct_all.shape[0]}): '
          f'{correct_all.mean(axis=0)}')

    is_simple = np.asarray(is_simple, np.bool8)
    print(f'simple ({is_simple.sum()}): '
          f'{correct_all[is_simple].mean(axis=0)}')
    print(f'nested ({(~is_simple).sum()}): '
          f'{correct_all[~is_simple].mean(axis=0)}')

    function_has_correct_all = np.asarray(function_has_correct_all, np.bool8)
    print(
        f'Function: {function_has_correct_all.sum()}/{len(function_has_correct_all)}'
        f' {function_has_correct_all.mean():.2%}')


def infer_gpt(paths: list[Path], vocab_path: Path, model_path: Path, top_k: int):
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    with open(model_path, 'rb') as f:
        name_preds: dict[tuple[str, str],
                         list[list[list[str]]]] = pickle.load(f)

    data = []
    info = []

    for path in paths:
        batch, func_param_to_infer_id, func_names = \
            preprocess_pydd_src(path.read_text(), vocab)

        func_type_seqs = []
        for tokens_arr, name_arrs, infer_name_ids, type_seqs_arr in batch:
            tokens_arr = torch.from_numpy(tokens_arr)
            name_arrs = [torch.from_numpy(name_arr) for name_arr in name_arrs]
            infer_name_ids = torch.from_numpy(infer_name_ids)
            data.append((tokens_arr, name_arrs, infer_name_ids, []))
            func_type_seqs.append(type_seqs_arr)

        info.append((path, func_names, func_param_to_infer_id, func_type_seqs))

    correct_all = []
    is_simple = []
    function_has_correct_all = []
    for file_path, func_names, func_param_to_infer_id, func_type_seqs in info:
        kernel_preds = []

        print(file_path.parent.name)

        name = file_path.parent.name, file_path.name
        param_type_seqs = name_preds[name]
        func_param_to_preds = {}
        param_id = 0

        import typed_ast.ast3 as ast
        tree = ast.parse(file_path.read_text())
        for fdef in tree.body:
            if not isinstance(fdef, ast.FunctionDef):
                continue
            for arg in fdef.args.args:
                func_param_to_preds[(fdef.name, arg.arg)] \
                    = param_type_seqs[param_id]
                param_id += 1
        assert param_id == len(param_type_seqs)

        for func_name, param_to_infer_id, type_seqs in \
                zip(func_names, func_param_to_infer_id, func_type_seqs):
            function_has_correct = True
            for param, infer_id in param_to_infer_id.items():

                pred_type_seqs = func_param_to_preds[(func_name, param)]
                assert len(pred_type_seqs) >= top_k
                pred_type_seqs = pred_type_seqs[:top_k]
                kernel_preds.append(pred_type_seqs)

                type_seq_arr = type_seqs[infer_id]
                if type_seq_arr is None:
                    continue
                type_seq = [type_tokens[x] for x in type_seq_arr]

                is_simple.append(len(type_seq) == 1)

                correct = []
                has_correct = False
                for pred_seq in pred_type_seqs:
                    if pred_seq == type_seq:
                        correct.append(1)
                        has_correct = True
                    else:
                        correct.append(0)
                correct_all.append(correct)
                function_has_correct = function_has_correct and has_correct

                # if not has_correct:
                #     print(type_seq)
                #     print('\t', pred_seqs)
                # print(f'file: {file_path}\tfunc: {func_name}\tparam: {param}')
                # print(f'    label: {type_seq}')
                # print(f'    preds:')
                # for pred_seq in pred_type_seqs:
                #     if pred_seq == type_seq:
                #         mark = '*'
                #     else:
                #         mark = ' '
                #     print(f'    {mark}   {pred_seq}')
                # print()
            function_has_correct_all.append(function_has_correct)

        compute_ml_combs(kernel_preds)

    correct_all = np.asarray(correct_all, dtype=np.float32)
    correct_all = np.cumsum(correct_all, axis=1)
    print(f'all    ({correct_all.shape[0]}): '
          f'{correct_all.mean(axis=0)}')

    is_simple = np.asarray(is_simple, np.bool8)
    print(f'simple ({is_simple.sum()}): '
          f'{correct_all[is_simple].mean(axis=0)}')
    print(f'nested ({(~is_simple).sum()}): '
          f'{correct_all[~is_simple].mean(axis=0)}')

    function_has_correct_all = np.asarray(function_has_correct_all, np.bool8)
    print(
        f'Function: {function_has_correct_all.sum()}/{len(function_has_correct_all)}'
        f' {function_has_correct_all.mean():.2%}')
