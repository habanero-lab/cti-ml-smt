from typing import Optional, Sequence
import numpy as np
import torch
from pathlib import Path
from .pydd_types import type_tokens


BatchPreds = list[list[tuple[float, list[int]]]]


def eval_beam(batch_preds: BatchPreds,
              label_type_seqs: list[torch.Tensor],
              beam_size: int):
    labels = [tuple(seq.detach().cpu().numpy()) for seq in label_type_seqs]

    batch_correct = np.zeros(beam_size, dtype=np.int32)
    batch_basic_correct = np.zeros(beam_size, dtype=np.int32)
    batch_nested_correct = np.zeros(beam_size, dtype=np.int32)
    batch_basic_total = 0
    batch_nested_total = 0
    for preds, label in zip(batch_preds, labels):
        correct = [tuple(pred) == label for _, pred in preds]
        if len(correct) < beam_size:
            correct += [False] * (beam_size - len(correct))
        correct = np.array(correct, dtype=np.int32)
        correct = np.cumsum(correct)

        batch_correct += correct
        if len(label) == 1:
            batch_basic_total += 1
            batch_basic_correct += correct
        else:
            batch_nested_total += 1
            batch_nested_correct += correct

    return batch_correct, len(labels), \
        batch_basic_correct, batch_basic_total, \
        batch_nested_correct, batch_nested_total


def eval_pydd(info: list[tuple[Path, list[str], list[dict[str, int]], list[list[Optional[np.ndarray]]]]],
              preds: BatchPreds,
              top_k: int):
    correct_all = []
    is_simple = []
    function_has_correct_all = []
    preds_start = 0
    for file_path, func_names, func_param_to_infer_id, func_type_seqs in info:
        kernel_preds: list[list[list[str]]] = []
        print(file_path.parent.name)

        for func_name, param_to_infer_id, type_seqs in \
                zip(func_names, func_param_to_infer_id, func_type_seqs):
            function_has_correct = True
            for param, infer_id in param_to_infer_id.items():
                type_seq_arr = type_seqs[infer_id]
                if type_seq_arr is None:
                    continue
                type_seq = [type_tokens[x] for x in type_seq_arr]

                is_simple.append(len(type_seq) == 1)

                pred_seqs = []
                pred_logps = []
                pred = preds[preds_start + infer_id]
                for logp, pred_seq in pred[:top_k]:
                    pred_seq = [type_tokens[x] for x in pred_seq]
                    pred_seqs.append(pred_seq)
                    pred_logps.append(logp)
                kernel_preds.append(pred_seqs)

                correct = []
                has_correct = False
                for pred_seq in pred_seqs:
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
                # for pred_seq, pred_logp in zip(pred_seqs, pred_logps):
                #     if pred_seq == type_seq:
                #         mark = '*'
                #     else:
                #         mark = ' '
                #     print(f'    {mark}   {np.exp(pred_logp):.2%}\t{pred_seq}')
                # print()
            function_has_correct_all.append(function_has_correct)

            preds_start += len(type_seqs)

        compute_ml_combs(kernel_preds)

    assert preds_start == len(preds)

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


def compute_ml_combs(param_preds: Sequence[Sequence[Sequence[str]]]):
    num_types = np.empty(
        (len(param_preds), len(param_preds[0])), dtype=np.float64)
    for i, preds in enumerate(param_preds):
        for j, seq in enumerate(preds):
            x = 1
            for s in seq:
                if s in ('int', 'float'):
                    x *= 2
            num_types[i, j] = x
    num_types = np.cumsum(num_types, axis=1)
    num_combs = np.prod(num_types, axis=0)
    print(f'Num ML combs: {num_combs}')
