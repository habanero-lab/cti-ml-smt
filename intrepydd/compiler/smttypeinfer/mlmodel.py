from typing import Sequence, TYPE_CHECKING
import torch
from pathlib import Path
import pickle
import sys
import abc
import ast
import transformers

from .solver import z3
import glb
if TYPE_CHECKING:
    from .typeinfer import FunctionInfo
    from .typemanager import TypeManager

# fmt: off
sys.path.append(
    str((Path(__file__).parent.parent.parent.parent / 'training').resolve()))
import mlti.common
import mlti.rnn
import mlti.tf
# fmt: on


def get_ml_model() -> 'MLModel':
    model_name = glb.args.smtti_ml_model
    if model_name == 'rnn':
        return RNNModel()
    if model_name == 'tf':
        return TFModel()
    if model_name == 'chatgpt':
        return ChatGPTModel()
    if model_name == 'freq':
        return FreqModel()
    assert False


class MLModel(abc.ABC):

    @abc.abstractmethod
    def run_inference(self, src: str, type_manager: 'TypeManager') \
            -> tuple[list[list[z3.ExprRef]], list[dict[str, int]], list[str]]:
        ...

    def get_ml_constraints(self,
                           functions: dict[str, 'FunctionInfo'],
                           src: str,
                           type_manager: 'TypeManager') -> list[list[z3.ExprRef]]:
        infer_type_exprs, func_param_to_infer_id, func_names = \
            self.run_inference(src, type_manager)

        ml_constraints: list[list[z3.ExprRef]] = []
        preds_start = 0
        for func_name, param_to_infer_id in zip(func_names, func_param_to_infer_id):
            func_info = functions[func_name]
            for arg, param_type in zip(func_info.fdef.args.args, func_info.param_types):
                infer_id = param_to_infer_id[arg.arg]
                type_exprs = infer_type_exprs[preds_start + infer_id]
                ml_constraint = [param_type == type_expr
                                 for type_expr in type_exprs]
                ml_constraints.append(ml_constraint)
            preds_start += len(func_info.param_types)

        assert preds_start == len(infer_type_exprs)

        return ml_constraints


class RNNModel(MLModel):

    def __init__(self):
        with open(glb.args.smtti_vocab, 'rb') as f:
            self.vocab: mlti.rnn.preprocess.Vocab = pickle.load(f)
        self.model: mlti.rnn.model.MltiModel = torch.load(glb.args.smtti_model_path,
                                                          map_location='cpu')
        self.model.eval()

    def run_inference(self, src: str, type_manager: 'TypeManager') \
            -> tuple[list[list[z3.ExprRef]], list[dict[str, int]], list[str]]:
        batch, func_param_to_infer_id, func_names = \
            mlti.rnn.preprocess.preprocess_pydd_src(src, self.vocab)

        data = []
        for tokens_arr, name_arrs, infer_name_ids, _ in batch:
            tokens_arr = torch.from_numpy(tokens_arr)
            name_arrs = [torch.from_numpy(name_arr) for name_arr in name_arrs]
            infer_name_ids = torch.from_numpy(infer_name_ids)
            data.append((tokens_arr, name_arrs, infer_name_ids, []))

        tokens, names, subname_indices, infer_name_ids, _ = \
            mlti.rnn.dataset.MltiDataset.collate(data)

        with torch.inference_mode():
            output = self.model.infer(tokens, names, subname_indices, infer_name_ids,
                                      glb.args.smtti_beam_size, glb.args.smtti_max_type_length)

        infer_type_exprs = []
        for preds in output:
            type_exprs = []
            for _, type_ids_seq in preds[:glb.args.smtti_top_k]:
                type_seq = [mlti.common.pydd_types.type_tokens[i]
                            for i in type_ids_seq]
                type_expr, _ = _type_seq_to_type_expr(type_seq, type_manager)
                type_exprs.append(type_expr)
            infer_type_exprs.append(type_exprs)

        return infer_type_exprs, func_param_to_infer_id, func_names


class TFModel(MLModel):

    def __init__(self):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            glb.args.smtti_tf_encoder)
        self.model: mlti.tf.model.MltiModel = torch.load(glb.args.smtti_model_path,
                                                         map_location='cpu')
        self.model.eval()

    def run_inference(self, src: str, type_manager: 'TypeManager') \
            -> tuple[list[list[z3.ExprRef]], list[dict[str, int]], list[str]]:
        batch, func_param_to_infer_id, func_names = \
            mlti.tf.preprocess.preprocess_pydd_src(src, self.tokenizer)

        data = []
        for batch_encoding, token_ids, _ in batch:
            batch_encoding = batch_encoding.convert_to_tensors('pt')
            token_ids = torch.from_numpy(token_ids)
            data.append((batch_encoding, token_ids, []))

        input_encoding, token_ids, _ = \
            mlti.tf.dataset.MltiDataset.collate(data, self.tokenizer)

        with torch.inference_mode():
            output = self.model.infer(input_encoding, token_ids,
                                      glb.args.smtti_beam_size, glb.args.smtti_max_type_length)

        infer_type_exprs = []
        for preds in output:
            type_exprs = []
            for _, type_ids_seq in preds[:glb.args.smtti_top_k]:
                type_seq = [mlti.common.pydd_types.type_tokens[i]
                            for i in type_ids_seq]
                type_expr, _ = _type_seq_to_type_expr(type_seq, type_manager)
                type_exprs.append(type_expr)
            infer_type_exprs.append(type_exprs)

        return infer_type_exprs, func_param_to_infer_id, func_names


class ChatGPTModel(MLModel):

    def __init__(self):
        with open(glb.args.smtti_model_path, 'rb') as f:
            name_preds: dict[tuple[str, str],
                             list[list[list[str]]]] = pickle.load(f)
        path = Path(glb.args.file).resolve()
        name = path.parent.name, path.name
        self.param_type_seqs = name_preds[name]

    def run_inference(self, src: str, type_manager: 'TypeManager') \
            -> tuple[list[list[z3.ExprRef]], list[dict[str, int]], list[str]]:
        param_type_exprs = []
        for type_seqs in self.param_type_seqs:
            type_exprs = []
            param_type_exprs.append(type_exprs)
            for type_seq in type_seqs[:glb.args.smtti_top_k]:
                type_expr, _ = _type_seq_to_type_expr(type_seq, type_manager)
                type_exprs.append(type_expr)

        src = src.replace('pfor', 'for')
        func_names: list[str] = []
        func_param_to_infer_id: list[dict[str, int]] = []
        tree = ast.parse(src)
        for fdef in tree.body:
            if not isinstance(fdef, ast.FunctionDef):
                continue
            func_names.append(fdef.name)
            param_to_infer_id = {}
            for arg in fdef.args.args:
                param_to_infer_id[arg.arg] = len(param_to_infer_id)
            func_param_to_infer_id.append(param_to_infer_id)

        return param_type_exprs, func_param_to_infer_id, func_names


class FreqModel(MLModel):

    def __init__(self):
        with open(glb.args.smtti_vocab, 'rb') as f:
            _ = pickle.load(f)
            freq_type_seqs: list[Sequence[str]] = pickle.load(f)
        self.type_seqs = freq_type_seqs[:glb.args.smtti_top_k]

    def run_inference(self, src: str, type_manager: 'TypeManager') \
            -> tuple[list[list[z3.ExprRef]], list[dict[str, int]], list[str]]:
        type_exprs = []
        for type_seq in self.type_seqs:
            type_expr, _ = _type_seq_to_type_expr(type_seq, type_manager)
            type_exprs.append(type_expr)

        src = src.replace('pfor', 'for')
        func_names: list[str] = []
        func_param_to_infer_id: list[dict[str, int]] = []
        tree = ast.parse(src)
        num_params = 0
        for fdef in tree.body:
            if not isinstance(fdef, ast.FunctionDef):
                continue
            func_names.append(fdef.name)
            param_to_infer_id = {}
            for arg in fdef.args.args:
                param_to_infer_id[arg.arg] = len(param_to_infer_id)
            func_param_to_infer_id.append(param_to_infer_id)
            num_params += len(param_to_infer_id)

        infer_type_exprs = [type_exprs] * num_params

        return infer_type_exprs, func_param_to_infer_id, func_names


def _type_seq_to_type_expr(type_seq: Sequence[str], type_manager: 'TypeManager') \
        -> tuple[z3.ExprRef, Sequence[str]]:
    name = type_seq[0]
    if name == 'list':
        t, type_seq = _type_seq_to_type_expr(type_seq[1:], type_manager)
        return type_manager.create_list(t), type_seq
    if name == 'array':
        ndim = int(type_seq[2])
        dtype = type_manager.create_dtype()
        dtype_name = type_seq[1]
        if dtype_name == 'int':
            type_manager.add_constraint(z3.Or(
                dtype == type_manager.dtype.int32,
                dtype == type_manager.dtype.int64
            ))
        elif dtype_name == 'float':
            type_manager.add_constraint(z3.Or(
                dtype == type_manager.dtype.float32,
                dtype == type_manager.dtype.float64
            ))
        elif dtype_name == 'bool':
            type_manager.add_constraint(dtype == type_manager.dtype.bool)
        else:
            assert False
        return type_manager.create_array(dtype, ndim), type_seq[3:]
    if name == 'int':
        dtype = type_manager.create_dtype()
        type_manager.add_constraint(z3.Or(
            dtype == type_manager.dtype.int32,
            dtype == type_manager.dtype.int64
        ))
        return type_manager.create_array(dtype, 0), type_seq[1:]
    if name == 'float':
        dtype = type_manager.create_dtype()
        type_manager.add_constraint(z3.Or(
            dtype == type_manager.dtype.float32,
            dtype == type_manager.dtype.float64
        ))
        return type_manager.create_array(dtype, 0), type_seq[1:]
    if name == 'bool':
        return type_manager.create_array(type_manager.dtype.bool, 0), type_seq[1:]
    if name == 'dict':
        # kt, type_seq = _type_seq_to_type_expr(type_seq[1:], type_manager)
        # vt, type_seq = _type_seq_to_type_expr(type_seq, type_manager)
        kt = type_manager.create_array(ndim=0)
        vt, type_seq = _type_seq_to_type_expr(type_seq[1:], type_manager)
        return type_manager.create_dict(kt, vt), type_seq
    # if name == 'sparsemat':
    #     dtype = type_manager.create_dtype()
    #     dtype_name = type_seq[1]
    #     if dtype_name == 'int':
    #         type_manager.add_constraint(z3.Or(
    #             dtype == type_manager.dtype.int32,
    #             dtype == type_manager.dtype.int64
    #         ))
    #     elif dtype_name == 'float':
    #         type_manager.add_constraint(z3.Or(
    #             dtype == type_manager.dtype.float32,
    #             dtype == type_manager.dtype.float64
    #         ))
    #     elif dtype_name == 'bool':
    #         type_manager.add_constraint(dtype == type_manager.dtype.bool)
    #     else:
    #         assert False
    #     return type_manager.create_sparsemat(dtype), type_seq[2:]
    # if name == 'heap':
    #     kt, type_seq = _type_seq_to_type_expr(type_seq[1:], type_manager)
    #     vt, type_seq = _type_seq_to_type_expr(type_seq, type_manager)
    #     return type_manager.create_heap(kt, vt), type_seq
    # if name == 'tuple':
    #     t, type_seq = _type_seq_to_type_expr(type_seq[1:], type_manager)
    #     arity = int(type_seq[0])
    #     return type_manager.create_tuple(t, arity), type_seq[1:]
    assert False
