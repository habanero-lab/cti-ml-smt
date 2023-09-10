from typing import Optional
import ast

MAX_NDIM = 3

type_tokens = ['int', 'float', 'bool', 'list', 'array', 'dict', '1', '2', '3']

type_vocab = {t: i for i, t in enumerate(type_tokens)}


def dataset_type_to_seq(typ: Optional[tuple]) -> Optional[list[str]]:
    if typ is None:
        return None
    seq = []
    if _dataset_type_to_seq_rec(typ, seq):
        return seq
    return None


def _dataset_type_to_seq_rec(typ: tuple, seq: list[str]) -> bool:
    name = typ[0]
    if name == 'list':
        seq.append('list')
        return _dataset_type_to_seq_rec(typ[1], seq)
    elif name == 'dict':
        seq.append('dict')
        return _dataset_type_to_seq_rec(typ[1], seq)
    elif name == 'array':
        arr_seq = []
        arr_seq.append('array')
        dtype = typ[1]
        if dtype.startswith('int'):
            arr_seq.append('int')
        elif dtype.startswith('float'):
            arr_seq.append('float')
        elif name.startswith('bool'):
            arr_seq.append('bool')
        else:
            return False
        ndim = typ[2]
        if ndim > MAX_NDIM:
            return False
        if ndim == 0:
            seq.append(arr_seq[1])
        else:
            arr_seq.append(str(ndim))
            seq += arr_seq
    else:
        if name.startswith('int'):
            seq.append('int')
        elif name.startswith('float'):
            seq.append('float')
        elif name.startswith('bool'):
            seq.append('bool')
        else:
            return False
    return True


def annotation_type_to_seq(typ: Optional[ast.expr]) -> Optional[list[str]]:
    if typ is None:
        return None
    seq = []
    if _annotation_type_to_seq_rec(typ, seq):
        return seq
    return None


def _annotation_type_to_seq_rec(typ: ast.expr, seq: list[str]):
    if isinstance(typ, ast.Name):
        name = typ.id
        if name.startswith('int'):
            seq.append('int')
        elif name.startswith('float') or name.startswith('double'):
            seq.append('float')
        elif name.startswith('bool'):
            seq.append('bool')
        else:
            raise ValueError(f'Unexpected type annotation: {ast.unparse(typ)}')
        return True
    if isinstance(typ, ast.Call):
        func = typ.func
        if not isinstance(func, ast.Name):
            raise ValueError(f'Unexpected type annotation: {ast.unparse(typ)}')
        name = func.id
        if name == 'Array':
            seq.append('array')
            if not (1 <= len(typ.args) <= 2):
                raise ValueError(
                    f'Unexpected type annotation: {ast.unparse(typ)}')
            dtype = typ.args[0]
            if not isinstance(dtype, ast.Name):
                raise ValueError(
                    f'Unexpected type annotation: {ast.unparse(typ)}')
            dtype = dtype.id
            if dtype.startswith('int'):
                seq.append('int')
            elif dtype.startswith('float') or dtype.startswith('double'):
                seq.append('float')
            elif name.startswith('bool'):
                seq.append('bool')
            else:
                raise ValueError(f'Unexpected array dtype: {dtype}')
            if len(typ.args) > 1:
                ndim = typ.args[1]
                if not (isinstance(ndim, ast.Constant)
                        and isinstance(ndim.value, int)
                        and ndim.value <= MAX_NDIM and ndim.value > 0):
                    raise ValueError(
                        f'Unexpected array ndim: {ast.unparse(ndim)}')
                seq.append(str(ndim.value))
            else:
                seq.append('1')
            return True
        elif name == 'List':
            seq.append('list')
            if not len(typ.args) == 1:
                raise ValueError(
                    f'Unexpected type annotation: {ast.unparse(typ)}')
            return _annotation_type_to_seq_rec(typ.args[0], seq)
        elif name == 'Dict':
            seq.append('dict')
            if not len(typ.args) == 2:
                raise ValueError(
                    f'Unexpected type annotation: {ast.unparse(typ)}')
            key_type = typ.args[0]
            if not (isinstance(key_type, ast.Name)
                    and key_type.id.startswith('int')):
                raise ValueError(
                    f'Unexpected type annotation: {ast.unparse(typ)}')
            return _annotation_type_to_seq_rec(typ.args[1], seq)
        else:
            raise ValueError(f'Unexpected type annotation: {ast.unparse(typ)}')
    raise ValueError(f'Unexpected type annotation: {ast.unparse(typ)}')
