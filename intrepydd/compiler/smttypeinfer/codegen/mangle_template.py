import {CPP_MODULE_NAME}
import {PYTHON_MODULE_NAME}
import numpy as np
from numba.typed.typeddict import Dict
from itertools import product


class UnsupportedType(Exception):
    pass


def mangle_array_type_name(x: np.ndarray):
    name = dtype_names.get(x.dtype.type, None)
    if name is None:
        raise UnsupportedType
    return [f'a{{name}}{{x.ndim}}K']


def mangle_list_type_name(x: list):
    return [f'l{{name}}' for name in get_mangled_type_names(x[0])]


def mangle_tuple_type_name(x: tuple):
    name_segs = [[f't{{len(x)}}']]
    for y in x:
        name_segs.append(get_mangled_type_names(y))
    return [''.join(x) for x in product(*name_segs)]


def mangle_dict_type_name(x: 'dict | Dict'):
    ktype_names = get_mangled_type_names(next(iter(x)))
    vtype_names = get_mangled_type_names(next(iter(x.values())))
    return [f'd{{k}}{{v}}' for k, v in product(ktype_names, vtype_names)]


dtype_names = {{
    np.bool8: 'b',
    np.int64: 'i64',
    np.int32: 'i32',
    np.float64: 'f64',
    np.float32: 'f32'
}}


type_name_manglers = {{
    np.ndarray: mangle_array_type_name,
    list: mangle_list_type_name,
    tuple: mangle_tuple_type_name,
    dict: mangle_dict_type_name,
    Dict: mangle_dict_type_name,
    bool: lambda _: ['b'],
    np.bool8: lambda _: ['b'],
    int: lambda _: ['i32', 'i64'],
    np.int64: lambda _: ['i64'],
    np.int32: lambda _: ['i32'],
    float: lambda _: ['f32', 'f64'],
    np.float64: lambda _: ['f64'],
    np.float32: lambda _: ['f32'],
    str: lambda _: ['s']
}}


def get_mangled_type_names(x):
    mangler = type_name_manglers.get(type(x), None)
    if mangler is None:
        raise UnsupportedType
    return mangler(x)


def get_mangled_function_names(original_name: str, args):
    name_segs = [[f'_F{{len(original_name)}}{{original_name}}']]
    for arg in args:
        name_segs.append(get_mangled_type_names(arg))
    return [''.join(x) for x in product(*name_segs)]
