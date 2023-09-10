import mytypes
import glb
from glb import TypeError

# Record some info for cpp functions

# {func_name => [return_type, cpp_module]}
# '' means built-in functions

# Fields (default value):
# 1. return type (mytypes.void)
# 2. cpp module ('')
# 3. number of versions (1)
# 4. a bit vector: availability in -O0,1,2 (4)
#      __,__,__
#      O2 O1 O0
#      4: only available in -O2

float64_reduction = (mytypes.float64, 'reduction')
int64List_reduction = (mytypes.int64_list, 'reduction')
bool_reduction = (mytypes.bool, 'reduction')
# TODO
# means return type should be the element type pf the first argument
# arg0ElemType_reduction = (mytypes.float64, 'reduction')
# arg0Type_elemwise = (mytypes.float64_ndarray, 'elemwise')


def func_zeros_ret_type(arg_types):
    ndim = -1
    shape_ty = arg_types[0]
    if mytypes.is_list(shape_ty) \
       or mytypes.is_tuple(shape_ty):
        ndim = shape_ty.get_init_size()
    elif mytypes.is_int(shape_ty):
        ndim = 1
    else:
        raise TypeError('wrong arg type in zeros like function')

    # -- if data type not specified
    if len(arg_types) == 1:
        return mytypes.NpArray(mytypes.float64, ndim)
    # -- if data type specified
    elif len(arg_types) == 2:
        return mytypes.NpArray(arg_types[1], ndim)
    else:
        assert False


def func_zeros_2d_ret_type(arg_types):
    if len(arg_types) == 1:
        return mytypes.float64_2darray
    elif len(arg_types) == 2:
        return mytypes.NpArray(arg_types[1], 2)
    else:
        assert False


def func_sum_ret_type(arg_types):
    if len(arg_types) == 1:
        if isinstance(arg_types[0], mytypes.NpArray):
            return arg_types[0].get_elt_type()  # Is this right Jun? => Right
        else:
            raise TypeError('wrong arg type in sum like function')
    elif len(arg_types) == 2:
        if isinstance(arg_types[0], mytypes.NpArray) and mytypes.is_int(arg_types[1]):
            return one_dim_less(arg_types[0])
        else:
            raise TypeError('wrong arg type in sum like function')
    else:
        assert False


def array_or_scalar(arg_types, result_etype=None, promo_etype=None):
    if len(arg_types) == 1:
        arg_type = arg_types[0]
        if isinstance(arg_type, mytypes.NpArray):
            etype = result_etype if result_etype is not None else arg_type.etype
            return array_type_like(arg_type, dtype=etype)
        if not (mytypes.is_int(arg_type) or mytypes.is_float(arg_type) or isinstance(arg_types[0], mytypes.BoolType)):
            raise TypeError('wrong arg type')
        return result_etype if result_etype is not None else arg_type
    if len(arg_types) == 2:
        arg_type0 = arg_types[0]
        arg_type1 = arg_types[1]
        if result_etype is not None:
            etype = result_etype
        else:
            assert promo_etype is not None
            if isinstance(arg_type0, mytypes.NpArray) and isinstance(arg_type1, mytypes.NpArray):
                if arg_type0.ndim != arg_type1.ndim:
                    raise TypeError('wrong arg type')
                if arg_type0.etype == arg_type1.etype:
                    etype = arg_type0.etype
                else:
                    etype = promo_etype
            else:
                etype = promo_etype
        ndim = None
        if isinstance(arg_type0, mytypes.NpArray):
            ndim = arg_type0.ndim
        if isinstance(arg_type1, mytypes.NpArray):
            if ndim is None or ndim == -1:
                ndim = arg_type1.ndim
        if ndim is not None:
            return mytypes.NpArray(dtype=etype, ndim=ndim)
        return etype
    assert False


def array_type_like(array_type: mytypes.NpArray, dtype=None, ndim=None, layout=None):
    dtype = array_type.etype if dtype is None else dtype
    ndim = array_type.ndim if ndim is None else ndim
    layout = array_type.layout if layout is None else layout
    return mytypes.NpArray(dtype, ndim, layout)


def one_dim_less(array_type: mytypes.NpArray):
    ndim = array_type.ndim - 1
    if ndim < -1:
        ndim = -1
    return array_type_like(array_type, ndim=ndim)


def matmult_ret_type(arg_types):
    assert len(arg_types) == 2
    arg_type0 = arg_types[0]
    arg_type1 = arg_types[1]
    if not (isinstance(arg_type0, mytypes.NpArray) and
            isinstance(arg_type1, mytypes.NpArray) and
            arg_type0.etype == mytypes.double and
            arg_type1.etype == mytypes.double):
        raise TypeError('wrong arg type in matmult')
    dim0 = arg_type0.ndim
    dim1 = arg_type1.ndim
    if dim0 == 1 or dim1 == 1:
        return mytypes.float64_1darray
    if dim0 == -1 or dim1 == -1:
        return mytypes.float64_ndarray
    if dim0 == 2 and dim1 == 2:
        return mytypes.float64_2darray
    raise TypeError('wrong arg type in matmult')


def matmult1_ret_type(arg_types):
    assert len(arg_types) == 3
    arg_type0 = arg_types[0]
    arg_type1 = arg_types[1]
    arg_type2 = arg_types[2]
    if not (isinstance(arg_type0, mytypes.NpArray) and
            isinstance(arg_type1, mytypes.NpArray) and
            isinstance(arg_type2, mytypes.NpArray) and
            arg_type0.ndim == 2 and
            arg_type1.ndim == 2 and
            arg_type2.ndim == 2 and
            arg_type0.etype == mytypes.double and
            arg_type1.etype == mytypes.double and
            arg_type2.etype == mytypes.double):
        raise TypeError('wrong arg type in matmult1')
    return mytypes.int32


def func_scalar_ret_type(arg_types, ret_type):
    if len(arg_types) == 1:
        if not (mytypes.is_int(arg_types[0]) or
                mytypes.is_float(arg_types[0]) or
                isinstance(arg_types[0], mytypes.BoolType)):
            raise TypeError('wrong arg type in scalar function')
    else:
        assert len(arg_types) == 0
    return ret_type


def sum_rows_ret_type(arg_types):
    assert len(arg_types) == 2
    if not (mytypes.is_array(arg_types[0]) and
            arg_types[0].ndim == 2 and
            mytypes.is_array(arg_types[1]) and
            arg_types[1].ndim == 1 and
            mytypes.is_int(arg_types[1].dtype)):
        raise TypeError('wrong arg type in sum_rows')
    return array_type_like(arg_types[0], ndim=1)


def range_ret_type(arg_types):
    assert 1 <= len(arg_types) <= 3
    for arg_type in arg_types:
        if not mytypes.is_int(arg_type):
            raise TypeError('wrong arg type in range')
    return mytypes.int32_list


def get_row_ret_type(arg_types):
    assert len(arg_types) == 2
    if not (mytypes.is_array(arg_types[0]) and
            arg_types[0].ndim == 2 and
            mytypes.is_int(arg_types[1])):
        raise TypeError('wrong arg type in get_row like function')
    return one_dim_less(arg_types[0])


def set_row_ret_type(arg_types):
    assert len(arg_types) == 3
    if not (
        mytypes.is_array(arg_types[0]) and
        arg_types[0].ndim == 2 and
        mytypes.is_int(arg_types[1]) and
        (
            arg_types[0].etype == arg_types[2] or
            (
                mytypes.is_array(arg_types[2]) and
                arg_types[2].ndim == 1 and
                arg_types[0].etype == arg_types[2].etype
            )
        )
    ):
        raise TypeError('wrong arg type in set_row/col')
    return mytypes.void


def plus_eq_ret_type(arg_types):
    assert len(arg_types) == 2
    if not (mytypes.is_array(arg_types[0]) and
            (arg_types[0] == arg_types[1] or arg_types[0].etype == arg_types[1])):
        raise TypeError('wrong arg type in plus/minus_eq')
    return mytypes.void


def append_ret_type(arg_types):
    assert len(arg_types) == 2
    if not (mytypes.is_list(arg_types[0]) and
            arg_types[0].etype == arg_types[1]):
        raise TypeError('wrong arg type in append')
    return mytypes.void


def fill_ret_type(arg_types):
    assert len(arg_types) == 2
    if not (mytypes.is_array(arg_types[0]) and
            (arg_types[1] == mytypes.double or
            arg_types[0].etype == arg_types[1])):
        raise TypeError('wrong arg type in fill')
    return mytypes.void


def len_ret_type(arg_types):
    assert len(arg_types) == 1
    if not (mytypes.is_array(arg_types[0]) or
            mytypes.is_list(arg_types[0]) or
            isinstance(arg_types[0], mytypes.DictType) or
            isinstance(arg_types[0], mytypes.Heap)):
        raise TypeError('wrong arg type in len')
    return mytypes.int64


def copy_ret_type(arg_types):
    assert len(arg_types) == 1
    if not mytypes.is_array(arg_types[0]):
        raise TypeError('wrong arg type in copy')
    return arg_types[0]


def shape_ret_type(arg_types):
    if len(arg_types) == 1:
        if not mytypes.is_array(arg_types[0]):
            raise TypeError('wrong arg type in shape')
        return mytypes.int64_list
    else:
        assert len(arg_types) == 2
        if not (mytypes.is_array(arg_types[0]) and mytypes.is_int(arg_types[1])):
            raise TypeError('wrong arg type in shape')
        return mytypes.int64


def stride_ret_type(arg_types):
    assert len(arg_types) == 2
    if not (mytypes.is_array(arg_types[0]) and
            mytypes.is_int(arg_types[1])):
        raise TypeError('wrong arg type in stride')
    return mytypes.int64


def transpose_ret_type(arg_types):
    assert len(arg_types) == 1
    if not (mytypes.is_array(arg_types[0]) and arg_types[0].ndim == 2):
        raise TypeError('wrong arg type in transpose')
    return arg_types[0]


def transpose1_ret_type(arg_types):
    assert len(arg_types) == 2
    if not (mytypes.is_array(arg_types[0]) and arg_types[0] == arg_types[1]):
        raise TypeError('wrong arg type in transpose1')
    return mytypes.int32


def innerprod_ret_type(arg_types):
    assert len(arg_types) == 2
    if not (mytypes.is_array(arg_types[0]) and
            mytypes.is_array(arg_types[1]) and
            arg_types[0].ndim == 1 and
            arg_types[1].ndim == 1 and
            arg_types[0].etype == mytypes.double and
            arg_types[1].etype == mytypes.double):
        raise TypeError('wrong arg type in innerprod')
    return mytypes.double


funcinfo = {
    'print': (mytypes.void, '', 1, 5),
    'append': (append_ret_type, ''),
    'fill': (fill_ret_type, ''),
    'len': (len_ret_type, '', 1, 5),
    'boolean': [lambda arg_types: func_scalar_ret_type(arg_types, mytypes.bool), ''],
    'int32': [lambda arg_types: func_scalar_ret_type(arg_types, mytypes.int32), ''],
    'int64': [lambda arg_types: func_scalar_ret_type(arg_types, mytypes.int64), ''],
    'float32': [lambda arg_types: func_scalar_ret_type(arg_types, mytypes.float32), ''],
    'float64': [lambda arg_types: func_scalar_ret_type(arg_types, mytypes.float64), ''],
    'time': [mytypes.float64, ''],
    'range': [range_ret_type,
              'NpArray', 1, 5],
    'zeros': (func_zeros_ret_type,
              'NpArray'),
    'zeros_2d': (func_zeros_2d_ret_type,
                 'NpArray'),
    'empty': 'zeros',  # same as zeros
    'arange': 'empty',
    'copy': (copy_ret_type, 'NpArray'),
    'numpy_random_rand': 'zeros',
    'empty_2d': 'zeros_2d',
    'empty_like': (lambda arg_types: arg_types[0],
                   'NpArray'),
    'get_row': (get_row_ret_type, 'NpArray'),
    'get_row_ptr': 'get_row',
    'get_col': 'get_row',

    'set_row': (set_row_ret_type, 'NpArray'),
    'set_col': (set_row_ret_type, 'NpArray'),
    'sum_rows': (sum_rows_ret_type, 'NpArray'),
    'plus_eq': (plus_eq_ret_type, 'NpArray'),
    'minus_eq': (plus_eq_ret_type, 'NpArray'),
    'shape': [shape_ret_type, 'NpArray'],
    'stride': [stride_ret_type, 'NpArray'],
    'arraysub': [mytypes.void, 'NpArray'],
    'dsyrk': [mytypes.float64_2darray, 'lib'],
    'omp': [mytypes.void, 'omp'],
    # Reduction (TODO: add implementation versions)
    'sum': (lambda arg_types: func_sum_ret_type(arg_types), 'reduction'),
    'prod': 'sum',  # means same as above
    'min': 'sum',
    'max': 'sum',
    'argmin': int64List_reduction,
    'argmax': int64List_reduction,
    'any': bool_reduction,
    'all': 'any',
    'allclose': 'any',
    'where': [mytypes.int32_1darray, 'reduction'],
    'where1': [mytypes.int32_1darray, 'reduction'],
    # Element-wise unary (TODO: add implementation versions)
    'abs': (lambda arg_types: arg_types[0], 'elemwise'),
    'minus': 'abs',
    'isnan': (lambda arg_types: array_or_scalar(arg_types, result_etype=mytypes.bool), 'elemwise'),
    'isinf': 'isnan',
    'elemwise_not': 'isnan',
    'sqrt': (lambda arg_types: array_or_scalar(arg_types, result_etype=mytypes.float64), 'elemwise'),
    'exp': 'sqrt',
    'cos': 'sqrt',
    'sin': 'sqrt',
    'tan': 'sqrt',
    'acos': 'sqrt',
    'asin': 'sqrt',
    'atan': 'sqrt',
    # Element-wise binary (TODO: add implementation versions)
    'add': (lambda arg_types: array_or_scalar(arg_types, promo_etype=mytypes.float64), 'elemwise'),
    'sub': 'add',
    'mul': 'add',
    '_sub': (mytypes.void, 'elemwise'),
    'maximum': 'add',
    'pow': (lambda arg_types: array_or_scalar(arg_types, result_etype=mytypes.float64), 'elemwise'),
    'div': 'pow',
    'floor_div': 'pow',
    'log': 'sqrt',
    'eq': (lambda arg_types: array_or_scalar(arg_types, result_etype=mytypes.bool), 'elemwise'),
    'neq': 'eq',
    'lt': 'eq',
    'gt': 'eq',
    'le': 'eq',
    'ge': 'eq',
    'logical_and': 'eq',
    'logical_or': 'eq',
    'logical_xor': 'eq',
    'logical_not': 'isnan',
    'compatibility_check': [mytypes.int64_list, 'elemwise'],
    # Matrix operations
    'transpose': [transpose_ret_type, 'matrixop'],
    'innerprod': [innerprod_ret_type, 'matrixop'],
    'innerprod1': [mytypes.float64, 'matrixop'],
    'matmult1': [matmult1_ret_type, 'matrixop'],
    'transpose1': [transpose1_ret_type, 'matrixop'],
    'matmult': [matmult_ret_type, 'matrixop', 2],
    'syrk': [mytypes.float64_ndarray, 'matrixop', 2],
    'syr2k': 'syrk',
    'symm': 'syrk',
    'trmm': 'syrk',
    'lu': 'syrk',
    'ludcmp': 'syrk',
    'qr': 'syrk',
    'eig': 'syrk',
    'svd': 'syrk',
    'tril': 'syrk',
    'triu': 'syrk',
    'diag': 'syrk',
    'kron': 'syrk',
    'convolve': 'syrk',
    # Sparse matrix operations
    'empty_spm': [mytypes.float64_sparray,   # TODO: can be int and etc
                  'sparsemat'],
    'csr_to_spm': 'empty_spm',
    'arr_to_spm': 'empty_spm',
    'spm_to_csr': [mytypes.int32,
                   'sparsemat'],
    'spm_set_item': [mytypes.void,
                     'sparsemat'],
    'spm_set_item_unsafe': 'spm_set_item',
    'getval': [mytypes.float64,
               'sparsemat'],
    'getnnz': [mytypes.int32,
               'sparsemat'],
    'getcol': 'getnnz',
    'nnz': 'getnnz',
    'spm_add': [mytypes.float64_sparray,
                'sparsemat'],
    'spm_mul': 'spm_add',
    'spmm': 'spm_add',
    'spmv': [mytypes.float64_ndarray,
             'sparsemat'],
    'spmm_dense': [mytypes.float64_2darray,
                   'sparsemat'],
    'sparse_diags': 'empty_spm',
    'spm_diags': 'empty_spm',
    'spm_mask': [mytypes.float64_sparray,
                 'sparsemat'],
    'heapinit_empty': [mytypes.int32_int32_heap,
                       'heapq'],
    'heapinit': [mytypes.float32_heap,
                 'heapq'],
    'heappush': [mytypes.void,
                 'heapq'],
    'heappop': [mytypes.void,
                'heapq'],
    'heappeek_key': [mytypes.float32,
                     'heapq'],
    'heappeek_val': [mytypes.float32,
                     'heapq'],
    'heapsize': [mytypes.int32,
                 'heapq'],

    'heap_get_key': [mytypes.int32,
                     'heapq'],
    'randint': [mytypes.int32, ''],
    'hex': [mytypes.string, ''],
    'hextoi': [mytypes.int32, ''],
    'stoi': [mytypes.int32, ''],
    'strtol': [mytypes.int32, ''],
}

# {cpp_library => [compiler_flags, linker_flags]}
# FIXME: dont think this is in use anymore
packageinfo = {
    'NpArray': [],
    'reduction': [],
    'elemwise': [],
    'matrixop': [],
    'sparsemat': [],
    'heapq': [],
    'omp': ['-fopenmp', '-fopenmp'],
    'lib': ['-I /usr/include/openblas',
            '-lblas'],
}


# used_packages = {'NpArray', 'reduction', 'elemwise', 'matrixop', 'sparsemat', 'heapq', 'omp'}
used_packages = ['shared/NpArray', 'py/reduction', 'py/elemwise',
                 'py/matrixop', 'py/sparsemat', 'py/heapq']
# used_packages = {'NpArray'}


# global/built-in/library?
def is_global_func(func_name, module=''):
    # # TODO: to fix this
    return func_name in funcinfo
    # opt_bits = get_opt_level_bits(func_name)
    # is_available = 2**glb.args.opt_level & opt_bits
    # return func_name in funcinfo and is_available


def get_func_info(funcname, module='pydd'):
    '''
    A function is uniquely identiied by its module, name and signature.
    But due to the format of funcinfo table, we don't deal with argument
    signature here.
    '''
    if module == 'pydd':
        if funcname in funcinfo:
            v = funcinfo[funcname]
            while isinstance(v, str):
                v = funcinfo[v]
            return v

            # if isinstance(v, str):
            #     return get_func_info(v, module)
            # else:
            #     return v
        else:
            raise glb.UndefinedSymbolException(module, funcname)
    else:
        m_info = glb.get_imported_module_info(module)
        v = m_info.funcinfo[funcname]
        while isinstance(v, str):
            v = m_info.funcinfo[v]
        return v

# def get_ret_type(call_sig):
# #    print(vars(call_sig))
#     info = get_func_info(call_sig.funcname, call_sig.module)
#     if info and len(info) > 0:
#         ty = info[0]
#         if callable(ty):
#             return ty(call_sig.arg_types)
#         else:
#             return ty
#     else:
#         return mytypes.void


def get_type(name, module, arg_types=None):
    '''
    Get type of an external symbol via table lookup
    '''
    info = get_func_info(name, module)
    if info and len(info) > 0:
        ty = info[0]
        if callable(ty):
            return ty(arg_types)
        else:
            return ty
    else:
        return mytypes.void


def get_package(func_name):
    info = get_func_info(func_name)
    if info and len(info) > 1:
        return info[1]
    else:
        return ''


def get_version_count(func_name, module='pydd'):
    # # TODO: to fix this
    # return 1
    if module == 'std':
        return 1
    info = get_func_info(func_name, module)
    if info and len(info) > 2:
        return info[2]
    return 1


def get_opt_level_bits(func_name):
    info = get_func_info(func_name)
    if info and len(info) > 3:
        return info[3]
    return 4


def is_in_O0(func_name):
    info = get_func_info(func_name)
    if info and len(info) > 3:
        return info[3] & 1
    return False


def is_in_O1(func_name):
    info = get_func_info(func_name)
    if info and len(info) > 3:
        return info[3] & 2
    return False


def is_in_O2(func_name):
    info = get_func_info(func_name)
    if info and len(info) > 3:
        return info[3] & 4
    return True


def add_function_module(func_name):
    return
    global used_packages
    p = get_package(func_name)
    if p:
        used_packages.add(p)

# def register_lib_call(func_name):
#     global used_packages
#     used_packages.add(get_package(func_name))


def get_header_files():
    s = []

    # packages = ['shared/NpArray', 'py/reduction', 'py/elemwise',
    #             'py/matrixop', 'py/sparsemat', 'py/heapq', 'shared/omp']
    packages = used_packages

    # for p in used_packages:
    for p in packages:
        if p.startswith('shared/'):
            s.append(p+'.hpp')
        elif p.startswith('py/'):
            if glb.args.host == 'py':
                s.append(p+'.hpp')

    for p in glb.cpp_module.imports:
        if glb.is_imported_module_standard(p):
            s.append(p)
        else:
            s.append('%s/%s.hpp' % (p, p))

    return s


def get_compiler_flags():
    s = []
    # for p in used_packages:
    #     flags = packageinfo[p]
    #     if len(flags) > 0:
    #         s.append(flags[0])
    return s


def get_linker_flags():
    s = []

    if glb.args.blas:
        # Anaconda should already have these
        s += ['-liomp5', '-lmkl_rt', '-lmkl_core', '-lmkl_intel_thread',
              '-lmkl_intel_lp64', '-pthread']

    # for p in used_packages:
    #     flags = packageinfo[p]
    #     if len(flags) > 1:
    #         s.append(flags[1])
    return s
