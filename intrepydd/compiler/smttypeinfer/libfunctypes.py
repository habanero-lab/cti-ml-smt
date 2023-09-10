from typing import Callable
from . import solver
from .solver import z3
from .typemanager import TypeManager


def handle_libfunc_callsite(funcname: str, arg_types: list[z3.ExprRef],
                            type_manager: TypeManager) -> z3.ExprRef:
    return libfunc_type_info[funcname](arg_types, type_manager)


libfunc_type_info: dict[str, Callable[[list[z3.ExprRef], TypeManager],
                                      z3.ExprRef]] = {}


def register_libfunc_type_info(*funcnames: str):

    def decorator(f):
        for funcname in funcnames:
            libfunc_type_info[funcname] = f
        return f

    return decorator


@register_libfunc_type_info('print')
def _(arg_types: list[z3.ExprRef], type_manager: TypeManager):
    return type_manager.create_none()


@register_libfunc_type_info('time')
def _(arg_types: list[z3.ExprRef], type_manager: TypeManager):
    assert len(arg_types) == 0
    return type_manager.create_array(type_manager.dtype.float64, 0)


@register_libfunc_type_info('len')
def _(arg_types: list[z3.ExprRef], type_manager: TypeManager):
    assert len(arg_types) == 1
    argty = arg_types[0]
    type_manager.add_constraint(
        z3.Or(type_manager.is_array(argty), type_manager.is_list(argty),
              type_manager.is_dict(argty), type_manager.is_heap(argty)))
    return type_manager.create_array(type_manager.dtype.int64, 0)


@register_libfunc_type_info('append')
def _(arg_types: list[z3.ExprRef], type_manager: TypeManager):
    assert len(arg_types) == 2
    lty = arg_types[0]
    ety = arg_types[1]
    type_manager.add_constraint(type_manager.is_list(lty))
    type_manager.add_constraint(type_manager.type.list_etype(lty) == ety)
    return type_manager.create_none()


@register_libfunc_type_info('fill')
def _(arg_types: list[z3.ExprRef], type_manager: TypeManager):
    assert len(arg_types) == 2
    aty = arg_types[0]
    ety = arg_types[1]
    type_manager.add_constraint(type_manager.is_array(aty))
    type_manager.add_constraint(type_manager.is_scalar(ety))
    type_manager.add_constraint(
        type_manager.type.array_dtype(aty) == type_manager.type.array_dtype(
            ety))
    return type_manager.create_none()


@register_libfunc_type_info('boolean')
def _(arg_types: list[z3.ExprRef], type_manager: TypeManager):
    assert len(arg_types) <= 1
    if arg_types:
        type_manager.add_constraint(type_manager.is_scalar(arg_types[0]))
    return type_manager.create_array(type_manager.dtype.bool, 0)


@register_libfunc_type_info('int32')
def _(arg_types: list[z3.ExprRef], type_manager: TypeManager):
    assert len(arg_types) <= 1
    if arg_types:
        type_manager.add_constraint(type_manager.is_scalar(arg_types[0]))
    return type_manager.create_array(type_manager.dtype.int32, 0)


@register_libfunc_type_info('int64')
def _(arg_types: list[z3.ExprRef], type_manager: TypeManager):
    assert len(arg_types) <= 1
    if arg_types:
        type_manager.add_constraint(type_manager.is_scalar(arg_types[0]))
    return type_manager.create_array(type_manager.dtype.int64, 0)


@register_libfunc_type_info('float32')
def _(arg_types: list[z3.ExprRef], type_manager: TypeManager):
    assert len(arg_types) <= 1
    if arg_types:
        type_manager.add_constraint(type_manager.is_scalar(arg_types[0]))
    return type_manager.create_array(type_manager.dtype.float32, 0)


@register_libfunc_type_info('float64')
def _(arg_types: list[z3.ExprRef], type_manager: TypeManager):
    assert len(arg_types) <= 1
    if arg_types:
        type_manager.add_constraint(type_manager.is_scalar(arg_types[0]))
    return type_manager.create_array(type_manager.dtype.float64, 0)


@register_libfunc_type_info('range')
def _(arg_types: list[z3.ExprRef], type_manager: TypeManager):
    assert 1 <= len(arg_types) <= 3
    for arg_ty in arg_types:
        type_manager.add_constraint(type_manager.is_int(arg_ty))
    return type_manager.create_array(type_manager.dtype.int32, 1)


@register_libfunc_type_info('zeros', 'empty', 'arange', 'numpy_random_rand')
def _(arg_types: list[z3.ExprRef], type_manager: TypeManager):
    assert 1 <= len(arg_types) <= 2
    sty = arg_types[0]
    ndim = type_manager.create_size()
    type_manager.add_constraint(
        z3.Or(
            z3.And(type_manager.is_tuple(sty),
                   type_manager.tuple_all(sty, type_manager.is_int),
                   ndim == type_manager.get_tuple_size(sty)),
            z3.And(type_manager.is_int(sty), ndim == 1)))
    if len(arg_types) == 2:
        dty = arg_types[1]
        type_manager.add_constraint(type_manager.is_scalar(dty))
        dtype = type_manager.type.array_dtype(dty)
    else:
        dtype = type_manager.dtype.float64
    return type_manager.create_array(dtype, ndim)


@register_libfunc_type_info('copy', 'empty_like')
def _(arg_types: list[z3.ExprRef], type_manager: TypeManager):
    assert len(arg_types) == 1
    return arg_types[0]


@register_libfunc_type_info('get_row', 'get_col')
def _(arg_types: list[z3.ExprRef], type_manager: TypeManager):
    assert len(arg_types) == 2
    aty, ity = arg_types
    type_manager.add_constraint(
        z3.And(type_manager.is_array(aty),
               type_manager.type.array_ndim(aty) == 2))
    type_manager.add_constraint(type_manager.is_int(ity))
    return type_manager.create_array(type_manager.type.array_dtype(aty), 1)


@register_libfunc_type_info('set_row', 'set_col')
def _(arg_types: list[z3.ExprRef], type_manager: TypeManager):
    assert len(arg_types) == 3
    aty, ity, vty = arg_types
    type_manager.add_constraint(
        z3.And(type_manager.is_array(aty),
               type_manager.type.array_ndim(aty) == 2))
    type_manager.add_constraint(type_manager.is_int(ity))
    type_manager.add_constraint(
        z3.And(
            type_manager.is_array(vty),
            z3.Or(
                type_manager.type.array_ndim(vty) == 0,
                type_manager.type.array_ndim(vty) == 1),
            type_manager.type.array_dtype(aty) ==
            type_manager.type.array_dtype(vty)))
    return type_manager.create_none()


@register_libfunc_type_info('sum_rows')
def _(arg_types: list[z3.ExprRef], type_manager: TypeManager):
    assert len(arg_types) == 2
    aty, ity = arg_types
    type_manager.add_constraint(
        z3.And(type_manager.is_array(aty),
               type_manager.type.array_ndim(aty) == 2))
    type_manager.add_constraint(
        z3.And(
            type_manager.is_array(ity),
            type_manager.type.array_ndim(ity) == 1,
            z3.Or(
                type_manager.type.array_dtype(ity) == type_manager.dtype.int32,
                type_manager.type.array_dtype(ity) ==
                type_manager.dtype.int64)))
    return type_manager.create_array(type_manager.type.array_dtype(aty), 1)


@register_libfunc_type_info('plus_eq', 'minus_eq')
def _(arg_types: list[z3.ExprRef], type_manager: TypeManager):
    assert len(arg_types) == 2
    rty = libfunc_type_info['add'](arg_types, type_manager)
    type_manager.add_constraint(arg_types[0] == rty)
    return type_manager.create_none()


@register_libfunc_type_info('shape')
def _(arg_types: list[z3.ExprRef], type_manager: TypeManager):
    assert 1 <= len(arg_types) <= 2
    aty = arg_types[0]
    if len(arg_types) == 1:
        ty = type_manager.create_type()
        int64_scalar = type_manager.create_array(type_manager.dtype.int64, 0)
        type_manager.add_constraint(
            z3.Or(
                z3.And(
                    type_manager.is_array(aty),
                    ty == type_manager.create_tuple(
                        etypes=int64_scalar,
                        size=type_manager.type.array_ndim(aty))),
                z3.And(
                    type_manager.is_sparsemat(aty),
                    ty == type_manager.create_tuple(etypes=int64_scalar,
                                                    size=2))))
        return ty
    else:
        type_manager.add_constraint(
            z3.Or(type_manager.is_array(aty), type_manager.is_sparsemat(aty)))
        ity = arg_types[1]
        type_manager.add_constraint(type_manager.is_int(ity))
        ty = type_manager.create_array(type_manager.dtype.int64, 0)
        return ty


@register_libfunc_type_info('sum', 'prod', 'min', 'max')
def _(arg_types: list[z3.ExprRef], type_manager: TypeManager):
    assert 1 <= len(arg_types) <= 2
    aty = arg_types[0]
    type_manager.add_constraint(type_manager.is_array(aty))
    if len(arg_types) == 1:
        ty = type_manager.create_array(type_manager.type.array_dtype(aty), 0)
    else:
        type_manager.add_constraint(type_manager.is_int(arg_types[1]))
        ndim = type_manager.type.array_ndim(aty)
        ty = type_manager.create_array(type_manager.type.array_dtype(aty),
                                       ndim - 1)
    return ty


@register_libfunc_type_info('where')
def _(arg_types: list[z3.ExprRef], type_manager: TypeManager):
    assert len(arg_types) == 1
    aty = arg_types[0]
    type_manager.add_constraint(
        z3.And(type_manager.is_array(aty),
               type_manager.type.array_ndim(aty) == 1))
    return type_manager.create_array(type_manager.dtype.int32, 1)


@register_libfunc_type_info('abs', 'minus')
def _(arg_types: list[z3.ExprRef], type_manager: TypeManager):
    assert len(arg_types) == 1
    aty = arg_types[0]
    type_manager.add_constraint(type_manager.is_array(aty))
    return aty


@register_libfunc_type_info('isnan', 'isinf', 'elemwise_not', 'logical_not')
def _(arg_types: list[z3.ExprRef], type_manager: TypeManager):
    assert len(arg_types) == 1
    aty = arg_types[0]
    type_manager.add_constraint(type_manager.is_array(aty))
    return type_manager.create_array(type_manager.dtype.bool,
                                     type_manager.type.array_ndim(aty))


@register_libfunc_type_info('sqrt', 'exp', 'cos', 'sin', 'tan', 'acos', 'asin',
                            'atan')
def _(arg_types: list[z3.ExprRef], type_manager: TypeManager):
    assert len(arg_types) == 1
    aty = arg_types[0]
    type_manager.add_constraint(type_manager.is_array(aty))
    return type_manager.create_array(type_manager.dtype.float64,
                                     type_manager.type.array_ndim(aty))


@register_libfunc_type_info('add', 'sub', 'mul', 'maximum', 'minimum', 'floor_div')
def _(arg_types: list[z3.ExprRef], type_manager: TypeManager):
    assert len(arg_types) == 2
    xty, yty = arg_types
    type_manager.add_constraint(type_manager.is_array(xty))
    type_manager.add_constraint(type_manager.is_array(yty))
    xndim = type_manager.type.array_ndim(xty)
    yndim = type_manager.type.array_ndim(yty)
    xdt = type_manager.type.array_dtype(xty)
    ydt = type_manager.type.array_dtype(yty)
    ndim = z3.If(xndim > yndim, xndim, yndim)
    dtype = _dtype_coercion(xdt, ydt, type_manager)
    return type_manager.create_array(dtype, ndim)


def _dtype_coercion(xdt: z3.ExprRef, ydt: z3.ExprRef, type_manager: TypeManager):
    dt = type_manager.create_dtype()
    for x, y, z in _dtype_coercion_rules:
        xx = getattr(type_manager.dtype, x)
        yy = getattr(type_manager.dtype, y)
        zz = getattr(type_manager.dtype, z)
        type_manager.add_constraint(
            z3.Implies(z3.And(xdt == xx, ydt == yy), dt == zz))
    return dt


_dtype_coercion_rules: list[tuple[str, str, str]] = [
    ('bool', 'bool', 'bool'),
    ('bool', 'int32', 'int32'),
    ('bool', 'int64', 'int64'),
    ('bool', 'float32', 'float32'),
    ('bool', 'float64', 'float64'),
    ('int32', 'bool', 'int32'),
    ('int32', 'int32', 'int32'),
    ('int32', 'int64', 'int64'),
    ('int32', 'float32', 'float64'),
    ('int32', 'float64', 'float64'),
    ('int64', 'bool', 'int64'),
    ('int64', 'int32', 'int64'),
    ('int64', 'int64', 'int64'),
    ('int64', 'float32', 'float64'),
    ('int64', 'float64', 'float64'),
    ('float32', 'bool', 'float32'),
    ('float32', 'int32', 'float64'),
    ('float32', 'int64', 'float64'),
    ('float32', 'float32', 'float32'),
    ('float32', 'float64', 'float64'),
    ('float64', 'bool', 'float64'),
    ('float64', 'int32', 'float64'),
    ('float64', 'int64', 'float64'),
    ('float64', 'float32', 'float64'),
    ('float64', 'float64', 'float64'),
]


@register_libfunc_type_info('pow', 'div')
def _(arg_types: list[z3.ExprRef], type_manager: TypeManager):
    assert len(arg_types) == 2
    xty, yty = arg_types
    type_manager.add_constraint(type_manager.is_array(xty))
    type_manager.add_constraint(type_manager.is_array(yty))
    xndim = type_manager.type.array_ndim(xty)
    yndim = type_manager.type.array_ndim(yty)
    ndim = z3.If(xndim > yndim, xndim, yndim)
    return type_manager.create_array(type_manager.dtype.float64, ndim)


@register_libfunc_type_info('log')
def _(arg_types: list[z3.ExprRef], type_manager: TypeManager):
    if len(arg_types) == 1:
        aty = arg_types[0]
        type_manager.add_constraint(type_manager.is_array(aty))
        return type_manager.create_array(type_manager.dtype.float64,
                                         type_manager.type.array_ndim(aty))
    assert len(arg_types) == 2
    xty, yty = arg_types
    type_manager.add_constraint(type_manager.is_array(xty))
    type_manager.add_constraint(type_manager.is_array(yty))
    xndim = type_manager.type.array_ndim(xty)
    yndim = type_manager.type.array_ndim(yty)
    ndim = z3.If(xndim > yndim, xndim, yndim)
    return type_manager.create_array(type_manager.dtype.float64, ndim)


@register_libfunc_type_info('eq', 'neq', 'lt', 'gt', 'le', 'ge', 'logical_and',
                            'logical_or', 'logical_xor')
def _(arg_types: list[z3.ExprRef], type_manager: TypeManager):
    assert len(arg_types) == 2
    xty, yty = arg_types
    type_manager.add_constraint(type_manager.is_array(xty))
    type_manager.add_constraint(type_manager.is_array(yty))
    xndim = type_manager.type.array_ndim(xty)
    yndim = type_manager.type.array_ndim(yty)
    ndim = z3.If(xndim > yndim, xndim, yndim)
    return type_manager.create_array(type_manager.dtype.bool, ndim)


@register_libfunc_type_info('transpose')
def _(arg_types: list[z3.ExprRef], type_manager: TypeManager):
    assert len(arg_types) == 1
    aty = arg_types[0]
    type_manager.add_constraint(
        z3.And(type_manager.is_array(aty),
               type_manager.type.array_ndim(aty) == 2))
    return aty


@register_libfunc_type_info('innerprod')
def _(arg_types: list[z3.ExprRef], type_manager: TypeManager):
    assert len(arg_types) == 2
    for aty in arg_types:
        type_manager.add_constraint(
            z3.And(
                type_manager.is_array(aty),
                type_manager.type.array_dtype(aty) ==
                type_manager.dtype.float64,
                type_manager.type.array_ndim(aty) == 1))
    return type_manager.create_array(type_manager.dtype.float64, 0)


@register_libfunc_type_info('matmult')
def _(arg_types: list[z3.ExprRef], type_manager: TypeManager):
    assert len(arg_types) == 2
    ndims = []
    for aty in arg_types:
        ndim = type_manager.type.array_ndim(aty)
        ndims.append(ndim)
        type_manager.add_constraint(
            z3.And(
                type_manager.is_array(aty),
                type_manager.type.array_dtype(aty) ==
                type_manager.dtype.float64, z3.Or(ndim == 1, ndim == 2)))
    ndim = z3.If(ndims[0] < ndims[1], ndims[0], ndims[1])
    return type_manager.create_array(type_manager.dtype.float64, ndim)


@register_libfunc_type_info('heapinit_empty')
def _(arg_types: list[z3.ExprRef], type_manager: TypeManager):
    assert not arg_types
    return type_manager.create_heap()


@register_libfunc_type_info('heapinit')
def _(arg_types: list[z3.ExprRef], type_manager: TypeManager):
    assert len(arg_types) == 2
    return type_manager.create_heap(arg_types[0], arg_types[1])


@register_libfunc_type_info('heappush')
def _(arg_types: list[z3.ExprRef], type_manager: TypeManager):
    assert len(arg_types) == 3
    hty, kty, vty = arg_types
    type_manager.add_constraint(hty == type_manager.create_heap(kty, vty))
    return type_manager.create_none()


@register_libfunc_type_info('heappop')
def _(arg_types: list[z3.ExprRef], type_manager: TypeManager):
    assert len(arg_types) == 1
    hty = arg_types[0]
    type_manager.add_constraint(type_manager.is_heap(hty))
    return type_manager.create_none()


@register_libfunc_type_info('heappeek_key')
def _(arg_types: list[z3.ExprRef], type_manager: TypeManager):
    assert len(arg_types) == 1
    hty = arg_types[0]
    type_manager.add_constraint(type_manager.is_heap(hty))
    return type_manager.type.heap_ktype(hty)


@register_libfunc_type_info('heappeek_val')
def _(arg_types: list[z3.ExprRef], type_manager: TypeManager):
    assert len(arg_types) == 1
    hty = arg_types[0]
    type_manager.add_constraint(type_manager.is_heap(hty))
    return type_manager.type.heap_vtype(hty)


@register_libfunc_type_info('heapsize')
def _(arg_types: list[z3.ExprRef], type_manager: TypeManager):
    assert len(arg_types) == 1
    hty = arg_types[0]
    type_manager.add_constraint(type_manager.is_heap(hty))
    return type_manager.create_array(type_manager.dtype.int32, 0)


@register_libfunc_type_info('heap_get_key')
def _(arg_types: list[z3.ExprRef], type_manager: TypeManager):
    assert len(arg_types) == 2
    hty, ity = arg_types
    type_manager.add_constraint(type_manager.is_heap(hty))
    type_manager.add_constraint(type_manager.is_int(ity))
    return type_manager.type.heap_ktype(hty)


@register_libfunc_type_info('empty_spm')
def _(arg_types: list[z3.ExprRef], type_manager: TypeManager):
    assert len(arg_types) == 2
    type_manager.add_constraint(type_manager.is_int(arg_types[0]))
    type_manager.add_constraint(type_manager.is_int(arg_types[1]))
    return type_manager.create_sparsemat(type_manager.dtype.float64)


@register_libfunc_type_info('csr_to_spm')
def _(arg_types: list[z3.ExprRef], type_manager: TypeManager):
    assert len(arg_types) == 4
    vty, cty, ity, ncty = arg_types
    type_manager.add_constraint(
        z3.And(
            type_manager.is_array(vty),
            type_manager.type.array_ndim(vty) == 1,
            type_manager.type.array_dtype(vty) == type_manager.dtype.float64))
    type_manager.add_constraint(
        z3.And(type_manager.is_array(cty),
               type_manager.type.array_ndim(cty) == 1,
               type_manager.type.array_dtype(cty) == type_manager.dtype.int32))
    type_manager.add_constraint(
        z3.And(type_manager.is_array(ity),
               type_manager.type.array_ndim(ity) == 1,
               type_manager.type.array_dtype(ity) == type_manager.dtype.int32))
    type_manager.add_constraint(type_manager.is_int(ncty))
    return type_manager.create_sparsemat(type_manager.dtype.float64)


@register_libfunc_type_info('spm_add', 'spm_mul')
def _(arg_types: list[z3.ExprRef], type_manager: TypeManager):
    assert len(arg_types) == 2
    xty, yty = arg_types
    type_manager.add_constraint(type_manager.is_sparsemat(xty))
    type_manager.add_constraint(
        z3.Or(
            type_manager.is_sparsemat(yty),
            z3.And(
                type_manager.is_array(yty),
                z3.Or(
                    type_manager.type.array_ndim(yty) == 0,
                    type_manager.type.array_ndim(yty) == 2,
                ))))
    return type_manager.create_sparsemat(type_manager.dtype.float64)


@register_libfunc_type_info('sparse_diags', 'spm_diags')
def _(arg_types: list[z3.ExprRef], type_manager: TypeManager):
    assert len(arg_types) == 1
    aty = arg_types[0]
    type_manager.add_constraint(
        z3.And(type_manager.is_array(aty),
               type_manager.type.array_ndim(aty) == 1))
    return type_manager.create_sparsemat(type_manager.dtype.float64)


@register_libfunc_type_info('spmm')
def _(arg_types: list[z3.ExprRef], type_manager: TypeManager):
    assert len(arg_types) == 2
    type_manager.add_constraint(type_manager.is_sparsemat(arg_types[0]))
    type_manager.add_constraint(type_manager.is_sparsemat(arg_types[1]))
    return type_manager.create_sparsemat(type_manager.dtype.float64)


@register_libfunc_type_info('spmm_dense')
def _(arg_types: list[z3.ExprRef], type_manager: TypeManager):
    assert len(arg_types) == 2
    xty, yty = arg_types

    def is_2d_array(ty: z3.ExprRef):
        return z3.And(
            type_manager.is_array(ty),
            type_manager.type.array_ndim(ty) == 2,
        )

    is_sparsemat = type_manager.is_sparsemat

    type_manager.add_constraint(
        z3.Or(z3.And(is_sparsemat(xty), is_sparsemat(yty)),
              z3.And(is_sparsemat(xty), is_2d_array(yty)),
              z3.And(is_2d_array(xty), is_sparsemat(yty))))

    return type_manager.create_array(type_manager.dtype.float64, 2)


@register_libfunc_type_info('spmv')
def _(arg_types: list[z3.ExprRef], type_manager: TypeManager):
    assert len(arg_types) == 2
    mty, vty = arg_types
    type_manager.add_constraint(type_manager.is_sparsemat(mty))
    type_manager.add_constraint(
        z3.And(
            type_manager.is_array(vty),
            type_manager.type.array_ndim(vty) == 1,
        ))
    return type_manager.create_array(type_manager.dtype.float64, 1)
