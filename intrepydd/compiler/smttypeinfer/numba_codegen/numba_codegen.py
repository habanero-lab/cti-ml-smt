from typing import Optional, Callable
import numpy as np
import os
import typed_ast.ast3 as ast
import typed_astunparse as astunparse
import inspect

import glb
import mytypes
from ..typeinfer import get_mangled_function_name
from . import spm


def gen_numba_lib(module: ast.AST):
    assert isinstance(module, ast.Module)

    numba_module_name = glb.get_module_name()
    output_dir = os.path.dirname(numba_module_name)

    rewriter = NumbaRewriter()
    rewriter.visit(module)

    src_segs = [
        'import numpy as np',
        'import heapq',
        'import numba',
        'from numba.pycc import CC',
        f'cc = CC("{numba_module_name}")',
        f'cc.output_dir = "{output_dir}"'
    ]

    for fdef in module.body:
        if not isinstance(fdef, ast.FunctionDef):
            continue
        func_name = fdef.name
        sig = rewriter.functions[func_name]
        src_segs.append(f'@numba.njit()')
        src_segs.append(f'@cc.export("{func_name}", "{sig}")')
        src_segs.append(astunparse.unparse(fdef))

    for mangled_name, (name, sig) in inserted_spm_functions.items():
        src_segs.append(f'@numba.njit()')
        src_segs.append(f'@cc.export("{mangled_name}", "{sig}")')
        fdef = ast.parse(inspect.getsource(getattr(spm, name))).body[0]
        fdef.name = mangled_name
        src_segs.append(astunparse.unparse(fdef))

    src_segs.append('cc.compile()')

    src = '\n'.join(src_segs)

    exec(src, {'__name__': numba_module_name})


class SkipFunction(Exception):
    pass


def mytype_to_numba_type(t: mytypes.MyType) -> str:
    dtype = mytype_to_numba_dtype(t)
    if dtype is not None:
        return dtype
    if isinstance(t, mytypes.VoidType):
        return 'none'
    if isinstance(t, mytypes.NpArray):
        dtype = mytype_to_numba_dtype(t.etype)
        assert dtype is not None
        ndim = t.ndim
        assert ndim > 0
        return f'{dtype}[{",".join([":"]*ndim)}]'
    if isinstance(t, mytypes.SparseMat):
        dtype = mytype_to_numba_dtype(t.etype)
        assert dtype is not None
        return f'Tuple([{dtype}[:],int32[:],int32[:],int32])'
    if isinstance(t, mytypes.ListType):
        etype = mytype_to_numba_type(t.etype)
        return f'ListType({etype})'
    if isinstance(t, mytypes.DictType):
        ktype = mytype_to_numba_type(t.key)
        vtype = mytype_to_numba_type(t.value)
        if vtype == 'none':
            raise SkipFunction
        return f'DictType({ktype},{vtype})'
    if isinstance(t, mytypes.Heap):
        ktype = mytype_to_numba_type(t.key)
        vtype = mytype_to_numba_type(t.val)
        return f'ListType(Tuple([{ktype},{vtype}]))'
    if isinstance(t, mytypes.TupleType):
        etypes = [mytype_to_numba_type(et) for et in t.etypes]
        return f'Tuple([{",".join(etypes)}])'
    assert False


def mytype_to_numba_dtype(t: mytypes.MyType) -> Optional[str]:
    if t == mytypes.bool:
        return 'boolean'
    if t == mytypes.int32:
        return 'int32'
    if t == mytypes.int64:
        return 'int64'
    if t == mytypes.float32:
        return 'float32'
    if t == mytypes.float64:
        return 'float64'
    return None


class NumbaRewriter(ast.NodeTransformer):

    def __init__(self):
        self.functions: dict[str, str] = {}

    def visit_Module(self, node: ast.Module):
        new_body = []
        for fdef in node.body:
            if not isinstance(fdef, ast.FunctionDef):
                new_body.append(fdef)
                continue
            try:
                arg_types = []
                for arg in fdef.args.args:
                    arg_types.append(mytype_to_numba_type(arg.type))
                ret_type = mytype_to_numba_type(fdef.ret_type)
                sig = f'{ret_type}({",".join(arg_types)})'
                self.functions[fdef.name] = sig
                new_body.append(fdef)
            except SkipFunction:
                pass
        node.body = new_body

        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        node.returns = None
        return self.generic_visit(node)

    def visit_arg(self, node: ast.arg):
        node.annotation = None
        return self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        if node.value is None:
            return None
        return ast.Assign(targets=[self.visit(node.target)], value=self.visit(node.value))

    def visit_Call(self, node: ast.Call):
        ty = node.type
        node = self.generic_visit(node)
        assert isinstance(node, ast.Call)
        func = node.func
        if isinstance(func, ast.Name):
            funcname = func.id
            if funcname in self.functions:
                pass
            elif funcname in libfunc_transform_info:
                assert not node.keywords
                node = libfunc_transform_info[funcname](
                    funcname, node.args, node)
            elif hasattr(np, funcname):
                node = create_np_lib_call(funcname, node.args)
            elif funcname in ('range', 'len', 'print'):
                pass
            else:
                raise Exception(f'Unsupported function: {funcname}')
        node.type = ty
        return node


def create_np_lib_call(func_name: str,
                       args: list[ast.expr],
                       **kwargs: ast.expr):
    keywords = [ast.keyword(arg=name, value=expr) for name, expr in kwargs]
    return ast.Call(
        func=ast.Attribute(
            value=ast.Name(id='np', ctx=ast.Load()),
            attr=func_name,
            ctx=ast.Load()
        ),
        args=args,
        keywords=keywords
    )


def create_dummy(t: mytypes.MyType) -> ast.expr:
    dtype = mytype_to_numba_dtype(t)
    if dtype is not None:
        if dtype == 'boolean':
            dtype = 'bool_'
        return create_np_lib_call(dtype, [ast.Num(0)])
    if isinstance(t, mytypes.VoidType):
        return ast.NameConstant(None)
    if isinstance(t, mytypes.NpArray):
        dtype = mytype_to_numba_dtype(t.etype)
        assert dtype is not None
        if dtype == 'boolean':
            dtype = 'bool_'
        ndim = t.ndim
        assert ndim > 0
        return create_np_lib_call(
            'empty',
            [],
            shape=ast.List(elts=[ast.Num(1)]*ndim, ctx=ast.Load()),
            dtype=ast.Subscript(
                value=ast.Name(id='np', ctx=ast.Load()),
                slice=ast.Index(ast.Name(id=dtype, ctx=ast.Load())),
                ctx=ast.Load()
            )
        )
    if isinstance(t, mytypes.SparseMat):
        dtype = mytype_to_numba_dtype(t.etype)
        assert dtype is not None
        if dtype == 'boolean':
            dtype = 'bool_'
        return ast.parse(
            f'(np.empty(1,dtype=np.{dtype}),'
            'np.empty(1,dtype=np.int32),'
            'np.empty(1,dtype=np.int32),'
            'np.int32(1))').body[0].value
    if isinstance(t, mytypes.ListType):
        e = create_dummy(t.etype)
        return ast.List(elts=[e], ctx=ast.Load())
    if isinstance(t, mytypes.DictType):
        k = create_dummy(t.key)
        v = create_dummy(t.value)
        return ast.Dict(keys=[k], values=[v])
    if isinstance(t, mytypes.Heap):
        k = create_dummy(t.key)
        v = create_dummy(t.val)
        return ast.List(elts=[ast.Tuple(elts=[k, v], ctx=ast.Load())], ctx=ast.Load())
    if isinstance(t, mytypes.TupleType):
        es = [create_dummy(et) for et in t.etypes]
        return ast.Tuple(elts=es, ctx=ast.Load())
    assert False


libfunc_transform_info: dict[str, Callable[[str, list[ast.expr], ast.Call], ast.expr]] \
    = {}


def register_libfunc_transform_info(*funcnames: str):

    def decorator(f):
        for funcname in funcnames:
            libfunc_transform_info[funcname] = f
        return f

    return decorator


lib_func_name_map = {
    'sub': 'subtract',
    'mul': 'multiply',
    # 'innerprod': 'inner',
    'div': 'divide',
    'where': 'flatnonzero',
    'eq': 'equal',
    'neq': 'not_equal',
    'lt': 'less',
    'gt': 'greater',
    'le': 'less_equal',
    'ge': 'greater_equal'
}


@register_libfunc_transform_info(*lib_func_name_map.keys())
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    return create_np_lib_call(lib_func_name_map[func_name], args)


# Numba limitation
@register_libfunc_transform_info('innerprod')
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    assert len(args) == 2
    return create_np_lib_call('sum', [
        ast.BinOp(op=ast.Mult(),
                  left=args[0],
                  right=args[1])
    ])


@register_libfunc_transform_info('matmult')
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    assert len(args) == 2
    return ast.BinOp(op=ast.MatMult(),
                     left=args[0],
                     right=args[1])


@register_libfunc_transform_info('bool', 'int32', 'int64', 'float32', 'float64')
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    if len(args) == 1:
        return create_np_lib_call(func_name, args)
    assert len(args) == 0
    return create_np_lib_call(func_name, [ast.Num(0)])


@register_libfunc_transform_info('zeros', 'empty', 'arange')
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    assert 1 <= len(args) <= 2
    if len(args) == 1:
        return create_np_lib_call(func_name, args)
    t = mytype_to_numba_dtype(args[1].type)
    assert t is not None
    if t == 'boolean':
        t = 'bool_'
    return create_np_lib_call(func_name, [
        args[0],
        ast.Attribute(
            value=ast.Name(id='np', ctx=ast.Load()),
            attr=t,
            ctx=ast.Load()
        )
    ])


@register_libfunc_transform_info('shape')
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    if len(args) == 1:
        return create_np_lib_call(
            'array',
            [create_np_lib_call('shape', args[:1])]
        )
    else:
        assert len(args) == 2
        return ast.Subscript(
            value=create_np_lib_call('shape', args[:1]),
            slice=ast.Index(value=args[1]),
            ctx=ast.Load()
        )


@register_libfunc_transform_info('get_row')
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    assert len(args) == 2
    return ast.Subscript(
        value=args[0],
        slice=ast.Index(value=args[1]),
        ctx=ast.Load()
    )


@register_libfunc_transform_info('get_col')
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    assert len(args) == 2
    return ast.Subscript(
        value=args[0],
        slice=ast.Tuple(
            elt=[ast.Slice(), ast.Index(value=args[1])],
            ctx=ast.Load()
        ),
        ctx=ast.Load()
    )


@register_libfunc_transform_info('set_row')
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    assert len(args) == 3
    return ast.Assign(
        targets=[ast.Subscript(
            value=args[0],
            slice=ast.Index(value=args[1]),
            ctx=ast.Store()
        )],
        value=args[2]
    )


@register_libfunc_transform_info('set_col')
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    assert len(args) == 3
    return ast.Assign(
        targets=[ast.Subscript(
            value=args[0],
            slice=ast.Tuple(
                elt=[ast.Slice(),
                     ast.Index(value=args[1])],
                ctx=ast.Load()
            ),
            ctx=ast.Store()
        )],
        value=args[2]
    )


@register_libfunc_transform_info('plus_eq')
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    assert len(args) == 2
    args[0].ctx = ast.Store()
    return ast.AugAssign(
        target=args[0],
        op=ast.Add(),
        value=args[1]
    )


@register_libfunc_transform_info('minus_eq')
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    assert len(args) == 2
    args[0].ctx = ast.Store()
    return ast.AugAssign(
        target=args[0],
        op=ast.Sub(),
        value=args[1]
    )


@register_libfunc_transform_info('append')
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    assert len(args) == 2
    return ast.Call(
        func=ast.Attribute(
            value=args[0],
            attr='append',
            ctx=ast.Load()
        ),
        args=[args[1]],
        keywords=[]
    )


@register_libfunc_transform_info('heapinit_empty')
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    assert not args
    heap_type = node.type
    assert isinstance(heap_type, mytypes.Heap)
    ktype, vtype = heap_type.key, heap_type.val
    k = create_dummy(ktype)
    v = create_dummy(vtype)
    t = ast.Tuple(elts=[k, v], ctx=ast.Load())
    lc: ast.ListComp = ast.parse('[d for _ in range(0)]').body[0].value
    lc.elt = t
    return lc


@register_libfunc_transform_info('heapinit')
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    assert len(args) == 2
    return ast.List(
        elts=[ast.Tuple(elts=args, ctx=ast.Load())],
        ctx=ast.Load()
    )


@register_libfunc_transform_info('heappush')
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    assert len(args) == 3
    return ast.Call(
        func=ast.Attribute(
            value=ast.Name(id='heapq', ctx=ast.Load()),
            attr='heappush',
            ctx=ast.Load()
        ),
        args=[
            args[0],
            ast.Tuple(elts=[args[1], args[2]], ctx=ast.Load())
        ],
        keywords=[]
    )


@register_libfunc_transform_info('heappop')
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    assert len(args) == 1
    return ast.Call(
        func=ast.Attribute(
            value=ast.Name(id='heapq', ctx=ast.Load()),
            attr='heappop',
            ctx=ast.Load()
        ),
        args=args,
        keywords=[]
    )


@register_libfunc_transform_info('heappeek_key')
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    assert len(args) == 1
    return ast.Subscript(
        value=ast.Subscript(
            value=args[0],
            slice=ast.Index(value=ast.Num(0)),
            ctx=ast.Load()
        ),
        slice=ast.Index(value=ast.Num(0)),
        ctx=ast.Load()
    )


@register_libfunc_transform_info('heappeek_val')
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    assert len(args) == 1
    return ast.Subscript(
        value=ast.Subscript(
            value=args[0],
            slice=ast.Index(value=ast.Num(0)),
            ctx=ast.Load()
        ),
        slice=ast.Index(value=ast.Num(1)),
        ctx=ast.Load()
    )


@register_libfunc_transform_info('heapsize')
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    assert len(args) == 1
    return ast.Call(
        func=ast.Name(id='len', ctx=ast.Load()),
        args=args,
        keywords=[]
    )


@register_libfunc_transform_info('heap_get_key')
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    assert len(args) == 2
    return ast.Subscript(
        value=ast.Subscript(
            value=args[0],
            slice=ast.Index(value=args[1]),
            ctx=ast.Load()
        ),
        slice=ast.Index(value=ast.Num(0)),
        ctx=ast.Load()
    )


inserted_spm_functions: dict[str, tuple[str, str]] = {}


def insert_spm_function(name: str,
                        arg_types: list[mytypes.MyType],
                        ret_type: mytypes.MyType):
    global inserted_spm_functions
    mangled_name = get_mangled_function_name(name, arg_types)
    if mangled_name not in inserted_spm_functions:
        nbts = [mytype_to_numba_type(t) for t in arg_types]
        rt = mytype_to_numba_type(ret_type)
        sig = f'{rt}({",".join(nbts)})'
        inserted_spm_functions[mangled_name] = name, sig
    return mangled_name


@register_libfunc_transform_info('csr_to_spm')
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    assert len(args) == 4
    return ast.Tuple(elts=args, ctx=ast.Load())


@register_libfunc_transform_info('spm_mul')
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    assert len(args) == 2
    t0 = args[0].type
    t1 = args[1].type
    if isinstance(t1, mytypes.NpArray):
        spm_func_name = 'spm_mul_dense'
    elif isinstance(t1, mytypes.FloatType):
        spm_func_name = 'spm_mul_scalar'
    else:
        raise Exception(f'Numba codegen: spm_mul({t0}, {t1}) not supported')
    mangled_name = insert_spm_function(spm_func_name, [t0, t1], t0)
    return ast.Call(
        func=ast.Name(id=mangled_name, ctx=ast.Load()),
        args=args,
        keywords=[]
    )


@register_libfunc_transform_info('spm_add')
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    assert len(args) == 2
    t0 = args[0].type
    t1 = args[1].type
    if isinstance(t1, mytypes.SparseMat):
        spm_func_name = 'spm_add_sparse'
    else:
        raise Exception(f'Numba codegen: spm_add({t0}, {t1}) not supported')
    mangled_name = insert_spm_function(spm_func_name, [t0, t1], t0)
    return ast.Call(
        func=ast.Name(id=mangled_name, ctx=ast.Load()),
        args=args,
        keywords=[]
    )


@register_libfunc_transform_info('spmm_dense')
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    assert len(args) == 2
    t0 = args[0].type
    t1 = args[1].type
    if isinstance(t0, mytypes.NpArray):
        spm_func_name = 'spmm_dense_ds'
    else:
        raise Exception(f'Numba codegen: spmm_dense({t0}, {t1}) not supported')
    mangled_name = insert_spm_function(spm_func_name,
                                       [t0, t1],
                                       mytypes.NpArray(mytypes.float64, 2))
    return ast.Call(
        func=ast.Name(id=mangled_name, ctx=ast.Load()),
        args=args,
        keywords=[]
    )


@register_libfunc_transform_info('spm_diags')
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    assert len(args) == 1
    mangled_name = insert_spm_function('spm_diags', [args[0].type],
                                       mytypes.SparseMat(mytypes.float64))
    return ast.Call(
        func=ast.Name(id=mangled_name, ctx=ast.Load()),
        args=args,
        keywords=[]
    )


@register_libfunc_transform_info('spmm')
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    assert len(args) == 2
    mangled_name = insert_spm_function('spmm', [args[0].type, args[1].type],
                                       mytypes.SparseMat(mytypes.float64))
    return ast.Call(
        func=ast.Name(id=mangled_name, ctx=ast.Load()),
        args=args,
        keywords=[]
    )


@register_libfunc_transform_info('spmv')
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    assert len(args) == 2
    mangled_name = insert_spm_function('spmv', [args[0].type, args[1].type],
                                       mytypes.NpArray(mytypes.float64, 1))
    return ast.Call(
        func=ast.Name(id=mangled_name, ctx=ast.Load()),
        args=args,
        keywords=[]
    )
