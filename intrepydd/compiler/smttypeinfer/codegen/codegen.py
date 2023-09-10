from typing import Callable
from pathlib import Path
import typed_ast.ast3 as ast
import typed_astunparse as astunparse
import numpy as np
import glb
from ..typeinfer import FunctionInfo


mangle_template = (Path(__file__).parent / "mangle_template.py").read_text()
dispatch_template = (Path(__file__).parent / "dispatch_template.py").read_text()
memoize_template = "    global {ORIGINAL_NAME}\n    {ORIGINAL_NAME} = __f\n"
verbose_template = (
    "    print(f'Compiled: {{__f != {PYTHON_MODULE_NAME}.{ORIGINAL_NAME}}}, '\n"
    "          f'{ORIGINAL_NAME} called with args: {{args}}')\n"
)


def gen_python_wrapper(functions: dict[str, FunctionInfo], original_module: ast.AST):
    assert isinstance(original_module, ast.Module)

    module_name = glb.get_module_name(original=True)
    python_wrapper_name = module_name + ".py"
    cpp_module_name = glb.get_module_name()
    python_module_name = module_name + "_py"

    gen_python_implementation(
        python_module_name + ".py", module_name, original_module, has_wrapper=True
    )

    mangle_src = mangle_template.format(
        CPP_MODULE_NAME=cpp_module_name, PYTHON_MODULE_NAME=python_module_name
    )

    with open(python_wrapper_name, "w") as f:
        f.write(mangle_src)
        f.write("\n")
        for func_name, func_info in functions.items():
            memoize_src = (
                memoize_template.format(ORIGINAL_NAME=func_name)
                if glb.args.smtti_wrapper_memoize
                else ""
            )
            verbose_src = (
                verbose_template.format(
                    PYTHON_MODULE_NAME=python_module_name, ORIGINAL_NAME=func_name
                )
                if glb.args.smtti_wrapper_verbose
                else ""
            )
            dispatch_src = dispatch_template.format(
                CPP_MODULE_NAME=cpp_module_name,
                PYTHON_MODULE_NAME=python_module_name,
                ORIGINAL_NAME=func_name,
                EXTRA=memoize_src + verbose_src,
            )
            f.write(dispatch_src)
            f.write("\n")


def gen_python_implementation(
    path: str, module_name: str, original_module: ast.Module, has_wrapper: bool
):
    rewriter = PythonRewriter(module_name, has_wrapper)
    rewriter.visit(original_module)

    prefix = [
        "import numpy as np",
        "import heapq",
        "from scipy.sparse import csr_matrix, diags",
    ]

    if has_wrapper:
        prefix.append(f"import {module_name}")

    with open(path, "w") as f:
        f.write("\n".join(prefix))
        f.write("\n")
        f.write(astunparse.unparse(original_module))


class PythonRewriter(ast.NodeTransformer):
    def __init__(self, module_name: str, has_wrapper: bool):
        self.module_name = module_name
        self.has_wrapper = has_wrapper
        self.functions: set[str] = set()

    def visit_Module(self, node: ast.Module):
        for fdef in node.body:
            if not isinstance(fdef, ast.FunctionDef):
                continue
            self.functions.add(fdef.name)

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
        return ast.Assign(
            targets=[self.visit(node.target)], value=self.visit(node.value)
        )

    def visit_Call(self, node: ast.Call):
        node = self.generic_visit(node)
        assert isinstance(node, ast.Call)
        func = node.func
        if isinstance(func, ast.Name):
            funcname = func.id
            if funcname in self.functions:
                if self.has_wrapper:
                    # Rewrite a call to a user-defined function to its wrapper function
                    node.func = ast.Attribute(
                        value=ast.Name(id=self.module_name, ctx=ast.Load()),
                        attr=funcname,
                        ctx=func.ctx,
                    )
            elif funcname in libfunc_transform_info:
                assert not node.keywords
                node = libfunc_transform_info[funcname](funcname, node.args, node)
            elif hasattr(np, funcname):
                node = create_np_lib_call(funcname, node.args)
            elif funcname in ("range", "len", "print"):
                pass
            else:
                raise Exception(f"Unsupported function: {funcname}")
        return node


def create_np_lib_call(func_name: str, args: list[ast.expr], **kwargs: ast.expr):
    keywords = [ast.keyword(arg=name, value=expr) for name, expr in kwargs]
    return ast.Call(
        func=ast.Attribute(
            value=ast.Name(id="np", ctx=ast.Load()), attr=func_name, ctx=ast.Load()
        ),
        args=args,
        keywords=keywords,
    )


libfunc_transform_info: dict[
    str, Callable[[str, list[ast.expr], ast.Call], ast.expr]
] = {}


def register_libfunc_transform_info(*funcnames: str):
    def decorator(f):
        for funcname in funcnames:
            libfunc_transform_info[funcname] = f
        return f

    return decorator


lib_func_name_map = {
    "sub": "subtract",
    "mul": "multiply",
    "innerprod": "inner",
    "div": "divide",
    "where": "flatnonzero",
    "eq": "equal",
    "neq": "not_equal",
    "lt": "less",
    "gt": "greater",
    "le": "less_equal",
    "ge": "greater_equal",
}


@register_libfunc_transform_info(*lib_func_name_map.keys())
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    return create_np_lib_call(lib_func_name_map[func_name], args)


@register_libfunc_transform_info("matmult")
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    assert len(args) == 2
    return ast.BinOp(op=ast.MatMult(), left=args[0], right=args[1])


@register_libfunc_transform_info("bool", "int32", "int64", "float32", "float64")
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    if len(args) == 1:
        return create_np_lib_call(func_name, args)
    assert len(args) == 0
    return create_np_lib_call(func_name, [ast.Num(0)])


dtype_map = {
    "bool": "bool_",
    "int32": "int32",
    "int64": "int64",
    "float32": "float32",
    "float64": "float64",
}


@register_libfunc_transform_info("zeros", "empty", "arange")
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    assert 1 <= len(args) <= 2
    if len(args) == 1:
        return create_np_lib_call(func_name, args)

    dtype_expr = args[1]
    if not (
        isinstance(dtype_expr, ast.Call)
        and isinstance(dtype_expr.func, ast.Attribute)
        and isinstance(dtype_expr.func.value, ast.Name)
        and dtype_expr.func.value.id == "np"
        and dtype_expr.func.attr in dtype_map
    ):
        raise Exception(f"Unsupported dtype: {astunparse.dump(dtype_expr)}")
    t = dtype_map[dtype_expr.func.attr]

    return create_np_lib_call(
        func_name,
        [
            args[0],
            ast.Attribute(
                value=ast.Name(id="np", ctx=ast.Load()), attr=t, ctx=ast.Load()
            ),
        ],
    )


@register_libfunc_transform_info("shape")
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    if len(args) == 1:
        return create_np_lib_call("array", [create_np_lib_call("shape", args[:1])])
    else:
        assert len(args) == 2
        return ast.Subscript(
            value=create_np_lib_call("shape", args[:1]),
            slice=ast.Index(value=args[1]),
            ctx=ast.Load(),
        )


@register_libfunc_transform_info("get_row")
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    assert len(args) == 2
    return ast.Subscript(value=args[0], slice=ast.Index(value=args[1]), ctx=ast.Load())


@register_libfunc_transform_info("get_col")
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    assert len(args) == 2
    return ast.Subscript(
        value=args[0],
        slice=ast.Tuple(elt=[ast.Slice(), ast.Index(value=args[1])], ctx=ast.Load()),
        ctx=ast.Load(),
    )


@register_libfunc_transform_info("set_row")
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    assert len(args) == 3
    return ast.Assign(
        targets=[
            ast.Subscript(
                value=args[0], slice=ast.Index(value=args[1]), ctx=ast.Store()
            )
        ],
        value=args[2],
    )


@register_libfunc_transform_info("set_col")
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    assert len(args) == 3
    return ast.Assign(
        targets=[
            ast.Subscript(
                value=args[0],
                slice=ast.Tuple(
                    elt=[ast.Slice(), ast.Index(value=args[1])], ctx=ast.Load()
                ),
                ctx=ast.Store(),
            )
        ],
        value=args[2],
    )


@register_libfunc_transform_info("plus_eq")
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    assert len(args) == 2
    args[0].ctx = ast.Store()
    return ast.AugAssign(target=args[0], op=ast.Add(), value=args[1])


@register_libfunc_transform_info("minus_eq")
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    assert len(args) == 2
    args[0].ctx = ast.Store()
    return ast.AugAssign(target=args[0], op=ast.Sub(), value=args[1])


@register_libfunc_transform_info("append")
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    assert len(args) == 2
    return ast.Call(
        func=ast.Attribute(value=args[0], attr="append", ctx=ast.Load()),
        args=[args[1]],
        keywords=[],
    )


@register_libfunc_transform_info("heapinit_empty")
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    assert not args
    return ast.List(elts=[], ctx=ast.Load())


@register_libfunc_transform_info("heapinit")
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    assert len(args) == 2
    return ast.List(elts=[ast.Tuple(elts=args, ctx=ast.Load())], ctx=ast.Load())


@register_libfunc_transform_info("heappush")
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    assert len(args) == 3
    return ast.Call(
        func=ast.Attribute(
            value=ast.Name(id="heapq", ctx=ast.Load()), attr="heappush", ctx=ast.Load()
        ),
        args=[args[0], ast.Tuple(elts=[args[1], args[2]], ctx=ast.Load())],
        keywords=[],
    )


@register_libfunc_transform_info("heappop")
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    assert len(args) == 1
    return ast.Call(
        func=ast.Attribute(
            value=ast.Name(id="heapq", ctx=ast.Load()), attr="heappop", ctx=ast.Load()
        ),
        args=args,
        keywords=[],
    )


@register_libfunc_transform_info("heappeek_key")
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    assert len(args) == 1
    return ast.Subscript(
        value=ast.Subscript(
            value=args[0], slice=ast.Index(value=ast.Num(0)), ctx=ast.Load()
        ),
        slice=ast.Index(value=ast.Num(0)),
        ctx=ast.Load(),
    )


@register_libfunc_transform_info("heappeek_val")
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    assert len(args) == 1
    return ast.Subscript(
        value=ast.Subscript(
            value=args[0], slice=ast.Index(value=ast.Num(0)), ctx=ast.Load()
        ),
        slice=ast.Index(value=ast.Num(1)),
        ctx=ast.Load(),
    )


@register_libfunc_transform_info("heapsize")
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    assert len(args) == 1
    return ast.Call(func=ast.Name(id="len", ctx=ast.Load()), args=args, keywords=[])


@register_libfunc_transform_info("heap_get_key")
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    assert len(args) == 2
    return ast.Subscript(
        value=ast.Subscript(
            value=args[0], slice=ast.Index(value=args[1]), ctx=ast.Load()
        ),
        slice=ast.Index(value=ast.Num(0)),
        ctx=ast.Load(),
    )


@register_libfunc_transform_info("csr_to_spm")
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    assert len(args) == 4
    data, indices, indptr, ncols = args
    nrows = ast.BinOp(
        op=ast.Sub(),
        left=ast.Call(
            func=ast.Name(id="len", ctx=ast.Load()), args=[indptr], keywords=[]
        ),
        right=ast.Num(n=1),
    )
    return ast.Call(
        func=ast.Name(id="csr_matrix", ctx=ast.Load()),
        args=[ast.Tuple(elts=[data, indices, indptr], ctx=ast.Load())],
        keywords=[
            ast.keyword(
                arg="shape", value=ast.Tuple(elts=[nrows, ncols], ctx=ast.Load())
            )
        ],
    )


@register_libfunc_transform_info("spm_mul")
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    assert len(args) == 2
    # return ast.BinOp(op=ast.Mult(), left=args[0], right=args[1])
    return ast.Call(
        func=ast.Attribute(value=args[0], attr="multiply", ctx=ast.Load()),
        args=[args[1]],
        keywords=[],
    )


@register_libfunc_transform_info("spm_add")
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    assert len(args) == 2
    return ast.BinOp(op=ast.Add(), left=args[0], right=args[1])


@register_libfunc_transform_info("spmm_dense", "spmm", "spmv")
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    assert len(args) == 2
    return ast.BinOp(op=ast.MatMult(), left=args[0], right=args[1])


@register_libfunc_transform_info("spm_diags")
def _(func_name: str, args: list[ast.expr], node: ast.Call):
    assert len(args) == 1
    return ast.Call(func=ast.Name(id="diags", ctx=ast.Load()), args=args, keywords=[])
