from typing import Optional, Sequence
import typed_ast.ast3 as ast
import copy

from .solver import z3
import glb
import utils
import mytypes
from symboltable import symtab
from utils import get_operator_expansion_func
from .typemanager import TypeManager
from .libfunctypes import handle_libfunc_callsite
from .mlmodel import get_ml_model


def infer(module: ast.AST, src: str):
    assert isinstance(module, ast.Module)

    if glb.args.verbose:
        import time
        t_start = time.time()

    preprocessor = Preprocessor()
    preprocessor.visit(module)
    max_tuple_size = preprocessor.max_tuple_size

    type_manager = TypeManager(max_tuple_size)

    functions: dict[str, FunctionInfo] = {}
    new_body = []
    for stmt in module.body:
        if isinstance(stmt, ast.FunctionDef):
            functions[stmt.name] = FunctionInfo(stmt, type_manager)
        else:
            new_body.append(stmt)
    module.body = new_body

    type_constraint_collector = TypeConstraintCollector(
        functions, type_manager)
    for func_info in functions.values():
        type_constraint_collector.do_func_body(func_info.fdef)

    param_types = []
    for func_info in functions.values():
        param_types += func_info.param_types

    if glb.args.smtti_use_bounds:
        if glb.args.smtti_same_bitwidth:
            int_dtype = type_manager.create_dtype()
            float_dtype = type_manager.create_dtype()
            type_manager.add_constraint(z3.Or(
                int_dtype == type_manager.dtype.int32,
                int_dtype == type_manager.dtype.int64
            ))
            type_manager.add_constraint(z3.Or(
                float_dtype == type_manager.dtype.float32,
                float_dtype == type_manager.dtype.float64
            ))
            dtypes = [int_dtype, float_dtype, type_manager.dtype.bool]
        else:
            dtypes = None
        for ty in param_types:
            type_manager.add_constraint(
                type_manager.is_valid_param_type(ty, dtypes=dtypes))

    type_manager.solver.check()

    if not glb.args.smtti_use_ml:
        num_models, generated_functions = \
            generate_typed_functions(type_manager, functions, module)

    else:
        ml_model = get_ml_model()
        ml_constraints = ml_model.get_ml_constraints(
            functions, src, type_manager)

        if not glb.args.smtti_ml_auto:
            param_ml_constraints = [z3.Or(*cs) for cs in ml_constraints]
            ml_ratio = glb.args.smtti_ml_ratio
            assert 0. <= ml_ratio <= 1.
            if ml_ratio == 1.:
                type_manager.add_constraint(z3.And(*param_ml_constraints))
            else:
                num_params = sum(
                    map(lambda func_info: len(
                        func_info.param_types), functions.values())
                )
                threshold = int(num_params * ml_ratio)
                if threshold < num_params * ml_ratio:
                    threshold += 1
                type_manager.add_constraint(
                    z3.Sum(*[z3.If(ml_constraint, 1, 0)
                             for ml_constraint in param_ml_constraints])
                    >= threshold
                )
            num_models, generated_functions = \
                generate_typed_functions(type_manager, functions, module)

        else:  # smtti_ml_auto
            num_params = sum(
                map(lambda func_info: len(func_info.param_types),
                    functions.values())
            )
            num_models = 0
            generated_functions = set()

            ks = [1, 5, 10, 20]
            for i in range(len(ks)):
                if ks[i] > glb.args.smtti_top_k:
                    ks = ks[:i]
                    break
            if glb.args.verbose:
                print(f'Top-K to use: {ks}')

            ms = [num_params ** i for i in range(len(ks) - 1, -1, -1)]
            scores = []
            for cs in ml_constraints:
                for k, m in zip(ks, ms):
                    scores.append(z3.If(z3.Or(*cs[:k]), m, 0))
            score = z3.Sum(*scores)

            # l = 0
            # r = sum(ms) * num_params
            # max_score = -1
            # while l <= r:
            #     mid = (l + r) // 2
            #     if glb.args.verbose:
            #         print(f'Testing score range: {l}, {r}')
            #     result = type_manager.solver.check(score >= mid)
            #     if result == z3.sat:
            #         l = mid + 1
            #         max_score = mid
            #     else:
            #         assert result == z3.unsat
            #         r = mid - 1
            # assert max_score >= 0

            # if glb.args.verbose:
            #     print(f'Max score: {max_score}')

            # if max_score > 0:
            #     type_manager.add_constraint(score >= max_score)
            # num_models, generated_functions = \
            #     generate_typed_functions(
            #         type_manager, functions, module, num_models, generated_functions)
            # if glb.args.verbose:
            #     print(f'Num versions: {num_models}')

            min_versions = glb.args.smtti_ml_auto_min_vers
            l = 0
            r = sum(ms) * num_params
            while True:
                max_score = -1
                while l <= r:
                    mid = (l + r) // 2
                    if glb.args.verbose:
                        print(f'Testing score range: {l}, {r}')
                    result = type_manager.solver.check(score >= mid)
                    if result == z3.sat:
                        l = mid + 1
                        max_score = mid
                    else:
                        assert result == z3.unsat
                        r = mid - 1

                if max_score < 0:
                    break

                if glb.args.verbose:
                    print(f'Max score: {max_score}')

                num_models, generated_functions = \
                    generate_typed_functions(type_manager, functions, module,
                                             num_models, generated_functions,
                                             score >= max_score)
                if glb.args.verbose:
                    print(f'Num versions: {num_models}')
                if num_models >= min_versions or max_score == 0:
                    break
                l = 0
                r = max_score - 1

            # for threshold in range(num_params, -1, -1):
            #     if glb.args.verbose:
            #         print(f'Allow {num_params - threshold}/{num_params} '
            #               'non-ML params')
            #     done = False
            #     for k in ks:
            #         if glb.args.verbose:
            #             print(f'Using top-{k} predictions')
            #         param_ml_constraints = [
            #             z3.Or(*cs[:k]) for cs in ml_constraints]
            #         type_manager.solver.push()
            #         type_manager.add_constraint(
            #             z3.Sum(*[z3.If(ml_constraint, 1, 0)
            #                      for ml_constraint in param_ml_constraints])
            #             >= threshold
            #         )
            #         num_models, generated_functions = \
            #             generate_typed_functions(
            #                 type_manager, functions, module, num_models, generated_functions)
            #         if num_models > 0:
            #             if glb.args.verbose:
            #                 print(f'{num_models} models when allowing '
            #                       f'{num_params - threshold}/{num_params} non-ML params '
            #                       f'using top-{k} predictions')
            #             done = True
            #             break
            #         type_manager.solver.pop()
            #     if done:
            #         break

            # for threshold in range(num_params, -1, -1):
            #     if glb.args.verbose:
            #         print(
            #             f'Allow {num_params - threshold}/{num_params} non-ML params')
            #     type_manager.solver.push()
            #     type_manager.add_constraint(
            #         z3.Sum(*[z3.If(ml_constraint, 1, 0)
            #                  for ml_constraint in ml_constraints])
            #         >= threshold
            #     )
            #     num_models, num_generated_functions = \
            #         generate_typed_functions(type_manager, functions, module)
            #     if num_models > 0:
            #         if glb.args.verbose:
            #             print(f'{num_models} models when allowing '
            #                   f'{num_params - threshold}/{num_params} non-ML params')
            #         break
            #     type_manager.solver.pop()

    if glb.args.verbose:
        print(f'Num models: {num_models}')
        print(f'Num generated functions: {len(generated_functions)}')
        import time
        t_end = time.time()
        print(f'SMT type inference time: {t_end - t_start}')

    if num_models == 0:
        raise glb.TypeError('Type error: type constraints unsatisfiable')

    return functions


class MaxNumModelsExceeded(Exception):
    pass


def generate_typed_functions(type_manager: TypeManager,
                             functions: dict[str, 'FunctionInfo'],
                             module: ast.Module,
                             count: int = 0,
                             generated_func_names: Optional[set[str]] = None,
                             *assumptions: z3.ExprRef):
    solver = type_manager.solver
    param_types = []
    for func_info in functions.values():
        param_types += func_info.param_types

    if generated_func_names is None:
        generated_func_names = set()

    def generate_for_model(model: z3.ModelRef):
        nonlocal count
        count += 1

        if glb.args.verbose:
            if count % 100 == 0:
                print(f'count: {count}')
        if count > glb.args.smtti_max_smt_models:
            raise MaxNumModelsExceeded

        model_constraints = []

        for func_name, func_info in functions.items():
            param_types: list[mytypes.MyType] = []
            for t in func_info.param_types:
                ty = model.eval(t)
                model_constraints.append(t == ty)
                myty = type_manager.to_mytype(ty)
                param_types.append(myty)
            mangled_name = get_mangled_function_name(func_name, param_types)
            if mangled_name in generated_func_names:
                continue
            generated_func_names.add(mangled_name)

            if glb.args.smtti_count_only:
                continue

            fdef = generate_typed_function(mangled_name,
                                           func_info,
                                           model,
                                           type_manager,
                                           functions)
            module.body.append(fdef)

            symtab.register_user_func(fdef)
            glb.cpp_module.add_function(fdef)

        return model_constraints

    if not param_types:
        if solver.check(*assumptions) == z3.sat:
            generate_for_model(solver.model())
    else:
        try:
            while True:
                if solver.check(*assumptions) == z3.unsat:
                    break
                model_constraints = generate_for_model(solver.model())
                solver.add(z3.Not(z3.And(*model_constraints)))
        except MaxNumModelsExceeded:
            print(f'More than {glb.args.smtti_max_smt_models} models found. '
                  f'Only the first {glb.args.smtti_max_smt_models} used.')

    return count, generated_func_names


# def generate_typed_functions(type_manager: TypeManager,
#                              functions: dict[str, 'FunctionInfo'],
#                              module: ast.Module):
#     solver = type_manager.solver
#     param_types = []
#     for func_info in functions.values():
#         param_types += func_info.param_types

#     count = 0
#     total = 0

#     generated_func_names: set[str] = set()

#     def generate_for_model(model: z3.ModelRef):
#         nonlocal count, total
#         count += 1
#         total += 1

#         if glb.args.verbose:
#             if count % 100 == 0:
#                 print(f'count: {count}')
#             if total % 100 == 0:
#                 print(f'total: {total}')
#         if count > glb.args.smtti_max_smt_models:
#             raise MaxNumModelsExceeded

#         for func_name, func_info in functions.items():
#             param_types: list[mytypes.MyType] = []
#             for t in func_info.param_types:
#                 ty = model.eval(t)
#                 myty = type_manager.to_mytype(ty)
#                 param_types.append(myty)
#             mangled_name = get_mangled_function_name(func_name, param_types)
#             if mangled_name in generated_func_names:
#                 continue
#             generated_func_names.add(mangled_name)

#             if glb.args.smtti_count_only:
#                 continue

#             fdef = generate_typed_function(mangled_name,
#                                            func_info,
#                                            model,
#                                            type_manager,
#                                            functions)
#             module.body.append(fdef)

#             symtab.register_user_func(fdef)
#             glb.cpp_module.add_function(fdef)

#     def rec_enum(i):
#         nonlocal count, total

#         t = param_types[i]
#         solver.push()

#         while True:
#             if solver.check() == z3.unsat:
#                 total += 1
#                 if total % 100 == 0:
#                     if glb.args.verbose:
#                         print(f'total: {total}')
#                 break
#             model = solver.model()
#             ty = model.eval(t)

#             if i == len(param_types) - 1:
#                 generate_for_model(model)
#             else:
#                 solver.push()
#                 solver.add(t == ty)
#                 rec_enum(i + 1)
#                 solver.pop()

#             solver.add(t != ty)

#         solver.pop()

#     if not param_types:
#         if solver.check() == z3.sat:
#             generate_for_model(solver.model())
#     else:
#         try:
#             rec_enum(0)
#         except MaxNumModelsExceeded:
#             print(f'More than {glb.args.smtti_max_smt_models} models found. '
#                   f'Only the first {glb.args.smtti_max_smt_models} used.')

#     return count, len(generated_func_names)


def get_mangled_function_name(original_name: str,
                              param_types: Sequence[mytypes.MyType]):
    name_segs = [f'_F{len(original_name)}{original_name}']
    for param_type in param_types:
        name_segs.append(param_type.get_mangled_name())
    return ''.join(name_segs)


def generate_typed_function(mangled_name: str,
                            func_info: 'FunctionInfo',
                            model: z3.ModelRef,
                            type_manager: TypeManager,
                            functions: dict[str, 'FunctionInfo']):
    fdef = copy.deepcopy(func_info.fdef)
    fdef.name = mangled_name

    typemap: dict[str, mytypes.MyType] = {}
    for name, ty in func_info.type_map.items():
        typemap[name] = type_manager.to_mytype(model.eval(ty))
    fdef.localtypes = typemap
    fdef.typemap = typemap

    for arg, param_type in zip(fdef.args.args, func_info.param_types):
        arg.type = type_manager.to_mytype(model.eval(param_type))

    fdef.ret_type = type_manager.to_mytype(model.eval(func_info.ret_type))

    TypeAnnotator(func_info, model, type_manager, functions).visit(fdef)

    return fdef


class TypeAnnotator(ast.NodeVisitor):

    def __init__(self,
                 func_info: 'FunctionInfo',
                 model: z3.ModelRef,
                 type_manager: TypeManager,
                 functions: dict[str, 'FunctionInfo']):
        self.func_info = func_info
        self.model = model
        self.type_manager = type_manager
        self.functions = functions

    def visit(self, node: ast.AST):
        expr_id = getattr(node, 'expr_id', -1)
        if expr_id >= 0:
            ty = self.func_info.expr_id_to_type[expr_id]
            myty = self.type_manager.to_mytype(self.model.eval(ty))
            node.type = myty
        super().visit(node)

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)

        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            func_info = self.functions.get(func_name, None)
            if func_info is not None:
                # User-defined function
                param_types: list[mytypes.MyType] = []
                for t in func_info.param_types:
                    ty = self.model.eval(t)
                    myty = self.type_manager.to_mytype(ty)
                    param_types.append(myty)
                mangled_name = get_mangled_function_name(
                    func_name, param_types)
                node.func.id = mangled_name


class Preprocessor(ast.NodeVisitor):

    def __init__(self):
        self.max_tuple_size = 0

    def visit_Tuple(self, node: ast.Tuple):
        self.max_tuple_size = max(self.max_tuple_size, len(node.elts))
        self.generic_visit(node)


class FunctionInfo:

    def __init__(self, fdef: ast.FunctionDef, type_manager: TypeManager):
        self.fdef = fdef
        self.type_map: dict[str, z3.ExprRef] = {}
        self.expr_id_to_type: list[z3.ExprRef] = []

        if glb.args.smtti_use_annot and fdef.returns is not None:
            self.ret_type = type_manager.from_mytype(
                utils.get_annotation_type(fdef.returns))
        else:
            self.ret_type = type_manager.create_type()

        self.param_types: list[z3.ExprRef] = []
        for arg in fdef.args.args:
            if glb.args.smtti_use_annot and arg.annotation is not None:
                arg_type = type_manager.from_mytype(
                    utils.get_annotation_type(arg.annotation))
            else:
                arg_type = type_manager.create_type()
            self.type_map[arg.arg] = arg_type
            self.param_types.append(arg_type)

    def record_expr_type(self, node: ast.expr, ty: z3.ExprRef):
        assert not hasattr(node, 'expr_id')
        expr_id = len(self.expr_id_to_type)
        setattr(node, 'expr_id', expr_id)
        self.expr_id_to_type.append(ty)


class TypeConstraintCollector:

    def __init__(self, functions: dict[str, FunctionInfo],
                 type_manager: TypeManager):
        self.functions = functions
        self.type_manager = type_manager
        self.type = type_manager.type
        self.dtype = type_manager.dtype
        self.func_info: Optional[FunctionInfo] = None
        self.func_has_return = False

    def add_constraint(self, constraint):
        self.type_manager.add_constraint(constraint)

    def do_func_body(self, node: ast.FunctionDef):
        assert self.func_info is None
        self.func_info = self.functions[node.name]
        self.func_has_return = False

        for stmt in node.body:
            self.do_stmt(stmt)

        if not self.func_has_return:
            self.add_constraint(
                self.type_manager.is_none(self.func_info.ret_type))

        self.func_info = None

    def is_str_stmt(self, node: ast.stmt) -> bool:
        if isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Str):
                return True
        return False

    def do_stmt(self, node: ast.stmt) -> None:
        if isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Str):
                return

        if isinstance(node, ast.Assign):
            self.do_Assign(node)
        elif isinstance(node, ast.AnnAssign):
            self.do_AnnAssign(node)
        elif isinstance(node, ast.AugAssign):
            self.do_AugAssign(node)
        elif isinstance(node, ast.For):
            self.do_For(node)
        elif isinstance(node, ast.While):
            self.do_While(node)
        elif isinstance(node, ast.If):
            self.do_If(node)
        elif isinstance(node, ast.Return):
            self.do_Return(node)
        elif isinstance(node, ast.Expr):
            self.do_Expr(node)
        elif isinstance(node, ast.Break):
            return
        else:
            glb.exit_on_unsupported_node(node)

    def do_expr(self, node: ast.expr) -> z3.ExprRef:
        if isinstance(node, ast.Num):
            ty = self.do_Num(node)
        elif isinstance(node, ast.Name):
            ty = self.do_name(node)
        elif isinstance(node, ast.NameConstant):
            ty = self.do_name_constant(node)
        elif isinstance(node, ast.Tuple):
            ty = self.do_Tuple(node)
        elif isinstance(node, ast.List):
            ty = self.do_List(node)
        elif isinstance(node, ast.UnaryOp):
            ty = self.do_UnaryOp(node)
        elif isinstance(node, ast.BinOp):
            ty = self.do_BinOp(node)
        elif isinstance(node, ast.BoolOp):
            ty = self.do_BoolOp(node)
        elif isinstance(node, ast.Compare):
            ty = self.do_Compare(node)
        elif isinstance(node, ast.Subscript):
            ty = self.do_Subscript(node)
        elif isinstance(node, ast.Call):
            ty = self.do_Call(node)
        else:
            glb.exit_on_unsupported_node(node)
        self.add_constraint(self.type_manager.is_valid_type(ty))
        self.func_info.record_expr_type(node, ty)
        return ty

    def do_BoolOp(self, node: ast.BoolOp):
        assert len(node.values) >= 2
        funcname = get_operator_expansion_func(node.op)
        assert funcname != 'n/a'
        ty = self.do_expr(node.values[0])
        for val in node.values[1:]:
            vty = self.do_expr(val)
            ty = handle_libfunc_callsite(
                funcname, [ty, vty], self.type_manager)
        return ty

    def do_Compare(self, node: ast.Compare):
        if len(node.comparators) != 1:
            glb.exit_on_unsupported_node(node)
        funcname = get_operator_expansion_func(node.ops[0])
        assert funcname != 'n/a'
        lty = self.do_expr(node.left)
        rty = self.do_expr(node.comparators[0])
        return handle_libfunc_callsite(funcname, [lty, rty], self.type_manager)

    def do_UnaryOp(self, node: ast.UnaryOp):
        ty = self.do_expr(node.operand)
        funcname = get_operator_expansion_func(node.op)
        assert funcname != 'n/a'
        return handle_libfunc_callsite(funcname, [ty], self.type_manager)
        # self.add_constraint(self.type_manager.is_array(ty))
        # if isinstance(node.op, ast.Not):
        #     self.add_constraint(self.type.array_dtype(ty) == self.dtype.bool)
        #     return ty
        # elif isinstance(node.op, ast.USub):
        #     self.add_constraint(self.type.array_dtype(ty) != self.dtype.bool)
        #     return ty
        # else:
        #     glb.exit_on_unsupported_node(node)

    def do_BinOp(self, node: ast.BinOp):
        lty = self.do_expr(node.right)
        rty = self.do_expr(node.left)
        funcname = get_operator_expansion_func(node.op)
        assert funcname != 'n/a'
        return handle_libfunc_callsite(funcname, [lty, rty], self.type_manager)

    def do_Subscript(self, node: ast.Subscript):
        vty = self.do_expr(node.value)
        assert isinstance(node.slice, ast.Index)
        ity = self.do_expr(node.slice.value)
        ty = self.type_manager.create_type()

        if isinstance(node.slice.value, ast.Tuple):
            slice_dim = len(node.slice.value.elts)
            array_index_constraint = self.type_manager.tuple_all(
                ity, self.type_manager.is_int)
            self.add_constraint(
                z3.Or(
                    z3.And(
                        self.type_manager.is_array(vty),
                        self.type.array_ndim(vty) == slice_dim,
                        array_index_constraint,
                        ty == self.type_manager.create_array(
                            self.type.array_dtype(vty), 0)),
                    z3.And(self.type_manager.is_dict(vty),
                           self.type.dict_ktype(vty) == ity,
                           ty == self.type.dict_vtype(vty))))
        else:
            slice_dim = 1
            array_index_constraint = self.type_manager.is_int(ity)
            self.add_constraint(
                z3.Or(
                    z3.And(
                        self.type_manager.is_array(vty),
                        self.type.array_ndim(vty) == slice_dim,
                        array_index_constraint,
                        ty == self.type_manager.create_array(
                            self.type.array_dtype(vty), 0)),
                    z3.And(self.type_manager.is_list(vty),
                           self.type_manager.is_int(ity),
                           ty == self.type.list_etype(vty)),
                    z3.And(self.type_manager.is_dict(vty),
                           self.type.dict_ktype(vty) == ity,
                           ty == self.type.dict_vtype(vty))))

        return ty

    def do_Expr(self, node: ast.Expr):
        if isinstance(node.value, ast.Call):
            self.do_expr(node.value)
        else:
            glb.exit_on_unsupported_node(node)

    def do_Call(self, node: ast.Call):
        arg_types = [self.do_expr(arg) for arg in node.args]

        if not isinstance(node.func, ast.Name):
            glb.exit_on_node(node.func, 'unsupported function')

        # Either should be a user-defined function or a global function
        funcname = node.func.id

        if funcname.startswith('_'):
            # functions like _add/_sub have no meaningful return value for now
            return self.type_manager.create_none()

        callee_info = self.functions.get(funcname, None)
        if callee_info is not None:
            # User-defined function
            for arg_type, param_type in zip(arg_types,
                                            callee_info.param_types):
                self.add_constraint(arg_type == param_type)
            return callee_info.ret_type
        else:
            return handle_libfunc_callsite(funcname, arg_types,
                                           self.type_manager)

    def do_AnnAssign(self, node: ast.AnnAssign):
        assert node.value
        vty = self.do_expr(node.value)
        if glb.args.smtti_use_annot:
            annot_ty = self.type_manager.from_mytype(
                utils.get_annotation_type(node.annotation))
            self.add_constraint(vty == annot_ty)
        tty = self.do_expr(node.target)
        self.add_constraint(tty == vty)
        node.targets = [node.target]

    def do_AugAssign(self, node: ast.AugAssign):
        rty = self.do_BinOp(
            ast.BinOp(left=node.target, op=node.op, right=node.value))
        tty = self.func_info.expr_id_to_type[node.target.expr_id]
        self.add_constraint(tty == rty)
        node.targets = [node.target]

    def do_Assign(self, node: ast.Assign):
        vty = self.do_expr(node.value)
        for target in node.targets:
            tty = self.do_expr(target)
            self.add_constraint(tty == vty)

    def do_name(self, node: ast.Name):
        assert self.func_info
        type_map = self.func_info.type_map
        name = node.id
        ty = type_map.get(name, None)
        if ty is None:
            ty = self.type_manager.create_type()
            type_map[name] = ty
        return ty

    def do_Num(self, node: ast.Num):
        ty = self.type_manager.create_type()
        if type(node.n) == float:
            self.add_constraint(self.type_manager.is_float(ty))
        elif type(node.n) == int:
            self.add_constraint(self.type_manager.is_int(ty))
            # self.add_constraint(
            #     z3.Or(self.type_manager.is_int(ty),
            #           self.type_manager.is_float(ty)))
        else:
            glb.exit_on_unsupported_node(node)
        return ty

    def do_name_constant(self, node: ast.NameConstant):
        if isinstance(node.value, bool):
            return self.type_manager.create_array(self.dtype.bool, 0)
        else:
            glb.exit_on_unsupported_node(node)

    def do_Tuple(self, node: ast.Tuple):
        etypes = [self.do_expr(elt) for elt in node.elts]
        return self.type_manager.create_tuple(etypes)

    def do_List(self, node: ast.List):
        ty = self.type_manager.create_list()
        if node.elts:
            ety = self.do_expr(node.elts[0])
            for elt in node.elts[1:]:
                self.add_constraint(ety == self.do_expr(elt))
            self.add_constraint(self.type.list_etype(ty) == ety)
        return ty

    def do_For(self, node: ast.For):
        tty = self.do_expr(node.target)
        ity = self.do_expr(node.iter)
        for stmt in node.body:
            self.do_stmt(stmt)
        assert not node.orelse

        constraint = z3.Or(
            z3.And(
                self.type_manager.is_array(ity),
                self.type.array_ndim(ity) == 1,
                tty == self.type_manager.create_array(
                    self.type.array_dtype(ity), 0)),
            z3.And(self.type_manager.is_list(ity),
                   tty == self.type.list_etype(ity)),
            z3.And(self.type_manager.is_dict(ity),
                   tty == self.type.dict_ktype(ity)))
        self.add_constraint(constraint)

    def do_While(self, node: ast.While):
        tty = self.do_expr(node.test)
        for stmt in node.body:
            self.do_stmt(stmt)
        assert not node.orelse
        self.add_constraint(self.type_manager.is_bool(tty))

    def do_If(self, node: ast.If):
        tty = self.do_expr(node.test)
        for stmt in node.body:
            self.do_stmt(stmt)
        for stmt in node.orelse:
            self.do_stmt(stmt)
        self.add_constraint(self.type_manager.is_bool(tty))

    def do_Return(self, node: ast.Return):
        assert self.func_info
        self.func_has_return = True
        ty = self.func_info.ret_type
        if node.value:
            vty = self.do_expr(node.value)
            self.add_constraint(ty == vty)
        else:
            self.add_constraint(self.type_manager.is_none(ty))
