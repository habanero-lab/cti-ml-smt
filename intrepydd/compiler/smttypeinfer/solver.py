import os
from typing import Sequence

_solver_name = os.getenv('SMT_SOLVER', 'cvc5')

if _solver_name == 'z3':
    import z3

    def Solver():
        solver = z3.Solver()
        return solver

    def get_constructor_name(ty: z3.ExprRef) -> str:
        return ty.decl().name()

    def get_arg(ty: z3.ExprRef, idx: int) -> z3.ExprRef:
        return ty.arg(idx)

    def as_long(ty: z3.ExprRef) -> int:
        return ty.as_long()

    def RecVar(name: str, sort: z3.SortRef):
        return z3.Const(name, sort)

    def RecFunction(name: str, *sig):
        return z3.RecFunction(name, sig)

    def RecAddDefinition(fun: z3.FuncDeclRef,
                         args: Sequence[z3.ExprRef],
                         body: z3.ExprRef):
        z3.RecAddDefinition(fun, args, body)
        return fun

elif _solver_name == 'cvc5':
    import cvc5.pythonic as z3

    z3.main_ctx().solver.setOption('fmf-fun', 'true')
    
    _And = z3.And
    _Or = z3.Or

    def And(*args):
        if len(args) > 1:
            return _And(*args)
        if len(args) == 1:
            return args[0]
        return True

    def Or(*args):
        if len(args) > 1:
            return _Or(*args)
        if len(args) == 1:
            return args[0]
        return True

    z3.And = And
    z3.Or = Or

    def Solver():
        solver = z3.Solver()
        return solver

    def get_constructor_name(ty: z3.ExprRef) -> str:
        if ty.ast.getNumChildren() == 0:
            return ty.ast.getSymbol()
        else:
            return ty.ast[0].getSymbol()

    def get_arg(ty: z3.ExprRef, idx: int) -> z3.ExprRef:
        return ty.arg(idx + 1)

    def as_long(ty: z3.ExprRef) -> int:
        return ty.ast.toPythonObj()

    def RecVar(name: str, sort: z3.SortRef):
        ctx = z3.get_ctx(sort.ctx)
        var = ctx.solver.mkVar(sort.ast, name)
        return z3.ExprRef(var, ctx)

    def RecFunction(name: str, *sig):
        return z3.Function(name, sig)

    def RecAddDefinition(fun: z3.FuncDeclRef,
                         args: Sequence[z3.ExprRef],
                         body: z3.ExprRef):
        ctx = z3.get_ctx(body.ctx)
        bvs = [arg.ast for arg in args]
        f = ctx.solver.defineFunRec(fun.ast, bvs, body.ast)
        return z3.FuncDeclRef(f, ctx)

else:
    raise Exception(f'Unknown SMT solver name: {_solver_name}')
