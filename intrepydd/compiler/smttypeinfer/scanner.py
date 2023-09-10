import typed_ast.ast3 as ast
import glb


class Scanner(ast.NodeVisitor):

    def __init__(self) -> None:
        self.fdef = None

    def visit_Module(self, node: ast.Module):
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import):
        glb.cpp_module.add_imported_module(node.names[0])

    def visit_FunctionDef(self, node: ast.FunctionDef):
        parent_fdef = self.fdef
        self.fdef = node
        node.has_pfor = False
        self.generic_visit(node)
        self.fdef = parent_fdef

    def visit_For(self, node: ast.For):
        self.generic_visit(node)
        if node.type_comment == 'pfor':
            node.is_pfor = True
            self.fdef.has_pfor = True
        else:
            node.is_pfor = False
