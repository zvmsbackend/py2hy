import ast

import hy
from hy.models import *

def unpack_aliases(aliases: list[ast.alias]):
    names = []
    for name in aliases:
        names.append(Symbol(name.name))
        if name.asname is not None:
            names.append(Keyword("as"))
            names.append(Symbol(name.asname))
    return names

constant_types = {
    int: Integer,
    float: Float,
    complex: Complex,
    str: String,
}

binop_types = {
    ast.Add: '+',
    ast.Sub: '-',
    ast.Mult: '*',
    ast.Div: '/',
    ast.Pow: '**',
    ast.Mod: '%',
    ast.FloorDiv: '//',
    ast.MatMult: '@'
}

uop_types = {
    ast.UAdd: '+',
    ast.USub: '-',
    ast.Invert: 'bnot',
    ast.Not: 'not'
}

cmpop_types = {
    ast.LtE: '<=',
    ast.Lt: '<',
    ast.GtE: '>=',
    ast.Gt: '>',
    ast.Eq: '=',
    ast.NotEq: '!=',
    ast.Is: 'is',
    ast.IsNot: 'not?',
    ast.In: 'in',
    ast.Not: 'not-in'
}

class Unparser(ast.NodeVisitor):
    def visit_list(self, node: list):
        if len(node) == 1:
            return self.visit(node[0])
        return Expression(Symbol('do'), *map(self.visit, node))

    def unparse(self, node: ast.AST):
        return hy.repr(self.visit(node))[1:]

    def visit_Module(self, node: ast.Module):
        return self.visit(node.body)
    
    def visit_Expr(self, node: ast.Expr):
        return self.visit(node.value)
    
    def visit_NamedExpr(self, node: ast.NamedExpr):
        return Expression([Symbol('setx'),
                           self.visit_List(node.target), 
                           self.visit(node.value)])
    
    def visit_Import(self, node: ast.Import):
        return Expression([Symbol('import'), *unpack_aliases(node.names)])
    
    def visit_ImportFrom(self, node: ast.ImportFrom):
        return Expression([Symbol('import'),
                           Symbol('.' * (node.level or 0) + node.module or ''),
                           *unpack_aliases(node.names)])
    
    def visit_Assign(self, node: ast.Assign):
        if len(node.targets) == 1:
            return Expression([Symbol('setv'), 
                               self.visit(node.targets[0]),
                               self.visit(node.value)])
        sym = hy.gensym()
        return Expression([Symbol('let'),
                           List([sym, self.visit(node.value)]),
                           *(Expression([Symbol('setv'),
                                         self.visit(target),
                                         sym]) for target in node.targets)])
    
    def visit_Constant(self, node: ast.Constant):
        return constant_types.get(type(node.value), Symbol)(node.value)
            
    def visit_AugAssign(self, node: ast.AugAssign):
        return Expression(Symbol(binop_types[type(node.op)] + '='),
                          self.visit(node.target),
                          self.visit(node.target))
    
    def visit_Name(self, node: ast.Name):
        return Symbol(node.id)
    
    def visit_List(self, node: ast.List):
        return List(map(self.visit, node.elts))
    
    def visit_Tuple(self, node: ast.Tuple):
        return Tuple(map(self.visit, node.elts))
    
    def visit_Set(self, node: ast.Set):
        return Set(map(self.visit, node.elts))
    
    def visit_Dict(self, node: ast.Dict):
        elts = []
        for k, v in zip(node.keys, node.values):
            elts.append(self.visit(k))
            elts.append(self.visit(v))
        return Dict(elts)
    
    def visit_AnnAssign(self, node: ast.AnnAssign):
        if node.value is None:
            return Expression([Symbol('annotate'),
                               self.visit(node.target),
                               self.visit(node.annotation)])
        return Expression([Symbol('setv'), 
                           self.visit(ast.AnnAssign(value=None,
                                                    target=node.target,
                                                    annotation=node.annotation)),
                           self.visit(node.value)])
    
    def visit_Return(self, node: ast.Return):
        return Expression([Symbol('return'), self.visit(node.value)])
    
    def visit_Pass(self, node: ast.Pass):
        return Symbol('pass')
    
    def visit_Break(self, node: ast.Break):
        return Expression([Symbol('break')])
    
    def visit_Continue(self, node: ast.Continue):
        return Expression([Symbol('continue')])
    
    def visit_Delete(self, node: ast.Delete):
        return Expression([Symbol('del'), *map(self.visit, node.targets)])
    
    def visit_Assert(self, node: ast.Assert):
        return Expression([Symbol('assert'), 
                           self.visit(node.test), 
                           *(() if node.msg is None else self.visit(node.msg))])
    
    def visit_Global(self, node: ast.Global):
        return Expression([Symbol('global'), *map(Symbol, node.names)])
    
    def visit_Nonlocal(self, node: ast.Nonlocal):
        return Expression([Symbol('nonlocal'), *map(Symbol, node.names)])
    
    def visit_Await(self, node: ast.Await):
        return Expression([Symbol('await'), self.visit(node.value)])
    
    def visit_Yield(self, node: ast.Yield):
        return Expression([Symbol('yield'), self.visit(node.value)])
    
    def visit_YieldFrom(self, node: ast.YieldFrom):
        return Expression([Symbol('yield-from'), self.visit(node.value)])

    def visit_Raise(self, node: ast.Raise):
        return Expression([Symbol('raise'),
                           self.visit(node.exc),
                           *(() if node.cause is None 
                             else [Keyword('from'), self.visit(node.cause)])])
    
    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        match node.type:
            case None:
                type = List()
            case ast.Tuple():
                type = List(map(self.visit, node.type.elts))
            case _:
                type = self.visit(node.type)
        if node.name is not None:
            type = List([Symbol(node.name), type])
        return Expression([Symbol('except'), type, *map(self.visit, node.body)])

    def visit_Try(self, node: ast.Try):
        return Expression([Symbol('try'),
                           *map(self.visit, node.body),
                           *map(self.visit, node.handlers),
                           *([Expression([Symbol('else'), *map(self.visit, node.orelse)])]
                             if node.orelse else ()),
                           *([Expression([Symbol('finally'), *map(self.visit, node.finalbody)])]
                             if node.finalbody else ())])
    
    def visit_ClassDef(self, node: ast.ClassDef):
        bases = list(map(self.visit, node.bases))
        for kw in node.keywords:
            bases.append(Keyword(kw.arg))
            bases.append(self.visit(kw.value))
        return Expression([Symbol('defclass'),
                           *([List(map(self.visit, node.decorator_list))]
                             if node.decorator_list else ()),
                           Symbol(node.name),
                           List(bases),
                           *map(self.visit, node.body)])
    
    def visit_arg(self, node: ast.arg):
        if node.annotation is None:
            return Symbol(node.arg)
        return Expression([Symbol('annotate'), Symbol(node.arg), self.visit(node.annotation)])
    
    def _function_helper(self, 
                         node: ast.FunctionDef | ast.AsyncFunctionDef, 
                         sym: str):
        return Expression([Symbol(sym),
                          *([List(map(self.visit, node.decorator_list))]
                            if node.decorator_list else ()),
                          (Expression([Symbol('annotate'),
                                       Symbol(node.name),
                                       self.visit(node.returns)])
                            if node.returns else Symbol(node.name)),
                          List([*map(self.visit, node.args.posonlyargs),
                               *([Symbol('/')] if node.args.posonlyargs else ()),
                               *map(self.visit, node.args.args),
                               *(() if node.args.vararg is None
                                 else [Expression([Symbol('unpack-iterable'), self.visit(node.args.vararg)])]),
                               *([Symbol('*')] if node.args.kwonlyargs and node.args.kwarg is None
                                  else ()),
                               *map(self.visit, node.args.kwonlyargs),
                               *(() if node.args.kwarg is None
                                 else [Expression(Symbol('unpack-mapping'), self.visit(node.args.kwarg))])]),
                          *map(self.visit, node.body)])
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        return self._function_helper(node, 'defn')
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        return self._function_helper(node, 'defn/a')
    
    def _for_helper(self, node: ast.For | ast.AsyncFor, is_async: bool):
        return Expression([Symbol('for'),
                           List([self.visit(node.target), self.visit(node.iter)]),
                           *([Keyword('async')] if is_async else ())
                           *map(self.visit, node.body),
                           *([Expression([Symbol('else'), *map(self.visit, node.orelse)])]
                             if node.orelse else ())])
    
    def visit_For(self, node: ast.For):
        return self._for_helper(node, False)
    
    def visit_AsyncFor(self, node: ast.AsyncFor):
        return self._for_helper(node, True)
    
    def visit_If(self, node: ast.If):
        if not node.orelse:
            return Expression([Symbol('when'), 
                               self.visit(node.test),
                               *map(self.visit, node.body)])
        branches = []
        while len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
            branches.append(self.visit(node.test))
            branches.append(self.visit(node.body))
            node = node.orelse[0]
        if branches:
            return Expression([Symbol('cond'),
                               *branches,
                               self.visit(node.test),
                               self.visit(node.body),
                               *([Symbol('True'),
                               self.visit(node.orelse)] if node.orelse
                               else ())])
        return Expression([Symbol('if'),
                           self.visit(node.test),
                           self.visit(node.body),
                           self.visit(node.orelse)])
    
    def visit_While(self, node: ast.While):
        return Expression([Symbol('while'),
                           self.visit(node.test),
                           *map(self.visit, node.body),
                           *(() if node.orelse is None
                            else [Expression([Symbol('else'),
                                              *map(self.visit, node.orelse)])])])

    def _with_helper(self, node: ast.With | ast.AsyncWith, sym: str):
        withitems = []
        if len(node.items) == 1 and node.items[0].optional_vars is None:
            withitems.append(self.visit(node.items[0].context_expr))
        else:
            for item in node.items:
                withitems.append(Symbol('_') if item.optional_vars is None
                                 else self.visit(item.optional_vars))
                withitems.append(self.visit(item.context_expr))
        return Expression([Symbol(sym),
                           List(withitems),
                           *map(self.visit, node.body)])

    def visit_With(self, node: ast.With):
        return self._with_helper(node, 'with')
    
    def visit_AsyncWith(self, node: ast.AsyncWith):
        return self._with_helper(node, 'with/a')
    
    def _comp_helper(self, 
                     node: ast.ListComp | ast.GeneratorExp 
                     | ast.SetComp | ast.DictComp,
                     sym: str):
        generators = []
        for gen in node.generators:
            generators.append(self.visit(gen.target))
            generators.append(self.visit(gen.iter))
            for if_ in gen.ifs:
                generators.append(Keyword('if'))
                generators.append(self.visit(if_))
            if gen.is_async:
                generators.append(Keyword('async'))
        return Expression([Symbol(sym), 
                           *generators,
                           *([self.visit(node.key), self.visit(node.value)]
                             if isinstance(node, ast.DictComp) else [self.visit(node.elt)]),])
    
    def visit_ListComp(self, node: ast.ListComp):
        return self._comp_helper(node, 'lfor')
    
    def visit_GeneratorExp(self, node: ast.GeneratorExp):
        return self._comp_helper(node, 'gfor')
    
    def visit_SetComp(self, node: ast.SetComp):
        return self._comp_helper(node, 'sfor')
    
    def visit_DictComp(self, node: ast.DictComp):
        return self._comp_helper(node, 'dfor')
    
    def visit_IfExp(self, node: ast.IfExp):
        return Expression([Symbol('if'),
                           self.visit(node.test),
                           self.visit(node.body),
                           self.visit(node.orelse)])
    
    def visit_UnaryOp(self, node: ast.UnaryOp):
        return Expression([Symbol(uop_types[type(node.op)]),
                           self.visit(node.operand)])
    
    def visit_BinOp(self, node: ast.BinOp):
        op = node.op
        operands = []
        while isinstance(node.left, ast.BinOp) and node.left.op == op:
            operands.append(self.visit(node.right))
            node = node.left
        operands.append(self.visit(node.right))
        operands.append(self.visit(node.left))
        return Expression([Symbol(binop_types[type(op)]), *reversed(operands)])

    def visit_Compare(self, node: ast.Compare):
        if len(set(node.ops)) == 1:
            return Expression([Symbol(cmpop_types[type(node.ops[0])]),
                               self.visit(node.left),
                               *map(self.visit, node.comparators)])
        comparators = [self.visit(node.left)]
        for op, cmp in zip(node.ops, node.comparators):
            comparators.append(Symbol(cmpop_types[type(op)]))
            comparators.append(self.visit(cmp))
        return Expression([Symbol('chainc'), *comparators])
    
    def visit_BoolOp(self, node: ast.BoolOp):
        return Expression([Symbol('and' if isinstance(node.op, ast.And)
                                  else 'or'),
                           *map(self.visit, node.values)])
    
    def visit_Attribute(self, node: ast.Attribute):
        return Expression([Symbol('.'),
                           self.visit(node.value),
                           Symbol(node.attr)])
    
    def visit_Subscript(self, node: ast.Subscript):
        if isinstance(node.slice, ast.Slice):
            return Expression([Symbol('cut'),
                               self.visit(node.value),
                               self.visit(node.slice.lower),
                               self.visit(node.slice.upper),
                               self.visit(node.slice.step)])
        return Expression([Symbol('get'),
                           self.visit(node.value),
                           self.visit(node.slice)])

unparser = Unparser()

def unparse(code: str):
    return unparser.unparse(ast.parse(code))

print(unparse('''a[b:c:d]'''))