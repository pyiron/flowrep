from __future__ import annotations

import ast
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from flowrep.models import edge_models
from flowrep.models.nodes import union, workflow_model
from flowrep.models.parsers import object_scope, symbol_scope

WalkerFactory = Callable[
    [object_scope.ScopeProxy, symbol_scope.SymbolScope], "BodyWalker"
]


@runtime_checkable
class BodyWalker(Protocol):
    """What control flow parsers need to walk a sub-body."""

    scope: object_scope.ScopeProxy
    symbol_map: symbol_scope.SymbolScope
    nodes: union.Nodes

    @property
    def inputs(self) -> list[str]: ...

    @property
    def input_edges(self) -> edge_models.InputEdges: ...

    @property
    def edges(self) -> edge_models.Edges: ...

    @property
    def output_edges(self) -> edge_models.OutputEdges: ...

    @property
    def outputs(self) -> list[str]: ...

    def visit(self, stmt: ast.AST) -> None: ...

    def walk(self, statements: list[ast.stmt]) -> None: ...

    def visit_Assign(self, stmt: ast.Assign) -> None: ...

    def visit_AnnAssign(self, stmt: ast.AnnAssign) -> None: ...

    def visit_Expr(self, stmt: ast.Expr) -> None: ...

    def visit_For(self, tree: ast.For) -> None: ...

    def visit_If(self, tree: ast.If) -> None: ...

    def visit_Try(self, tree: ast.Try) -> None: ...

    def visit_While(self, tree: ast.While) -> None: ...

    def generic_visit(self, stmt: ast.AST) -> None: ...

    def build_model(self) -> workflow_model.WorkflowNode: ...
