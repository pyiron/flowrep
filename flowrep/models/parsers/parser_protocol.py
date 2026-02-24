from __future__ import annotations

import ast
from collections.abc import Callable, Collection
from types import FunctionType
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

    def visit(self, stmt: ast.stmt) -> None: ...

    def walk(self, statements: list[ast.stmt]) -> None: ...

    def handle_assign(self, body: ast.Assign | ast.AnnAssign) -> None: ...

    def handle_for(self, tree: ast.For) -> None: ...

    def handle_if(self, tree: ast.If) -> None: ...

    def handle_try(self, tree: ast.Try) -> None: ...

    def handle_while(self, tree: ast.While) -> None: ...

    def handle_appending_to_accumulator(self, append_call: ast.Call) -> None: ...

    def handle_return(
        self,
        body: ast.Return,
        func: FunctionType,
        output_labels: Collection[str],
    ) -> None: ...

    def build_model(self) -> workflow_model.WorkflowNode: ...
