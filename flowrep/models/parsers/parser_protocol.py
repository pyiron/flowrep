import ast
from collections.abc import Collection
from types import FunctionType
from typing import Protocol, runtime_checkable

from flowrep.models import edge_models
from flowrep.models.nodes import union, workflow_model
from flowrep.models.parsers import object_scope, symbol_scope


@runtime_checkable
class BodyWalker(Protocol):
    """What control flow parsers need to walk a sub-body."""

    symbol_scope: symbol_scope.SymbolScope
    outputs: list[str]
    nodes: union.Nodes
    output_edges: edge_models.OutputEdges

    @property
    def inputs(self) -> list[str]: ...

    @property
    def input_edges(self) -> edge_models.InputEdges: ...

    @property
    def edges(self) -> edge_models.Edges: ...

    def handle_assign(
        self, body: ast.Assign | ast.AnnAssign, scope: object_scope.ScopeProxy
    ) -> None: ...

    def handle_for(
        self, tree: ast.For, scope: object_scope.ScopeProxy, parsing_function_def: bool
    ) -> None: ...

    def handle_while(self, tree: ast.While, scope: object_scope.ScopeProxy) -> None: ...

    def handle_appending_to_accumulator(self, stmt: ast.Expr) -> tuple[str, str]: ...

    def handle_return(
        self,
        body: ast.Return,
        func: FunctionType,
        output_labels: Collection[str],
    ) -> None: ...

    def build_model(self) -> workflow_model.WorkflowNode: ...
