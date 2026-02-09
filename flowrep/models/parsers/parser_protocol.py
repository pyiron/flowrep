import ast
from typing import Protocol, runtime_checkable

from flowrep.models import edge_models
from flowrep.models.nodes import union, workflow_model
from flowrep.models.parsers import scope_helpers, symbol_scope


@runtime_checkable
class BodyWalker(Protocol):
    """What control flow parsers need to walk a sub-body."""

    symbol_to_source_map: symbol_scope.SymbolScope
    inputs: list[str]
    outputs: list[str]
    nodes: union.Nodes
    input_edges: edge_models.InputEdges
    edges: edge_models.Edges
    output_edges: edge_models.OutputEdges

    def handle_assign(
        self, body: ast.Assign | ast.AnnAssign, scope: scope_helpers.ScopeProxy
    ) -> None: ...

    def handle_for(
        self, tree: ast.For, scope: scope_helpers.ScopeProxy, parsing_function_def: bool
    ) -> None: ...

    def handle_appending_to_accumulator(
        self, stmt: ast.Expr, accumulators: set[str]
    ) -> tuple[str, str]: ...

    def build_model(self) -> workflow_model.WorkflowNode: ...
