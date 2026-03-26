from __future__ import annotations

import ast
from typing import Protocol, runtime_checkable

from pyiron_snippets import versions

from flowrep import edge_models
from flowrep.nodes import union, workflow_model
from flowrep.parsers import object_scope, symbol_scope


@runtime_checkable
class BodyWalker(Protocol):
    """What control flow parsers need to walk a sub-body."""

    scope: object_scope.ScopeProxy
    symbol_map: symbol_scope.SymbolScope
    info_factory: versions.VersionInfoFactory
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

    def build_model(self) -> workflow_model.WorkflowNode: ...

    def fork(
        self,
        *,
        new_symbol_map: symbol_scope.SymbolScope,
        new_scope: object_scope.ScopeProxy,
    ) -> BodyWalker: ...

    def walk(self, statements: list[ast.stmt]) -> None: ...

    def visit(self, stmt: ast.AST) -> None: ...
