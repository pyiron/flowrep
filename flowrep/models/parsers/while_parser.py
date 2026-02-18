from __future__ import annotations

import ast
from collections.abc import Callable
from typing import ClassVar

from flowrep.models import edge_models
from flowrep.models.nodes import helper_models, while_model
from flowrep.models.parsers import (
    atomic_parser,
    object_scope,
    parser_helpers,
    parser_protocol,
    symbol_scope,
)


class WhileParser:
    condition_label: ClassVar[str] = "condition"
    body_label: ClassVar[str] = "body"

    def __init__(self) -> None:
        self._body_walker: parser_protocol.BodyWalker | None = None
        self._labeled_condition_node: helper_models.LabeledNode | None = None
        self._condition_inputs: edge_models.InputEdges = {}

        self._inputs: list[str] = []
        self._outputs: list[str] = []
        self._input_edges: edge_models.InputEdges = {}
        self._output_edges: edge_models.OutputEdges = {}

    @property
    def _case(self) -> helper_models.ConditionalCase:
        if self._body_walker is None or self._labeled_condition_node is None:
            raise ValueError(
                "WhileParser does not have the data to build a case. Please "
                "`build_body` first."
            )
        return helper_models.ConditionalCase(
            condition=self._labeled_condition_node,
            body=helper_models.LabeledNode(
                label=self.body_label, node=self._body_walker.build_model()
            ),
        )

    def build_model(self) -> while_model.WhileNode:
        return while_model.WhileNode(
            inputs=self._inputs,
            outputs=self._outputs,
            case=self._case,
            input_edges=self._input_edges,
            output_edges=self._output_edges,
        )

    def build_body(
        self,
        tree: ast.While,
        scope: object_scope.ScopeProxy,
        symbol_map: symbol_scope.SymbolScope,
        walker_factory: Callable[
            [symbol_scope.SymbolScope], parser_protocol.BodyWalker
        ],
    ) -> None:
        """
        Walk a while-loop, building all internal state needed for
        :meth:`build_model`.

        Args:
            tree: The ``ast.While`` node.
            scope: Object-level scope for resolving callable references.
            symbol_map: The enclosing :class:`SymbolScope` (used for forking).
            walker_factory: Callable that creates a :class:`BodyWalker` from a
                :class:`SymbolScope`.  Avoids a circular import with
                ``workflow_parser.WorkflowParser``.
        """
        # 0. Fail early for unsupported syntax
        if tree.orelse:
            raise NotImplementedError(
                "While loops with else branches are not supported in our parsing "
                "syntax."
            )

        # 1. Parse the loop condition — pure AST, no parser state needed
        labeled_condition, condition_inputs = parse_while_condition(
            tree, scope, symbol_map
        )
        self._labeled_condition_node = helper_models.LabeledNode(
            label=self.condition_label,
            node=labeled_condition.node,
        )
        self._condition_inputs = edge_models.InputEdges(
            {
                edge_models.TargetHandle(
                    node=self.condition_label, port=target.port
                ): source
                for target, source in condition_inputs.items()
            }
        )

        # 2. Fork scope — carry_accumulators=False: the while-node model does
        #    not support accumulation across iterations, so outer accumulators
        #    must not leak into the while body.
        body_symbol_map = symbol_map.fork_scope({})

        # 3. Fresh body walker with the forked scope
        body_walker = walker_factory(body_symbol_map)

        body_walker.walk(tree.body, scope)
        reassigned_symbols = body_walker.symbol_map.reassigned_symbols
        if len(reassigned_symbols) == 0:
            raise ValueError(
                "While-loop body must reassign at least one symbol from the "
                "enclosing scope."
            )

        for symbol in reassigned_symbols:
            body_walker.symbol_map.produce(symbol, symbol)

        self._body_walker = body_walker
        self._inputs = [source.port for source in self._condition_inputs.values()]
        self._input_edges = dict(self._condition_inputs)
        for port in body_walker.inputs:
            self._input_edges[
                edge_models.TargetHandle(node=self.body_label, port=port)
            ] = edge_models.InputSource(port=port)
            if port not in self._inputs:
                self._inputs.append(port)

        self._outputs = reassigned_symbols
        self._output_edges = edge_models.OutputEdges(
            {
                edge_models.OutputTarget(port=symbol): edge_models.SourceHandle(
                    node=self.body_label, port=symbol
                )
                for symbol in reassigned_symbols
            }
        )


def parse_while_condition(
    while_stmt: ast.While,
    scope: object_scope.ScopeProxy,
    symbol_map: symbol_scope.SymbolScope,
) -> tuple[helper_models.LabeledNode, edge_models.InputEdges]:
    test_expr = while_stmt.test
    if not isinstance(test_expr, ast.Call):
        raise ValueError(
            "While-loop conditions must be a function call, but got "
            f"{type(test_expr).__name__}"
        )

    condition_node = atomic_parser.get_labeled_recipe(test_expr, set(), scope)
    if len(condition_node.node.outputs) != 1:
        raise ValueError(
            f"While-loop condition must return exactly one value (and it had better be "
            f"truthy), but got {condition_node.node.outputs}"
        )

    scope_copy = symbol_map.fork_scope({})
    parser_helpers.consume_call_arguments(scope_copy, test_expr, condition_node)
    condition_inputs = scope_copy.input_edges
    return condition_node, condition_inputs
