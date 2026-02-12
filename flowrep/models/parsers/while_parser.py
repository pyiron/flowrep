from __future__ import annotations

import ast
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


def walk_ast_while(
    body_walker: parser_protocol.BodyWalker,
    tree: ast.While,
    scope: object_scope.ScopeProxy,
) -> None:
    for body in tree.body:
        body_walker.visit(body, scope)


class WhileParser:
    condition_label: ClassVar[str] = "condition"
    body_label: ClassVar[str] = "body"

    def __init__(
        self,
        body_walker: parser_protocol.BodyWalker,
        labeled_condition: helper_models.LabeledNode,
        condition_inputs: edge_models.InputEdges,
    ):
        self.body_walker = body_walker
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

        self._inputs: list[str] = []
        self._outputs: list[str] = []
        # Note property: self._case
        self._input_edges: edge_models.InputEdges = {}
        self._output_edges: edge_models.OutputEdges = {}

    @property
    def _case(self) -> helper_models.ConditionalCase:
        return helper_models.ConditionalCase(
            condition=self._labeled_condition_node,
            body=helper_models.LabeledNode(
                label=self.body_label, node=self.body_walker.build_model()
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

    def build_body(self, tree: ast.While, scope: object_scope.ScopeProxy) -> None:

        walk_ast_while(self.body_walker, tree, scope)
        reassigned_symbols = self.body_walker.symbol_scope.reassigned_symbols
        if len(reassigned_symbols) == 0:
            raise ValueError(
                "While-loop body must reassign at least one symbol from the "
                "enclosing scope."
            )

        for symbol in reassigned_symbols:
            source = self.body_walker.symbol_scope[symbol]
            self.body_walker.outputs.append(symbol)
            self.body_walker.output_edges[edge_models.OutputTarget(port=symbol)] = (
                source
            )

        self._inputs = [source.port for source in self._condition_inputs.values()]
        self._input_edges = self._condition_inputs
        for port in self.body_walker.inputs:
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
    symbol_scope: symbol_scope.SymbolScope,
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

    scope_copy = symbol_scope.fork_scope({})
    parser_helpers.consume_call_arguments(scope_copy, test_expr, condition_node)
    condition_inputs = scope_copy.input_edges
    return condition_node, condition_inputs
