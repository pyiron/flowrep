from __future__ import annotations

import ast
from collections.abc import Callable

from flowrep.models import edge_models
from flowrep.models.nodes import helper_models, while_model
from flowrep.models.parsers import (
    atomic_parser,
    object_scope,
    parser_helpers,
    parser_protocol,
    symbol_scope,
)
from flowrep.models.parsers.parser_protocol import BodyWalker

WHILE_CONDITION_LABEL: str = "condition"
WHILE_BODY_LABEL: str = "body"


def parse_while_node(
    tree: ast.While,
    scope: object_scope.ScopeProxy,
    symbol_map: symbol_scope.SymbolScope,
    walker_factory: Callable[[symbol_scope.SymbolScope], parser_protocol.BodyWalker],
) -> while_model.WhileNode:
    """
    Walk a while-loop.

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
            "While loops with else branches are not supported in our parsing " "syntax."
        )

    # 1. Parse the loop condition — pure AST, no parser state needed
    labeled_condition, condition_inputs = _parse_while_condition(
        tree, scope, symbol_map
    )
    labeled_condition_node = helper_models.LabeledNode(
        label=WHILE_CONDITION_LABEL,
        node=labeled_condition.node,
    )
    condition_inputs = edge_models.InputEdges(
        {
            edge_models.TargetHandle(
                node=WHILE_CONDITION_LABEL, port=target.port
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

    inputs, input_edges = _wire_inputs(body_walker, condition_inputs)
    outputs, output_edges = _wire_outputs(body_walker)

    case = helper_models.ConditionalCase(
        condition=labeled_condition_node,
        body=helper_models.LabeledNode(
            label=WHILE_BODY_LABEL, node=body_walker.build_model()
        ),
    )

    return while_model.WhileNode(
        inputs=inputs,
        outputs=outputs,
        case=case,
        input_edges=input_edges,
        output_edges=output_edges,
    )


def _wire_inputs(
    body_walker: BodyWalker, condition_inputs: edge_models.InputEdges
) -> tuple[list[str], edge_models.InputEdges]:
    inputs = [source.port for source in condition_inputs.values()]
    input_edges = dict(condition_inputs)
    for port in body_walker.inputs:
        input_edges[edge_models.TargetHandle(node=WHILE_BODY_LABEL, port=port)] = (
            edge_models.InputSource(port=port)
        )
        if port not in inputs:
            inputs.append(port)
    return inputs, input_edges


def _wire_outputs(body_walker: BodyWalker) -> tuple[list[str], edge_models.OutputEdges]:
    reassigned_symbols = body_walker.symbol_map.reassigned_symbols
    outputs = reassigned_symbols
    output_edges = edge_models.OutputEdges(
        {
            edge_models.OutputTarget(port=symbol): edge_models.SourceHandle(
                node=WHILE_BODY_LABEL, port=symbol
            )
            for symbol in reassigned_symbols
        }
    )
    return outputs, output_edges


def _parse_while_condition(
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
