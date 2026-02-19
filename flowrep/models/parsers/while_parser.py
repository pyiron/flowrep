from __future__ import annotations

import ast
from ast import While

from flowrep.models import edge_models
from flowrep.models.nodes import helper_models, while_model
from flowrep.models.parsers import (
    case_helpers,
    object_scope,
    parser_protocol,
    symbol_scope,
)

WHILE_CONDITION_LABEL: str = "condition"
WHILE_BODY_LABEL: str = "body"


def parse_while_node(
    tree: ast.While,
    scope: object_scope.ScopeProxy,
    symbol_map: symbol_scope.SymbolScope,
    walker_factory: parser_protocol.WalkerFactory,
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
    _validate_syntax_is_supported(tree)

    # Parse the loop condition — pure AST, no parser state needed
    labeled_condition, condition_inputs = case_helpers.parse_case(
        tree.test, scope, symbol_map, WHILE_CONDITION_LABEL
    )

    body_walker = walker_factory(symbol_map.fork_scope())
    body_walker.walk(tree.body, scope)
    reassigned_symbols = body_walker.symbol_map.reassigned_symbols

    _validate_some_output_exists(reassigned_symbols)
    body_walker.symbol_map.produce_symbols(reassigned_symbols)

    inputs, input_edges = _wire_inputs(body_walker, condition_inputs)
    outputs, output_edges = _wire_outputs(body_walker)

    case = helper_models.ConditionalCase(
        condition=labeled_condition,
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


def _validate_syntax_is_supported(tree: While):
    if tree.orelse:
        raise NotImplementedError(
            "While loops with else branches are not supported in our parsing " "syntax."
        )


def _validate_some_output_exists(reassigned_symbols: list[str]):
    if len(reassigned_symbols) == 0:
        raise ValueError(
            "While-loop body must reassign at least one symbol from the "
            "enclosing scope."
        )


def _wire_inputs(
    body_walker: parser_protocol.BodyWalker, condition_inputs: edge_models.InputEdges
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


def _wire_outputs(
    body_walker: parser_protocol.BodyWalker,
) -> tuple[list[str], edge_models.OutputEdges]:
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
