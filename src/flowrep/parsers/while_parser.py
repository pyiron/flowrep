from __future__ import annotations

import ast
from ast import While

from flowrep import edge_models
from flowrep.parsers import (
    attribute_parser,
    case_helpers,
    parser_helpers,
    parser_protocol,
    symbol_scope,
)
from flowrep.prospective import helper_models, while_recipe

WHILE_CONDITION_LABEL: str = "condition"
WHILE_BODY_LABEL: str = "body"


def parse_while_node(
    walker: parser_protocol.BodyWalker, tree: ast.While
) -> tuple[while_recipe.WhileRecipe, parser_helpers.FlowControlBindings]:
    """
    Walk a while-loop.

    Args:
        walker: A walker to fork and use for collecting state inside the tree.
        tree: The ``ast.While`` node.
    """
    _validate_syntax_is_supported(tree)

    # Parse the loop condition — pure AST, no parser state needed
    labeled_condition, condition_inputs, condition_bindings = case_helpers.parse_case(
        tree.test,
        walker.scope,
        walker.symbol_map,
        walker.info_factory,
        WHILE_CONDITION_LABEL,
        walker.nodes,
    )

    body_walker = walker.fork(
        new_symbol_map=walker.symbol_map.fork(),
        new_scope=walker.scope,
    )
    body_walker.walk(tree.body)
    reassigned_symbols = body_walker.symbol_map.reassigned_symbols

    _validate_some_output_exists(reassigned_symbols)
    _reject_looped_attribute_roots(tree.test, walker.symbol_map, reassigned_symbols)
    parser_helpers.reject_input_alias_outputs(
        body_walker.symbol_map, reassigned_symbols, "while-loop"
    )
    body_walker.symbol_map.produce_symbols(reassigned_symbols)

    inputs, input_edges = _wire_inputs(
        body_walker, condition_inputs, reassigned_symbols
    )
    outputs, output_edges = _wire_outputs(body_walker)

    case = helper_models.ConditionalCase(
        condition=labeled_condition,
        body=helper_models.LabeledRecipe(
            label=WHILE_BODY_LABEL, recipe=body_walker.build_model()
        ),
    )

    return (
        while_recipe.WhileRecipe(
            inputs=inputs,
            outputs=outputs,
            case=case,
            input_edges=input_edges,
            output_edges=output_edges,
        ),
        condition_bindings,
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


def _reject_looped_attribute_roots(
    test: ast.expr,
    symbol_map: symbol_scope.SymbolScope,
    reassigned_symbols: list[str],
) -> None:
    """Raise if a condition attribute chain is rooted at a symbol the body reassigns.

    Python re-evaluates a while condition every iteration, so ``x.val`` is re-read
    against the *updated* ``x``. A flowrep condition is a single call fed by hoisted
    inputs: its getattr peer sits outside the loop, is never re-read, and is not a
    while output, so it never feeds back. Rather than silently diverge from Python we
    refuse. An attribute on a symbol the loop does not touch hoists faithfully and is
    allowed -- which is why the guard is on the *root*, not on attribute access.

    Runs after the body walk, because ``reassigned_symbols`` is the parser's ground
    truth (it includes symbols reassigned by nested flow control, not just bare
    assignments). By then the getattr peers have already been injected into the
    enclosing scope; that is harmless, since the exception aborts the whole parse.
    """
    if not isinstance(
        test, ast.Call
    ):  # pragma: no cover - parse_case rejects non-calls
        return
    looped = set(reassigned_symbols)
    arguments = list(test.args) + [kw.value for kw in test.keywords]
    for argument in arguments:
        if not attribute_parser.is_data_attribute(argument, symbol_map):
            continue
        root = attribute_parser.chain_root(argument)
        if root is not None and root.id in looped:
            chain = ast.unparse(argument)
            raise ValueError(
                f"While-condition attribute access {chain!r} is rooted at "
                f"{root.id!r}, which the loop body reassigns. Python would re-read "
                f"{chain!r} every iteration, but a flowrep while-condition is a "
                f"single call fed by hoisted inputs -- there is no place inside the "
                f"loop for the attribute access. Either bind it outside the loop "
                f"(e.g. `v = {chain}`) if you meant to read it once, or move the "
                f"attribute access into the condition function itself."
            )


def _wire_inputs(
    body_walker: parser_protocol.BodyWalker,
    condition_inputs: edge_models.InputEdges,
    reassigned_symbols: list[str],
) -> tuple[list[str], edge_models.InputEdges]:
    inputs = [source.port for source in condition_inputs.values()]
    input_edges = dict(condition_inputs)
    for port in body_walker.inputs:
        input_edges[edge_models.TargetHandle(node=WHILE_BODY_LABEL, port=port)] = (
            edge_models.InputSource(port=port)
        )
        if port not in inputs:
            inputs.append(port)

    # Catch symbols that are reassigned internally, but not used as input to the body
    # or condition
    for symbol in reassigned_symbols:
        if symbol not in inputs:
            inputs.append(symbol)
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
