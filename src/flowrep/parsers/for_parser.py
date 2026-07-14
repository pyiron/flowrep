from __future__ import annotations

import ast
from typing import NamedTuple

from flowrep import edge_models
from flowrep.parsers import (
    attribute_parser,
    parser_helpers,
    parser_protocol,
    symbol_scope,
)
from flowrep.prospective import for_recipe, helper_models, union_types

FOR_BODY_LABEL: str = "body"


class _IterationAxis(NamedTuple):
    """One axis of `for x in xs`.

    *variable* is the loop variable (a *body* input port). *port* is the for-node
    input port feeding it: a bare symbol lends its own name, an attribute chain gets a
    generated one. *binding* is the getattr peer the enclosing walker must wire into
    *port*, or ``None`` when the collection was a plain symbol.
    """

    variable: str
    port: str
    binding: edge_models.SourceHandle | None = None


AccumulatorMap = dict[str, str]
"""
Maps accumulator names, xs, to appended symbol names, x, in statements like xs.append(x)
"""


def parse_for_node(
    walker: parser_protocol.BodyWalker, tree: ast.For
) -> tuple[for_recipe.ForEachRecipe, parser_helpers.FlowControlBindings]:
    """
    Walk a for-loop.

    Args:
        walker: A walker to fork and use for collecting state inside the tree.
        tree: The top-level ``ast.For`` node (may contain immediately
            nested for-headers that declare additional iteration axes).
    """
    # Parse the iteration header — pure AST plus attribute-chain injection into the
    # enclosing walker's own nodes/scope
    nested_iters, zipped_iters, body_tree = _parse_for_iterations(
        tree, walker.symbol_map, walker.nodes
    )
    all_iters = nested_iters + zipped_iters

    # When we fork the scope here, we replace iterated-over symbols with iteration
    # variables, all as InputSources from the body's perspective. An attribute-chain
    # axis has no parent symbol to remap, so its variable is added fresh instead.
    body_symbol_map = walker.symbol_map.fork(
        {axis.port: axis.variable for axis in all_iters if axis.binding is None},
        added_symbols=[axis.variable for axis in all_iters if axis.binding is not None],
        available_accumulators=walker.symbol_map.declared_accumulators.copy(),
    )

    body_walker = walker.fork(new_symbol_map=body_symbol_map, new_scope=walker.scope)
    body_walker.walk(body_tree.body)
    consumed = body_walker.symbol_map.consumed_accumulators

    _validate_some_output_exists(consumed)
    _validate_no_unused_iterators(all_iters, body_walker, consumed)
    _validate_no_leaked_reassignments(
        all_iters, body_walker, consumed, walker.symbol_map
    )

    nested_ports = [axis.variable for axis in nested_iters]
    zipped_ports = [axis.variable for axis in zipped_iters]

    inputs, input_edges = _wire_inputs(body_walker, all_iters)
    outputs, output_edges = _wire_outputs(body_walker, input_edges)

    body_node = helper_models.LabeledRecipe(
        label=FOR_BODY_LABEL, recipe=body_walker.build_model()
    )

    bindings: parser_helpers.FlowControlBindings = {
        axis.port: axis.binding for axis in all_iters if axis.binding is not None
    }

    return (
        for_recipe.ForEachRecipe(
            inputs=inputs,
            outputs=outputs,
            body_node=body_node,
            input_edges=input_edges,
            output_edges=output_edges,
            nested_ports=nested_ports,
            zipped_ports=zipped_ports,
        ),
        bindings,
    )


def _validate_some_output_exists(consumed: AccumulatorMap):
    if len(consumed) == 0:
        raise ValueError("For nodes must use up at least one accumulator symbol.")


def _validate_no_unused_iterators(
    all_iters: list[_IterationAxis],
    body_walker: parser_protocol.BodyWalker,
    consumed: AccumulatorMap,
):
    """
    Every iteration variable must actually be consumed inside the body.
    An unused iterator likely indicates a bug; if the user only needs the structural
    effect (e.g. repetition count), they should make the dependency explicit.
    """
    iterating_symbols = {axis.variable for axis in all_iters}
    consumed_symbols = set(body_walker.inputs) | set(consumed.values())
    if unused := iterating_symbols - consumed_symbols:
        raise ValueError(
            f"For-node iteration variable(s) {sorted(unused)} are never "
            f"used inside the node body. Either use them or remove them "
            f"from the iteration header."
        )


def _validate_no_leaked_reassignments(
    all_iters: list[_IterationAxis],
    body_walker: parser_protocol.BodyWalker,
    consumed: AccumulatorMap,
    symbol_map: symbol_scope.SymbolScope,
):
    """
    Check for internal symbol reassignments that would leak to un-captured outputs --
    the only outputs we allow from a for node are iterated outputs!
    """
    body_reassigned = set(body_walker.symbol_map.reassigned_symbols)
    accumulator_outputs = set(consumed)
    unreturned_reassignments = (
        body_reassigned - accumulator_outputs - {axis.variable for axis in all_iters}
    )
    leaked_reassignments = unreturned_reassignments.intersection(symbol_map.keys())
    if leaked_reassignments:
        raise ValueError(
            f"For-loop body reassigns symbol(s) {sorted(leaked_reassignments)} "
            f"from the enclosing scope. This is not supported because for-node "
            f"outputs are determined by accumulators. If you need the reassigned "
            f"value after the loop, accumulate it explicitly."
        )


def _wire_inputs(
    body_walker: parser_protocol.BodyWalker, all_iters: list[_IterationAxis]
) -> tuple[list[str], edge_models.InputEdges]:
    consumed = body_walker.symbol_map.consumed_accumulators
    broadcast_symbols = [
        s
        for s in body_walker.inputs
        if s not in set(consumed.values())
        and s not in {axis.variable for axis in all_iters}
    ]  # Need to keep it consistently ordered, so don't use a simple set op
    scattered_symbols = [axis.port for axis in all_iters]
    inputs = broadcast_symbols + scattered_symbols
    broadcast_inputs = {
        edge_models.TargetHandle(
            node=FOR_BODY_LABEL, port=port
        ): edge_models.InputSource(port=port)
        for port in broadcast_symbols
    }
    scattered_inputs = {
        edge_models.TargetHandle(
            node=FOR_BODY_LABEL, port=axis.variable
        ): edge_models.InputSource(port=axis.port)
        for axis in all_iters
    }
    input_edges = broadcast_inputs | scattered_inputs
    return inputs, input_edges


def _wire_outputs(
    body_walker: parser_protocol.BodyWalker, input_edges: edge_models.InputEdges
) -> tuple[list[str], edge_models.OutputEdges]:
    consumed = body_walker.symbol_map.consumed_accumulators
    outputs = list(consumed)
    output_edges: edge_models.OutputEdges = {}
    for accumulator_symbol, appended_symbol in consumed.items():
        target = edge_models.OutputTarget(port=accumulator_symbol)
        if appended_symbol in body_walker.outputs:
            output_edges[target] = edge_models.SourceHandle(
                node=FOR_BODY_LABEL, port=appended_symbol
            )
        else:
            output_edges[target] = input_edges[
                edge_models.TargetHandle(node=FOR_BODY_LABEL, port=appended_symbol)
            ]
    return outputs, output_edges


def _resolve_collection(
    iter_expr: ast.expr,
    symbol_map: symbol_scope.SymbolScope,
    nodes: union_types.Recipes,
    reserved_ports: set[str],
) -> tuple[str, edge_models.SourceHandle | None]:
    """The for-node input port that feeds one iteration axis, plus any peer to wire.

    A for recipe has no room to host a peer, so an attribute chain becomes a getattr
    peer of the for node in the enclosing scope and reaches it through a generated
    port. See :func:`attribute_parser.generate_port_name` for the naming rule.
    """
    if isinstance(iter_expr, ast.Name):
        return iter_expr.id, None
    if attribute_parser.is_data_attribute(iter_expr, symbol_map):
        handle = attribute_parser.inject_attribute_chain(iter_expr, symbol_map, nodes)
        port = attribute_parser.generate_port_name(
            iter_expr, symbol_map.unavailable_names | reserved_ports
        )
        reserved_ports.add(port)
        return port, handle
    raise ValueError(
        f"For iteration must iterate over a symbol, or an attribute of one, but got "
        f"'{ast.unparse(iter_expr)}'."
    )


def _parse_for_iterations(
    for_stmt: ast.For,
    symbol_map: symbol_scope.SymbolScope,
    nodes: union_types.Recipes,
) -> tuple[list[_IterationAxis], list[_IterationAxis], ast.For]:
    """
    Parse for-node iteration structure, handling zip and immediately nested iterations.

    Returns (nested_iterations, zipped_iterations, innermost_for_tree).
    """
    nested: list[_IterationAxis] = []
    zipped: list[_IterationAxis] = []
    reserved_ports: set[str] = set()

    current = for_stmt
    while isinstance(current, ast.For):
        is_zip, axes = _parse_single_for_header(
            current, symbol_map, nodes, reserved_ports
        )

        if is_zip:
            zipped.extend(axes)
        else:
            nested.extend(axes)

        # Check for nested for-declaration (single statement that's another For)
        if len(current.body) >= 1 and isinstance(current.body[0], ast.For):
            current = current.body[0]
        else:
            break

    return nested, zipped, current


def _parse_single_for_header(
    for_stmt: ast.For,
    symbol_map: symbol_scope.SymbolScope,
    nodes: union_types.Recipes,
    reserved_ports: set[str],
) -> tuple[bool, list[_IterationAxis]]:
    """
    Parse a single for-header.

    Returns (is_zipped, axes).
    """
    iter_expr = for_stmt.iter
    target = for_stmt.target

    # Check for zip()
    if isinstance(iter_expr, ast.Call) and _is_zip_call(iter_expr):
        if not isinstance(target, ast.Tuple):
            raise ValueError("zip() iteration requires tuple unpacking")

        vars_list = [elt.id for elt in target.elts if isinstance(elt, ast.Name)]
        if len(vars_list) != len(target.elts):
            raise ValueError("zip() iteration targets must be simple names")

        if len(vars_list) != len(iter_expr.args):
            raise ValueError(
                f"zip() variable count ({len(vars_list)}) must match "
                f"argument count ({len(iter_expr.args)})"
            )

        axes = []
        for variable, arg in zip(vars_list, iter_expr.args, strict=True):
            port, binding = _resolve_collection(arg, symbol_map, nodes, reserved_ports)
            axes.append(_IterationAxis(variable, port, binding))
        return True, axes

    port, binding = _resolve_collection(iter_expr, symbol_map, nodes, reserved_ports)

    if isinstance(target, ast.Name):
        return False, [_IterationAxis(target.id, port, binding)]
    elif isinstance(target, ast.Tuple):
        # for a, b in items (tuple unpacking without zip)
        raise ValueError(
            "Tuple unpacking in for-nodes requires zip(). "
            "Use 'for a, b in zip(as, bs):' instead of 'for a, b in items:'"
        )
    else:
        raise ValueError(f"Unsupported for iteration target: {type(target)}")


def _is_zip_call(node: ast.Call) -> bool:
    """Check if a Call node is a call to zip()."""
    return isinstance(node.func, ast.Name) and node.func.id == "zip"
