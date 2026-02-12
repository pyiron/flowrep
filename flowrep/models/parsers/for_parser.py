from __future__ import annotations

import ast
from typing import ClassVar

from flowrep.models import edge_models
from flowrep.models.nodes import for_model, helper_models
from flowrep.models.parsers import object_scope, parser_protocol


def walk_ast_for(
    body_walker: parser_protocol.BodyWalker,
    tree: ast.For,
    scope: object_scope.ScopeProxy,
) -> None:
    for body in tree.body:
        body_walker.visit(body, scope)


class ForParser:
    body_label: ClassVar[str] = "body"

    def __init__(self, body_walker: parser_protocol.BodyWalker):
        self.body_walker = body_walker

        # When these are all filled, we are ready to `build_model`
        self._inputs: list[str] = []
        self._input_edges: edge_models.InputEdges = {}
        # Note property: self._body_node
        self._output_edges: edge_models.OutputEdges = {}
        self._outputs: list[str] = []
        self._nested_ports: list[str] = []
        self._zipped_ports: list[str] = []

    @property
    def _body_node(self) -> helper_models.LabeledNode:
        return helper_models.LabeledNode(
            label="body", node=self.body_walker.build_model()
        )

    def build_model(self) -> for_model.ForNode:
        return for_model.ForNode(
            inputs=self._inputs,
            outputs=self._outputs,
            body_node=self._body_node,
            input_edges=self._input_edges,
            output_edges=self._output_edges,
            nested_ports=self._nested_ports,
            zipped_ports=self._zipped_ports,
        )

    def build_body(
        self,
        tree: ast.For,
        scope: object_scope.ScopeProxy,
        nested_iters: list[tuple[str, str]],
        zipped_iters: list[tuple[str, str]],
    ) -> None:
        all_iters = nested_iters + zipped_iters

        walk_ast_for(self.body_walker, tree, scope)
        if len(self.body_walker.symbol_scope.used_accumulator_map) == 0:
            raise ValueError("For nodes must use up at least one accumulator symbol.")
        used_accumulator_symbol_map = self.body_walker.symbol_scope.used_accumulator_map

        # Every iteration variable must actually be consumed inside the body.
        # An unused iterator likely indicates a bug; if the user only needs the
        # structural effect (e.g. repetition count), they should make the
        # dependency explicit.
        iterating_symbols = {var for var, _ in all_iters}
        consumed_symbols = set(self.body_walker.inputs) | set(
            used_accumulator_symbol_map.values()
        )
        if unused := iterating_symbols - consumed_symbols:
            raise ValueError(
                f"For-node iteration variable(s) {sorted(unused)} are never "
                f"used inside the node body. Either use them or remove them "
                f"from the iteration header."
            )

        broadcast_symbols = [
            s
            for s in self.body_walker.inputs
            if s not in set(used_accumulator_symbol_map.values())
            and s not in {iterating_symbol for iterating_symbol, _ in all_iters}
        ]  # Need to keep it consistently ordered, so don't use a simple set op
        scattered_symbols = [scattered_symbol for _, scattered_symbol in all_iters]

        self._inputs = broadcast_symbols + scattered_symbols
        self._outputs = list(used_accumulator_symbol_map)
        self._nested_ports = [var for var, _ in nested_iters]
        self._zipped_ports = [var for var, _ in zipped_iters]

        broadcast_inputs = {
            edge_models.TargetHandle(
                node=self.body_label, port=port
            ): edge_models.InputSource(port=port)
            for port in broadcast_symbols
        }
        scattered_inputs = {
            edge_models.TargetHandle(
                node=self.body_label, port=body_port
            ): edge_models.InputSource(port=for_port)
            for body_port, for_port in all_iters
        }
        self._input_edges = broadcast_inputs | scattered_inputs

        self._output_edges = {}
        for accumulator_symbol, appended_symbol in used_accumulator_symbol_map.items():
            target = edge_models.OutputTarget(port=accumulator_symbol)
            if appended_symbol in self.body_walker.outputs:
                self._output_edges[target] = edge_models.SourceHandle(
                    node=self.body_label, port=appended_symbol
                )
            else:
                self._output_edges[target] = self._input_edges[
                    edge_models.TargetHandle(node=self.body_label, port=appended_symbol)
                ]

        return None


def parse_for_iterations(
    for_stmt: ast.For,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], ast.For]:
    """
    Parse for-node iteration structure, handling zip and immediately nested iterations.

    Returns (nested_iterations, zipped_iterations) where each is a list of
    (variable_name, source_symbol) tuples.
    """
    nested: list[tuple[str, str]] = []
    zipped: list[tuple[str, str]] = []

    current = for_stmt
    while isinstance(current, ast.For):
        is_zip, pairs = _parse_single_for_header(current)

        if is_zip:
            zipped.extend(pairs)
        else:
            nested.extend(pairs)

        # Check for nested for-declaration (single statement that's another For)
        if len(current.body) >= 1 and isinstance(current.body[0], ast.For):
            current = current.body[0]
        else:
            break

    return nested, zipped, current


def _parse_single_for_header(
    for_stmt: ast.For,
) -> tuple[bool, list[tuple[str, str]]]:
    """
    Parse a single for-header.

    Returns (is_zipped, [(var, source), ...]).
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

        sources = []
        for arg in iter_expr.args:
            if not isinstance(arg, ast.Name):
                raise ValueError("zip() arguments must be simple symbols")
            sources.append(arg.id)

        if len(vars_list) != len(sources):
            raise ValueError(
                f"zip() variable count ({len(vars_list)}) must match "
                f"argument count ({len(sources)})"
            )

        return True, list(zip(vars_list, sources, strict=True))

    # Simple iteration: for x in xs
    if not isinstance(iter_expr, ast.Name):
        raise ValueError(
            "For iteration must iterate over a symbol (not an inline expression)"
        )

    if isinstance(target, ast.Name):
        return False, [(target.id, iter_expr.id)]
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
