from __future__ import annotations

import ast
import dataclasses
from typing import ClassVar

from flowrep.models import base_models, edge_models
from flowrep.models.nodes import for_model, helper_models
from flowrep.models.parsers import parser_protocol, scope_helpers


def walk_ast_for(
    body_walker: parser_protocol.BodyWalker,
    tree: ast.For,
    scope: scope_helpers.ScopeProxy,
    accumulators: set[str],
) -> dict[str, str]:
    used_accumulators: list[str] = []
    used_accumulator_source_map: dict[str, str] = {}

    for body in tree.body:
        if isinstance(body, ast.Assign | ast.AnnAssign):
            body_walker.handle_assign(body, scope)
        elif isinstance(body, ast.For):
            body_walker.handle_for(body, scope, parsing_function_def=False)
        elif isinstance(body, ast.While | ast.If | ast.Try):
            raise NotImplementedError(
                f"Support for control flow statement {type(body)} is forthcoming."
            )
        elif isinstance(body, ast.Expr):
            used_accumulator, appended_symbol = (
                body_walker.handle_appending_to_accumulator(body, accumulators)
            )
            used_accumulators.append(used_accumulator)
            used_accumulator_source_map[used_accumulator] = appended_symbol
        else:
            raise TypeError(
                f"Workflow python definitions can only interpret assignments, a subset "
                f"of flow control (for/while/if/try) and a return, but ast found "
                f"{type(body)}"
            )

    if len(used_accumulators) == 0:
        raise ValueError("For loops must use up at least one accumulator symbol.")
    base_models.validate_unique(
        used_accumulators,
        f"Each accumulator may be appended to at most once, but appended "
        f"to: {used_accumulators}",
    )

    return used_accumulator_source_map


@dataclasses.dataclass
class ForBuildResult:
    """What the parent WorkflowParser needs back after build_body."""

    used_accumulators: dict[str, str]  # accumulator_symbol -> appended_symbol
    broadcast_symbols: list[str]
    scattered_symbols: list[str]


class ForParser:
    body_label: ClassVar[str] = "body"

    def __init__(
        self,
        body_walker: parser_protocol.BodyWalker,
        accumulators: set[str],
    ):
        self.body_walker = body_walker
        self.accumulators = accumulators

        # When these are all filled, we are ready to `build_model`
        self._inputs: list[str] = []
        self._input_edges: edge_models.InputEdges = {}
        # Note property: self._body_node
        self._output_edges: edge_models.OutputEdges = {}
        self._outputs: list[str] = []
        self._nested_ports: list[str] = []
        self._zipped_ports: list[str] = []
        self._transfer_edges: edge_models.TransferEdges = {}

        # These are internal state that doesn't translate directly to the final model

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
            transfer_edges=self._transfer_edges,
        )

    def build_body(
        self,
        tree: ast.For,
        scope: scope_helpers.ScopeProxy,
        nested_iters: list[tuple[str, str]],
        zipped_iters: list[tuple[str, str]],
    ) -> ForBuildResult:
        all_iters = nested_iters + zipped_iters

        used_accumulator_symbol_map = walk_ast_for(
            self.body_walker, tree, scope, self.accumulators
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
        self._transfer_edges = {}
        for accumulator_symbol, appended_symbol in used_accumulator_symbol_map.items():
            target = edge_models.OutputTarget(port=accumulator_symbol)
            if appended_symbol in self.body_walker.outputs:
                self._output_edges[target] = edge_models.SourceHandle(
                    node=self.body_label, port=appended_symbol
                )
            else:
                self._transfer_edges[target] = self._input_edges[
                    edge_models.TargetHandle(node=self.body_label, port=appended_symbol)
                ]

        return ForBuildResult(
            used_accumulators=used_accumulator_symbol_map,
            broadcast_symbols=broadcast_symbols,
            scattered_symbols=scattered_symbols,
        )


def parse_for_iterations(
    for_stmt: ast.For,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], ast.For]:
    """
    Parse for loop iteration structure, handling zip and immediately nested loops.

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

        # Check for nested for loop (single statement that's another For)
        if len(current.body) >= 1 and isinstance(current.body[0], ast.For):
            current = current.body[0]
        else:
            break

    return nested, zipped, current


def _parse_single_for_header(
    for_stmt: ast.For,
) -> tuple[bool, list[tuple[str, str]]]:
    """
    Parse a single for loop header.

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
            "For loop must iterate over a symbol (not an inline expression)"
        )

    if isinstance(target, ast.Name):
        return False, [(target.id, iter_expr.id)]
    elif isinstance(target, ast.Tuple):
        # for a, b in items (tuple unpacking without zip)
        raise ValueError(
            "Tuple unpacking in for loops requires zip(). "
            "Use 'for a, b in zip(as, bs):' instead of 'for a, b in items:'"
        )
    else:
        raise ValueError(f"Unsupported for loop target: {type(target)}")


def _is_zip_call(node: ast.Call) -> bool:
    """Check if a Call node is a call to zip()."""
    return isinstance(node.func, ast.Name) and node.func.id == "zip"
