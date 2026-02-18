from __future__ import annotations

import ast
import dataclasses
from collections.abc import Callable
from typing import ClassVar

from flowrep.models import edge_models
from flowrep.models.nodes import helper_models, if_model
from flowrep.models.parsers import (
    atomic_parser,
    object_scope,
    parser_helpers,
    parser_protocol,
    symbol_scope,
)


@dataclasses.dataclass
class _CaseComponents:
    """Intermediate data collected while processing a single if/elif branch."""

    condition: helper_models.LabeledNode
    condition_inputs_edges: edge_models.InputEdges
    body_walker: parser_protocol.BodyWalker
    body_label: str
    assigned_symbols: list[str]


class IfParser:
    condition_label_prefix: ClassVar[str] = "condition"
    body_label_prefix: ClassVar[str] = "body"
    else_label: ClassVar[str] = "else_body"

    def __init__(self) -> None:
        self._case_components: list[_CaseComponents] = []
        self._else_walker: parser_protocol.BodyWalker | None = None
        self._else_assigned: list[str] = []

        self._inputs: list[str] = []
        self._input_edges: edge_models.InputEdges = {}
        self._prospective_output_edges: dict[
            edge_models.OutputTarget, list[edge_models.SourceHandle]
        ] = {}
        self._outputs: list[str] = []

    def build_model(self) -> if_model.IfNode:
        cases = [
            helper_models.ConditionalCase(
                condition=cc.condition,
                body=helper_models.LabeledNode(
                    label=cc.body_label,
                    node=cc.body_walker.build_model(),
                ),
            )
            for cc in self._case_components
        ]
        else_case = (
            helper_models.LabeledNode(
                label=self.else_label,
                node=self._else_walker.build_model(),
            )
            if self._else_walker is not None
            else None
        )
        return if_model.IfNode(
            inputs=self._inputs,
            outputs=self._outputs,
            cases=cases,
            input_edges=self._input_edges,
            prospective_output_edges=self._prospective_output_edges,
            else_case=else_case,
        )

    def build_body(
        self,
        tree: ast.If,
        scope: object_scope.ScopeProxy,
        symbol_scope: symbol_scope.SymbolScope,
        walker_factory: Callable[
            [symbol_scope.SymbolScope], parser_protocol.BodyWalker
        ],
    ) -> None:
        """
        Walk an if/elif/else chain, building all internal state needed for
        :meth:`build_model`.

        Args:
            tree: The top-level ``ast.If`` node.
            scope: Object-level scope for resolving callable references.
            symbol_scope: The enclosing :class:`SymbolScope` (used for forking).
            walker_factory: Callable that creates a :class:`BodyWalker` from a
                :class:`SymbolScope`.  Avoids a circular import with
                ``workflow_parser.WorkflowParser``.
        """
        case_branches, else_stmts = parse_if_elif_chain(tree)

        # --- process each if / elif case ---
        for idx, (test_expr, body_stmts) in enumerate(case_branches):
            cond_label = f"{self.condition_label_prefix}_{idx}"
            body_label = f"{self.body_label_prefix}_{idx}"

            # Parse condition node
            labeled_cond, cond_inputs = parse_if_condition(
                test_expr, scope, symbol_scope
            )
            # Relabel to our naming scheme
            relabeled_cond = helper_models.LabeledNode(
                label=cond_label, node=labeled_cond.node
            )
            relabeled_inputs: edge_models.InputEdges = {
                edge_models.TargetHandle(node=cond_label, port=target.port): source
                for target, source in cond_inputs.items()
            }

            # Fork scope and walk body
            body_symbol_scope = symbol_scope.fork_scope({})
            body_walker = walker_factory(body_symbol_scope)
            body_walker.walk(body_stmts, scope)

            # Identify symbols assigned in this branch and produce them as body
            # outputs so that output_edges / build_model can reference them.
            assigned = _get_assigned_symbols(body_symbol_scope)
            for sym in assigned:
                body_symbol_scope.produce(sym, sym)

            self._case_components.append(
                _CaseComponents(
                    condition=relabeled_cond,
                    condition_inputs_edges=relabeled_inputs,
                    body_walker=body_walker,
                    body_label=body_label,
                    assigned_symbols=assigned,
                )
            )

        # --- process else case (if present) ---
        if else_stmts is not None:
            else_scope = symbol_scope.fork_scope({})
            self._else_walker = walker_factory(else_scope)
            self._else_walker.walk(else_stmts, scope)
            self._else_assigned = _get_assigned_symbols(else_scope)
            for sym in self._else_assigned:
                else_scope.produce(sym, sym)

        self._wire_outputs()
        self._wire_inputs()

    # ------------------------------------------------------------------
    # Private wiring helpers
    # ------------------------------------------------------------------

    def _wire_outputs(self) -> None:
        """Collect outputs and prospective output edges from all branches."""
        # Union of assigned symbols across all branches, preserving first-seen order
        all_outputs: list[str] = []
        seen: set[str] = set()
        for cc in self._case_components:
            for sym in cc.assigned_symbols:
                if sym not in seen:
                    seen.add(sym)
                    all_outputs.append(sym)
        for sym in self._else_assigned:
            if sym not in seen:
                seen.add(sym)
                all_outputs.append(sym)
        self._outputs = all_outputs

        # Build prospective output edges: each output maps to the list of branch
        # body nodes that can source it.
        self._prospective_output_edges = {}
        for output_name in all_outputs:
            target = edge_models.OutputTarget(port=output_name)
            sources: list[edge_models.SourceHandle] = []
            for cc in self._case_components:
                if output_name in cc.assigned_symbols:
                    sources.append(
                        edge_models.SourceHandle(node=cc.body_label, port=output_name)
                    )
            if self._else_walker is not None and output_name in self._else_assigned:
                sources.append(
                    edge_models.SourceHandle(node=self.else_label, port=output_name)
                )
            self._prospective_output_edges[target] = sources

    def _wire_inputs(self) -> None:
        """Collect input edges from conditions, case bodies, and else body."""
        self._input_edges = {}
        self._inputs = []

        def _add_input(port: str) -> None:
            if port not in self._inputs:
                self._inputs.append(port)

        # Condition inputs
        for cc in self._case_components:
            for target, source in cc.condition_inputs_edges.items():
                self._input_edges[target] = source
                _add_input(source.port)

        # Case body inputs
        for cc in self._case_components:
            for port in cc.body_walker.inputs:
                self._input_edges[
                    edge_models.TargetHandle(node=cc.body_label, port=port)
                ] = edge_models.InputSource(port=port)
                _add_input(port)

        # Else body inputs
        if self._else_walker is not None:
            for port in self._else_walker.inputs:
                self._input_edges[
                    edge_models.TargetHandle(node=self.else_label, port=port)
                ] = edge_models.InputSource(port=port)
                _add_input(port)


# ======================================================================
# Pure-AST helpers
# ======================================================================


def parse_if_elif_chain(
    tree: ast.If,
) -> tuple[list[tuple[ast.expr, list[ast.stmt]]], list[ast.stmt] | None]:
    """
    Flatten an ``if / elif / else`` chain.

    Returns:
        A tuple of ``(cases, else_body)`` where *cases* is a list of
        ``(test_expr, body_statements)`` for every ``if`` / ``elif`` branch,
        and *else_body* is the else body statements (or ``None`` if absent).
    """
    cases: list[tuple[ast.expr, list[ast.stmt]]] = []
    current = tree
    while True:
        cases.append((current.test, current.body))
        if not current.orelse:
            return cases, None
        elif len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
            current = current.orelse[0]
        else:
            return cases, current.orelse


def parse_if_condition(
    test_expr: ast.expr,
    scope: object_scope.ScopeProxy,
    parent_scope: symbol_scope.SymbolScope,
) -> tuple[helper_models.LabeledNode, edge_models.InputEdges]:
    """
    Parse a single if/elif condition expression.

    Validates that the condition is a function call returning exactly one value.
    Returns the labeled condition node and the input edges needed to feed it.
    """
    if not isinstance(test_expr, ast.Call):
        raise ValueError(
            "If/elif conditions must be a function call, but got "
            f"{type(test_expr).__name__}"
        )

    condition_node = atomic_parser.get_labeled_recipe(test_expr, set(), scope)
    if len(condition_node.node.outputs) != 1:
        raise ValueError(
            f"If/elif condition must return exactly one value (and it had better be "
            f"truthy), but got {condition_node.node.outputs}"
        )

    scope_copy = parent_scope.fork_scope({})
    parser_helpers.consume_call_arguments(scope_copy, test_expr, condition_node)
    return condition_node, scope_copy.input_edges


# ======================================================================
# Internal helpers
# ======================================================================


def _get_assigned_symbols(scope: symbol_scope.SymbolScope) -> list[str]:
    """
    Identify symbols that were assigned (registered to child nodes) within a
    forked scope.

    In a forked scope every inherited symbol starts as an :class:`InputSource`.
    Any key whose source is now a :class:`SourceHandle` must have been assigned
    by a node inside the branch.
    """
    return [key for key in scope if isinstance(scope[key], edge_models.SourceHandle)]
