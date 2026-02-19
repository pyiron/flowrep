from __future__ import annotations

import ast
import dataclasses
from collections.abc import Callable

from flowrep.models import edge_models, subgraph_validation
from flowrep.models.nodes import helper_models, if_model
from flowrep.models.parsers import (
    atomic_parser,
    object_scope,
    parser_helpers,
    parser_protocol,
    symbol_scope,
)

IF_CONDITION_LABEL_PREFIX: str = "condition"
IF_BODY_LABEL_PREFIX: str = "body"
IF_ELSE_LABEL: str = "else_body"


@dataclasses.dataclass
class _CaseComponents:
    """Intermediate data collected while processing a single if/elif branch."""

    condition: helper_models.LabeledNode
    condition_inputs_edges: edge_models.InputEdges
    body_walker: parser_protocol.BodyWalker
    body_label: str
    assigned_symbols: list[str]


def parse_if_node(
    tree: ast.If,
    scope: object_scope.ScopeProxy,
    symbol_map: symbol_scope.SymbolScope,
    walker_factory: Callable[[symbol_scope.SymbolScope], parser_protocol.BodyWalker],
):
    """
    Walk an if/elif/else chain.

    Args:
        tree: The top-level ``ast.If`` node.
        scope: Object-level scope for resolving callable references.
        symbol_map: The enclosing :class:`SymbolScope` (used for forking).
        walker_factory: Callable that creates a :class:`BodyWalker` from a
            :class:`SymbolScope`.  Avoids a circular import with
            ``workflow_parser.WorkflowParser``.
    """

    case_components: list[_CaseComponents] = []
    else_walker: parser_protocol.BodyWalker | None = None
    else_assigned: list[str] = []

    case_branches, else_stmts = _parse_if_elif_chain(tree)

    # --- process each if / elif case ---
    for idx, (test_expr, body_stmts) in enumerate(case_branches):
        cond_label = f"{IF_CONDITION_LABEL_PREFIX}_{idx}"
        body_label = f"{IF_BODY_LABEL_PREFIX}_{idx}"

        # Parse condition node
        labeled_cond, cond_inputs = _parse_if_condition(test_expr, scope, symbol_map)
        # Relabel to our naming scheme
        relabeled_cond = helper_models.LabeledNode(
            label=cond_label, node=labeled_cond.node
        )
        relabeled_inputs: edge_models.InputEdges = {
            edge_models.TargetHandle(node=cond_label, port=target.port): source
            for target, source in cond_inputs.items()
        }

        # Fork scope and walk body
        body_symbol_map = symbol_map.fork_scope()
        body_walker = walker_factory(body_symbol_map)
        body_walker.walk(body_stmts, scope)

        # Identify symbols assigned in this branch and produce them as body
        # outputs so that output_edges / build_model can reference them.
        assigned = body_symbol_map.assigned_symbols
        for sym in assigned:
            body_symbol_map.produce(sym)

        case_components.append(
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
        else_scope = symbol_map.fork_scope()
        else_walker = walker_factory(else_scope)
        else_walker.walk(else_stmts, scope)
        else_assigned = else_scope.assigned_symbols
        for sym in else_assigned:
            else_scope.produce(sym)

    inputs, input_edges = _wire_inputs(case_components, else_walker)
    outputs, prospective_output_edges = _wire_outputs(
        case_components, else_walker, else_assigned
    )

    cases = [
        helper_models.ConditionalCase(
            condition=cc.condition,
            body=helper_models.LabeledNode(
                label=cc.body_label,
                node=cc.body_walker.build_model(),
            ),
        )
        for cc in case_components
    ]
    else_case = (
        helper_models.LabeledNode(
            label=IF_ELSE_LABEL,
            node=else_walker.build_model(),
        )
        if else_walker is not None
        else None
    )
    return if_model.IfNode(
        inputs=inputs,
        outputs=outputs,
        cases=cases,
        input_edges=input_edges,
        prospective_output_edges=prospective_output_edges,
        else_case=else_case,
    )


def _wire_inputs(
    case_components, else_walker
) -> tuple[list[str], edge_models.InputEdges]:
    """Collect input edges from conditions, case bodies, and else body."""
    inputs = []
    input_edges = {}

    def _add_input(input_port: str) -> None:
        if input_port not in inputs:
            inputs.append(input_port)

    # Condition inputs
    for cc in case_components:
        for target, source in cc.condition_inputs_edges.items():
            input_edges[target] = source
            _add_input(source.port)

    # Case body inputs
    for cc in case_components:
        for port in cc.body_walker.inputs:
            input_edges[edge_models.TargetHandle(node=cc.body_label, port=port)] = (
                edge_models.InputSource(port=port)
            )
            _add_input(port)

    # Else body inputs
    if else_walker is not None:
        for port in else_walker.inputs:
            input_edges[edge_models.TargetHandle(node=IF_ELSE_LABEL, port=port)] = (
                edge_models.InputSource(port=port)
            )
            _add_input(port)
    return inputs, input_edges


def _wire_outputs(
    case_components, else_walker, else_assigned
) -> tuple[list[str], subgraph_validation.ProspectiveOutputEdges]:
    """Collect outputs and prospective output edges from all branches."""
    # Union of assigned symbols across all branches, preserving first-seen order
    outputs: list[str] = []
    seen: set[str] = set()
    for cc in case_components:
        for sym in cc.assigned_symbols:
            if sym not in seen:
                seen.add(sym)
                outputs.append(sym)
    for sym in else_assigned:
        if sym not in seen:
            seen.add(sym)
            outputs.append(sym)

    # Build prospective output edges: each output maps to the list of branch
    # body nodes that can source it.
    prospective_output_edges: subgraph_validation.ProspectiveOutputEdges = {}
    for output_name in outputs:
        target = edge_models.OutputTarget(port=output_name)
        sources: list[edge_models.SourceHandle] = []
        for cc in case_components:
            if output_name in cc.assigned_symbols:
                sources.append(
                    edge_models.SourceHandle(node=cc.body_label, port=output_name)
                )
        if else_walker is not None and output_name in else_assigned:
            sources.append(
                edge_models.SourceHandle(node=IF_ELSE_LABEL, port=output_name)
            )
        prospective_output_edges[target] = sources
    return outputs, prospective_output_edges


# ======================================================================
# Pure-AST helpers
# ======================================================================


def _parse_if_elif_chain(
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


def _parse_if_condition(
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

    scope_copy = parent_scope.fork_scope()
    parser_helpers.consume_call_arguments(scope_copy, test_expr, condition_node)
    return condition_node, scope_copy.input_edges
