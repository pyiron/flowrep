from __future__ import annotations

import ast
import dataclasses

from flowrep.models import edge_models
from flowrep.models.nodes import helper_models, if_model
from flowrep.models.parsers import case_helpers, parser_protocol

IF_CONDITION_LABEL_PREFIX: str = "condition"
IF_BODY_LABEL_PREFIX: str = "body"
IF_ELSE_LABEL: str = "else_body"


@dataclasses.dataclass
class _CaseComponents:
    """Intermediate data collected while processing a single if/elif branch."""

    condition: helper_models.LabeledNode
    condition_input_edges: edge_models.InputEdges
    body: case_helpers.WalkedBranch


def parse_if_node(tree: ast.If, walker: parser_protocol.BodyWalker) -> if_model.IfNode:
    """
    Walk an if/elif/else chain.

    Args:
        tree: The top-level ``ast.If`` node.
        walker: A walker to fork and use for collecting state inside the tree.
    """

    cases: list[_CaseComponents] = []
    else_branch: case_helpers.WalkedBranch | None = None

    ast_cases, else_stmts = _parse_if_elif_chain(tree)

    # --- process each if / elif case ---
    for idx, (test_expr, body_stmts) in enumerate(ast_cases):
        cond_label = f"{IF_CONDITION_LABEL_PREFIX}_{idx}"
        body_label = f"{IF_BODY_LABEL_PREFIX}_{idx}"

        labeled_cond, cond_inputs = case_helpers.parse_case(
            test_expr,
            walker.scope,
            walker.symbol_map,
            walker.info_factory,
            cond_label,
        )
        body = case_helpers.walk_branch(body_label, body_stmts, walker)
        cases.append(
            _CaseComponents(
                condition=labeled_cond,
                condition_input_edges=cond_inputs,
                body=body,
            )
        )

    # --- process else case (if present) ---
    if else_stmts is not None:
        else_branch = case_helpers.walk_branch(IF_ELSE_LABEL, else_stmts, walker)

    # --- wire edges ---
    body_branches = [cc.body for cc in cases]
    if else_branch is not None:
        body_branches.append(else_branch)

    inputs, input_edges = _wire_inputs(cases, body_branches)
    outputs, prospective_output_edges = case_helpers.wire_outputs(body_branches)

    model_cases = [
        helper_models.ConditionalCase(
            condition=cc.condition,
            body=cc.body.to_labeled_node(),
        )
        for cc in cases
    ]

    return if_model.IfNode(
        inputs=inputs,
        outputs=outputs,
        cases=model_cases,
        input_edges=input_edges,
        prospective_output_edges=prospective_output_edges,
        else_case=else_branch.to_labeled_node() if else_branch else None,
    )


def _wire_inputs(
    cases: list[_CaseComponents],
    body_branches: list[case_helpers.WalkedBranch],
) -> tuple[list[str], edge_models.InputEdges]:
    """Merge condition input edges with body/else branch input edges."""
    inputs: list[str] = []
    input_edges: edge_models.InputEdges = {}

    def _add_input(port: str) -> None:
        if port not in inputs:
            inputs.append(port)

    # Condition inputs first (preserves expected edge ordering)
    for cc in cases:
        for target, source in cc.condition_input_edges.items():
            input_edges[target] = source
            _add_input(source.port)

    # Body + else inputs via shared helper
    branch_inputs, branch_edges = case_helpers.wire_inputs(body_branches)
    input_edges.update(branch_edges)
    for port in branch_inputs:
        _add_input(port)

    return inputs, input_edges


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
