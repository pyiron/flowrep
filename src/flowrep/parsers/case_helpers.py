from __future__ import annotations

import ast
import dataclasses

from pyiron_snippets import versions

from flowrep import edge_models, subgraph_validation
from flowrep.parsers import (
    atomic_parser,
    attribute_parser,
    object_scope,
    parser_helpers,
    parser_protocol,
    symbol_scope,
)
from flowrep.prospective import constant_recipe, helper_models, union_types


def parse_case(
    test: ast.expr,
    scope: object_scope.ScopeProxy,
    symbol_map: symbol_scope.SymbolScope,
    info_factory: versions.VersionInfoFactory,
    label: str,
    nodes: union_types.Recipes,
    reserved_ports: set[str] | None = None,
) -> tuple[
    helper_models.LabeledRecipe,
    edge_models.InputEdges,
    dict[str, constant_recipe.ConstantRecipe],
]:
    """
    Parse a conditional expression.

    Validates that the statement is a function call returning exactly one value.
    Returns the labeled condition node, and the input edges neeeded to feed it.
    """
    if not isinstance(test, ast.Call):
        raise ValueError(
            "Test conditions must be a function call, but got " f"{type(test).__name__}"
        )
    attribute_parser.reject_method_call(test, symbol_map)

    condition = atomic_parser.get_labeled_recipe(test, set(), scope, info_factory)
    if len(condition.recipe.outputs) != 1:
        raise ValueError(
            f"If/elif condition must return exactly one value (and it had better be "
            f"truthy), but got {condition.recipe.outputs}"
        )

    scope_copy = symbol_map.fork()
    condition_bindings: dict[str, constant_recipe.ConstantRecipe] = {}
    parser_helpers.consume_call_arguments(
        scope_copy,
        test,
        condition,
        nodes,
        condition_bindings=condition_bindings,
        reserved_ports=reserved_ports,
    )
    relabeled_node, relabeled_inputs = _relabel_node_data(
        condition, scope_copy.input_edges, label
    )
    return relabeled_node, relabeled_inputs, condition_bindings


def _relabel_node_data(
    labeled_node: helper_models.LabeledRecipe,
    inputs: edge_models.InputEdges,
    new_label: str,
) -> tuple[helper_models.LabeledRecipe, edge_models.InputEdges]:
    relabeled_node = helper_models.LabeledRecipe(
        label=new_label, recipe=labeled_node.recipe
    )
    relabeled_inputs: edge_models.InputEdges = {
        edge_models.TargetHandle(node=new_label, port=target.port): source
        for target, source in inputs.items()
    }
    return relabeled_node, relabeled_inputs


@dataclasses.dataclass
class WalkedBranch:
    label: str
    walker: parser_protocol.BodyWalker
    assigned: list[str]

    def to_labeled_node(self) -> helper_models.LabeledRecipe:
        return helper_models.LabeledRecipe(
            label=self.label,
            recipe=self.walker.build_model(),
        )


def walk_branch(
    walker: parser_protocol.BodyWalker,
    label: str,
    stmts: list[ast.stmt],
) -> WalkedBranch:
    """
    Fork a walker and walk a conditional branch body.

    Both the :class:`SymbolScope` and the :class:`ScopeProxy` are forked so
    that symbol assignments *and* import-based scope extensions in one branch
    do not leak into sibling or parent branches.
    """
    symbol_fork = walker.symbol_map.fork()
    scope_fork = walker.scope.fork()
    branch_walker = walker.fork(
        new_symbol_map=symbol_fork,
        new_scope=scope_fork,
    )
    branch_walker.walk(stmts)
    assigned = symbol_fork.assigned_symbols
    symbol_fork.produce_symbols(assigned)
    return WalkedBranch(label, branch_walker, assigned)


def wire_inputs(
    branches: list[WalkedBranch],
) -> tuple[list[str], edge_models.InputEdges]:
    """Collect input edges from the condition and body branches."""
    inputs: list[str] = []
    input_edges: edge_models.InputEdges = {}

    def _add_input(port: str) -> None:
        if port not in inputs:
            inputs.append(port)

    for branch in branches:
        for port in branch.walker.inputs:
            input_edges[edge_models.TargetHandle(node=branch.label, port=port)] = (
                edge_models.InputSource(port=port)
            )
            _add_input(port)

    return inputs, input_edges


def wire_outputs(
    branches: list[WalkedBranch],
) -> tuple[list[str], subgraph_validation.ProspectiveOutputEdges]:
    """Collect outputs and prospective output edges from try and except bodies."""
    # Union of assigned symbols across all branches, preserving first-seen order
    outputs: list[str] = []
    seen: set[str] = set()
    for branch in branches:
        for sym in branch.assigned:
            if sym not in seen:
                seen.add(sym)
                outputs.append(sym)

    for branch in branches:
        parser_helpers.reject_input_alias_outputs(
            branch.walker.symbol_map, outputs, "branch"
        )

    # Build prospective output edges: each output maps to the list of branch
    # body nodes that can source it.
    prospective_output_edges: subgraph_validation.ProspectiveOutputEdges = {}
    for output_name in outputs:
        target = edge_models.OutputTarget(port=output_name)
        sources: list[edge_models.SourceHandle] = []
        for branch in branches:
            if output_name in branch.assigned:
                sources.append(
                    edge_models.SourceHandle(node=branch.label, port=output_name)
                )
        prospective_output_edges[target] = sources

    return outputs, prospective_output_edges
