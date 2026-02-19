import ast

from flowrep.models import edge_models
from flowrep.models.nodes import helper_models
from flowrep.models.parsers import (
    atomic_parser,
    object_scope,
    parser_helpers,
    symbol_scope,
)


def parse_case(
    test: ast.expr,
    scope: object_scope.ScopeProxy,
    symbol_map: symbol_scope.SymbolScope,
    label: str,
) -> tuple[helper_models.LabeledNode, edge_models.InputEdges]:
    """
    Parse a conditional expression.

    Validates that the statement is a function call returning exactly one value.
    Returns the labeled condition node, and the input edges neeeded to feed it.
    """
    if not isinstance(test, ast.Call):
        raise ValueError(
            "Test conditions must be a function call, but got " f"{type(test).__name__}"
        )

    condition = atomic_parser.get_labeled_recipe(test, set(), scope)
    if len(condition.node.outputs) != 1:
        raise ValueError(
            f"If/elif condition must return exactly one value (and it had better be "
            f"truthy), but got {condition.node.outputs}"
        )

    scope_copy = symbol_map.fork_scope()
    parser_helpers.consume_call_arguments(scope_copy, test, condition)
    return _relabel_node_data(condition, scope_copy.input_edges, label)


def _relabel_node_data(
    labeled_node: helper_models.LabeledNode,
    inputs: edge_models.InputEdges,
    new_label: str,
) -> tuple[helper_models.LabeledNode, edge_models.InputEdges]:
    relabeled_node = helper_models.LabeledNode(label=new_label, node=labeled_node.node)
    relabeled_inputs: edge_models.InputEdges = {
        edge_models.TargetHandle(node=new_label, port=target.port): source
        for target, source in inputs.items()
    }
    return relabeled_node, relabeled_inputs
