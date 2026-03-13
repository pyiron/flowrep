"""
This module holds a prototypical, minimal Workflow Management System (WfMS) for
flowrep recipes.
"""

from __future__ import annotations

from typing import Any

from flowrep.models import base_models, edge_models, live
from flowrep.models.api import schemas
from flowrep.models.nodes import (
    atomic_model,
    for_model,
    if_model,
    try_model,
    while_model,
    workflow_model,
)


def run_recipe(recipe: schemas.NodeType, **kwargs: Any) -> live.LiveNode:
    """
    Execute a flowrep recipe, returning a populated :class:`LiveNode`.

    All inputs are passed as keyword arguments matching the recipe's input port names.
    """
    match recipe:
        case atomic_model.AtomicNode():
            return _run_atomic(recipe, **kwargs)
        case workflow_model.WorkflowNode():
            return _run_workflow(recipe, **kwargs)
        case for_model.ForNode():
            return _run_for(recipe, **kwargs)
        case if_model.IfNode():
            return _run_if(recipe, **kwargs)
        case try_model.TryNode():
            return _run_try(recipe, **kwargs)
        case while_model.WhileNode():
            return _run_while(recipe, **kwargs)
        case _:
            raise ValueError(f"Unsupported recipe type: {type(recipe).__name__}")


# ---------------------------------------------------------------------------
# Atomic
# ---------------------------------------------------------------------------


def _run_atomic(recipe: atomic_model.AtomicNode, **kwargs: Any) -> live.Atomic:
    node = live.Atomic.from_recipe(recipe)
    _populate_input_ports(node, kwargs)
    result = _call_atomic(node)
    _store_atomic_outputs(node, result)
    return node


def _call_atomic(node: live.Atomic) -> Any:
    """
    Invoke the underlying function, respecting positional-only parameter kinds.

    Values are drawn from the live input ports; if a port has no value, its
    default is used.  A :class:`ValueError` is raised when neither is available.
    """
    recipe = node.recipe
    assert isinstance(recipe, atomic_model.AtomicNode)

    positional: list[Any] = []
    keyword: dict[str, Any] = {}

    for name in recipe.inputs:
        port = node.input_ports[name]
        val = port.value if not isinstance(port.value, live.NotData) else port.default
        if isinstance(val, live.NotData):
            raise ValueError(f"Input port '{name}' has no value and no default")

        kind = recipe.reference.restricted_input_kinds.get(name)
        if kind == base_models.RestrictedParamKind.POSITIONAL_ONLY:
            positional.append(val)
        else:
            keyword[name] = val

    return node.function(*positional, **keyword)


def _store_atomic_outputs(node: live.Atomic, result: Any) -> None:
    recipe = node.recipe
    assert isinstance(recipe, atomic_model.AtomicNode)
    output_names = list(node.output_ports.keys())

    if recipe.unpack_mode == atomic_model.UnpackMode.NONE:
        node.output_ports[output_names[0]].value = result

    elif recipe.unpack_mode == atomic_model.UnpackMode.TUPLE:
        if len(output_names) == 1:
            node.output_ports[output_names[0]].value = result
        else:
            for name, val in zip(output_names, result, strict=True):
                node.output_ports[name].value = val

    elif recipe.unpack_mode == atomic_model.UnpackMode.DATACLASS:
        for name in output_names:
            node.output_ports[name].value = getattr(result, name)


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


def _run_workflow(recipe: workflow_model.WorkflowNode, **kwargs: Any) -> live.Workflow:
    node = live.Workflow.from_recipe(recipe)
    _populate_input_ports(node, kwargs)

    for child_label in _topo_sort_children(recipe):
        child_inputs = _gather_child_inputs(child_label, recipe, node)
        child_recipe = recipe.nodes[child_label]
        child_node = run_recipe(child_recipe, **child_inputs)
        node.nodes[child_label] = child_node

    _populate_workflow_outputs(node, recipe)
    return node


def _topo_sort_children(recipe: workflow_model.WorkflowNode) -> list[str]:
    """Kahn's algorithm over sibling edges; deterministic tie-breaking by label."""
    in_degree: dict[str, int] = {label: 0 for label in recipe.nodes}
    successors: dict[str, list[str]] = {label: [] for label in recipe.nodes}

    for target, source in recipe.edges.items():
        in_degree[target.node] += 1
        successors[source.node].append(target.node)

    queue = sorted(
        (label for label in recipe.nodes if in_degree[label] == 0),
    )
    order: list[str] = []
    while queue:
        label = queue.pop(0)
        order.append(label)
        for succ in sorted(successors.get(label, [])):
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)

    if len(order) != len(recipe.nodes):
        raise ValueError("Cycle detected in workflow edges")
    return order


def _gather_child_inputs(
    child_label: str,
    recipe: workflow_model.WorkflowNode,
    workflow_node: live.Workflow,
) -> dict[str, Any]:
    """
    Resolve input values for a child node from workflow input ports and sibling
    output ports according to the recipe edges.

    Ports not covered by any edge are omitted — the child's own defaults (if any)
    will be used downstream.
    """
    child_recipe = recipe.nodes[child_label]
    inputs: dict[str, Any] = {}

    for port in child_recipe.inputs:
        th = edge_models.TargetHandle(node=child_label, port=port)

        if th in recipe.input_edges:
            parent_source = recipe.input_edges[th]
            inputs[port] = workflow_node.input_ports[parent_source.port].value
        elif th in recipe.edges:
            sibling_source = recipe.edges[th]
            sibling = workflow_node.nodes[sibling_source.node]
            inputs[port] = sibling.output_ports[sibling_source.port].value
        # else: port has a default on the child, _call_atomic will handle it

    return inputs


def _populate_workflow_outputs(
    node: live.Workflow, recipe: workflow_model.WorkflowNode
) -> None:
    for target, source in recipe.output_edges.items():
        if isinstance(source, edge_models.InputSource):
            val = node.input_ports[source.port].value
        else:
            child = node.nodes[source.node]
            val = child.output_ports[source.port].value
        node.output_ports[target.port].value = val


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _populate_input_ports(node: live.LiveNode, values: dict[str, Any]) -> None:
    for name, val in values.items():
        if name in node.input_ports:
            node.input_ports[name].value = val


# ---------------------------------------------------------------------------
# Flow-control stubs
# ---------------------------------------------------------------------------


def _run_for(recipe: for_model.ForNode, **kwargs: Any) -> live.FlowControl:
    raise NotImplementedError("For-node execution not yet implemented")


def _run_if(recipe: if_model.IfNode, **kwargs: Any) -> live.FlowControl:
    raise NotImplementedError("If-node execution not yet implemented")


def _run_try(recipe: try_model.TryNode, **kwargs: Any) -> live.FlowControl:
    raise NotImplementedError("Try-node execution not yet implemented")


def _run_while(recipe: while_model.WhileNode, **kwargs: Any) -> live.FlowControl:
    raise NotImplementedError("While-node execution not yet implemented")
