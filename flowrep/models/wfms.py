"""
This module holds a prototypical, minimal Workflow Management System (WfMS) for
flowrep recipes.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Collection
from typing import Any, cast

from pyiron_snippets import retrieve

from flowrep.models import base_models, edge_models, live
from flowrep.models.nodes import (
    atomic_model,
    for_model,
    helper_models,
    if_model,
    try_model,
    union,
    while_model,
    workflow_model,
)
from flowrep.models.parsers import label_helpers


def run_recipe(recipe: union.NodeType, **kwargs: Any) -> live.LiveNode:
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
            raise TypeError(f"Unsupported recipe type: {type(recipe).__name__}")


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
        fields = dataclasses.fields(result)
        for label, field in zip(node.recipe.outputs, fields, strict=True):
            node.output_ports[label].value = getattr(result, field.name)


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
        node.nodes[child_label] = child_node  # Overwrite with _executed_ child

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

    if len(order) != len(recipe.nodes):  # pragma: no cover
        raise ValueError(
            "Cycle detected in workflow edges. This should have been caught by the "
            "underlying recipe validation. Please raise a GitHub issue reporting "
            "how you got here!"
        )
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
# For
# ---------------------------------------------------------------------------


def _run_for(recipe: for_model.ForNode, **kwargs: Any) -> live.FlowControl:
    """
    Execute a for-node by scattering iterated inputs across body instances and
    collecting outputs into lists.

    Nested ports drive a Cartesian product; zipped ports are iterated in lockstep.
    Broadcast (non-iterated) inputs are passed unchanged to every body instance.
    Transferred outputs collect the per-iteration value of a scattered input,
    preserving the link between input element and body output element.
    """
    node = live.FlowControl.from_recipe(recipe)
    _populate_input_ports(node, kwargs)

    body_label = recipe.body_node.label
    body_recipe = recipe.body_node.node
    iterated_ports = recipe.iterated_ports

    # body iterated port -> for-node input name
    body_to_for: dict[str, str] = {}
    for port in iterated_ports:
        th = edge_models.TargetHandle(node=body_label, port=port)
        body_to_for[port] = recipe.input_edges[th].port

    # Reverse mapping for transferred outputs
    for_to_body: dict[str, str] = {v: k for k, v in body_to_for.items()}

    # Broadcast inputs (non-iterated body ports sourced from for-node inputs)
    broadcast: dict[str, Any] = {}
    for port in body_recipe.inputs:
        if port not in iterated_ports:
            th = edge_models.TargetHandle(node=body_label, port=port)
            if th in recipe.input_edges:
                src = recipe.input_edges[th]
                broadcast[port] = node.input_ports[src.port].value

    # Build iteration axes
    nested_iters = [
        cast(Collection, node.input_ports[body_to_for[p]].value)
        for p in recipe.nested_ports
    ]
    zipped_iters = [
        cast(Collection, node.input_ports[body_to_for[p]].value)
        for p in recipe.zipped_ports
    ]
    # Note that we simply cast iterated input values to the form we expect, and let the
    # user pay the price if runtime data is non-compliant.

    nested_combos = list(itertools.product(*nested_iters)) if nested_iters else [()]
    if zipped_iters:
        zip_len = len(zipped_iters[0])
        for zi in zipped_iters:
            if len(zi) != zip_len:
                raise ValueError("Zipped inputs must have equal lengths")
        zipped_combos = list(zip(*zipped_iters, strict=True))
    else:
        zipped_combos = [()]

    accumulators: dict[str, list[Any]] = {port: [] for port in recipe.outputs}

    for nested_vals, zipped_vals in itertools.product(nested_combos, zipped_combos):
        body_kwargs = dict(broadcast)
        for port, val in zip(recipe.nested_ports, nested_vals, strict=True):
            body_kwargs[port] = val
        for port, val in zip(recipe.zipped_ports, zipped_vals, strict=True):
            body_kwargs[port] = val

        child = run_recipe(body_recipe, **body_kwargs)
        idx = len(node.nodes)
        node.nodes[label_helpers.index_label(body_label, idx)] = child

        for target, source in recipe.output_edges.items():
            if isinstance(source, edge_models.SourceHandle):
                accumulators[target.port].append(child.output_ports[source.port].value)
            else:
                # Transferred output: collect the scattered input element
                body_port = for_to_body[source.port]
                accumulators[target.port].append(body_kwargs[body_port])

    for port, values in accumulators.items():
        node.output_ports[port].value = values

    return node


# ---------------------------------------------------------------------------
# While
# ---------------------------------------------------------------------------


def _run_while(recipe: while_model.WhileNode, **kwargs: Any) -> live.FlowControl:
    """
    Execute a while-node by repeatedly evaluating a condition and running a body.

    On each iteration the body outputs (which share names with a subset of inputs)
    feed back into the next condition/body evaluation.  If the condition is false on
    the first check, outputs are sourced from the initial input values.
    """
    node = live.FlowControl.from_recipe(recipe)
    _populate_input_ports(node, kwargs)

    cond_label = recipe.case.condition.label
    body_label = recipe.case.body.label
    cond_recipe = recipe.case.condition.node
    body_recipe = recipe.case.body.node

    # Working copy of current values — starts from inputs, body outputs update it
    current: dict[str, Any] = {
        name: node.input_ports[name].value for name in recipe.inputs
    }

    iteration = 0
    while True:
        # --- condition ---
        cond_kwargs = _gather_dynamic_child_inputs(
            cond_label, recipe.input_edges, current
        )
        cond_node = run_recipe(cond_recipe, **cond_kwargs)
        node.nodes[label_helpers.index_label(cond_label, iteration)] = cond_node

        if not _evaluate_condition(recipe.case, cond_node):
            break

        # --- body ---
        body_kwargs = _gather_dynamic_child_inputs(
            body_label, recipe.input_edges, current
        )
        body_node = run_recipe(body_recipe, **body_kwargs)
        node.nodes[label_helpers.index_label(body_label, iteration)] = body_node

        # Feed body outputs back into current values
        for target, source in recipe.output_edges.items():
            current[target.port] = body_node.output_ports[source.port].value

        iteration += 1

    for name in recipe.outputs:
        node.output_ports[name].value = current[name]

    return node


# ---------------------------------------------------------------------------
# If
# ---------------------------------------------------------------------------


def _run_if(recipe: if_model.IfNode, **kwargs: Any) -> live.FlowControl:
    """
    Execute an if-node by walking cases until a condition evaluates positively,
    then executing the matching body (or the else case).

    Output ports that have no source from the executed branch remain NOT_DATA.
    """
    node = live.FlowControl.from_recipe(recipe)
    _populate_input_ports(node, kwargs)

    for case in recipe.cases:
        # --- condition ---
        cond_kwargs = _gather_dynamic_child_inputs(
            case.condition.label, recipe.input_edges, node
        )
        cond_node = run_recipe(case.condition.node, **cond_kwargs)
        node.nodes[case.condition.label] = cond_node

        if _evaluate_condition(case, cond_node):
            _execute_if_branch(node, recipe, case.body)
            return node

    # No case matched — try else
    if recipe.else_case is not None:
        _execute_if_branch(node, recipe, recipe.else_case)

    return node


def _execute_if_branch(
    node: live.FlowControl,
    recipe: if_model.IfNode,
    branch: helper_models.LabeledNode,
) -> None:
    branch_kwargs = _gather_dynamic_child_inputs(branch.label, recipe.input_edges, node)
    branch_node = run_recipe(branch.node, **branch_kwargs)
    node.nodes[branch.label] = branch_node

    _populate_prospective_outputs(node, recipe.prospective_output_edges, branch.label)


# ---------------------------------------------------------------------------
# Try
# ---------------------------------------------------------------------------


def _run_try(recipe: try_model.TryNode, **kwargs: Any) -> live.FlowControl:
    """
    Execute a try-node: run the try body and, on exception, walk exception cases
    for a matching handler.  If no handler matches, the exception propagates.
    """
    node = live.FlowControl.from_recipe(recipe)
    _populate_input_ports(node, kwargs)

    try_kwargs = _gather_dynamic_child_inputs(
        recipe.try_node.label, recipe.input_edges, node
    )

    try:
        try_node = run_recipe(recipe.try_node.node, **try_kwargs)
        node.nodes[recipe.try_node.label] = try_node
        _populate_prospective_outputs(
            node, recipe.prospective_output_edges, recipe.try_node.label
        )
        return node
    except BaseException as exc:
        for case in recipe.exception_cases:
            exc_types = tuple(
                retrieve.import_from_string(info.fully_qualified_name)
                for info in case.exceptions
            )
            if isinstance(exc, exc_types):
                handler_kwargs = _gather_dynamic_child_inputs(
                    case.body.label, recipe.input_edges, node
                )
                handler_node = run_recipe(case.body.node, **handler_kwargs)
                node.nodes[case.body.label] = handler_node
                _populate_prospective_outputs(
                    node, recipe.prospective_output_edges, case.body.label
                )
                return node
        raise


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _populate_input_ports(node: live.LiveNode, values: dict[str, Any]) -> None:
    for name, val in values.items():
        if name in node.input_ports:
            node.input_ports[name].value = val
        else:
            raise ValueError(
                f"Input port '{name}' not found -- please select among "
                f"{node.recipe.inputs}"
            )


def _gather_dynamic_child_inputs(
    child_label: str,
    input_edges: edge_models.InputEdges,
    source: live.LiveNode | dict[str, Any],
) -> dict[str, Any]:
    """
    Gather inputs for a dynamic subgraph child.

    *source* can be a LiveNode (reads from ``input_ports``) or a plain dict
    (used by the while-node where current values are tracked in a dict).
    """
    result: dict[str, Any] = {}
    for target, edge_source in input_edges.items():
        if target.node == child_label:
            if isinstance(source, dict):
                result[target.port] = source[edge_source.port]
            else:
                result[target.port] = source.input_ports[edge_source.port].value
    return result


def _evaluate_condition(
    case: helper_models.ConditionalCase,
    cond_node: live.LiveNode,
) -> bool:
    if case.condition_output is not None:
        return bool(cond_node.output_ports[case.condition_output].value)
    output_name = next(iter(cond_node.output_ports))
    return bool(cond_node.output_ports[output_name].value)


def _populate_prospective_outputs(
    node: live.FlowControl,
    prospective_output_edges: dict[
        edge_models.OutputTarget, list[edge_models.SourceHandle]
    ],
    active_label: str,
) -> None:
    """Wire outputs from the branch that actually executed."""
    for target, sources in prospective_output_edges.items():
        for source in sources:
            if source.node == active_label and source.node in node.nodes:
                child = node.nodes[source.node]
                node.output_ports[target.port].value = child.output_ports[
                    source.port
                ].value
                break
