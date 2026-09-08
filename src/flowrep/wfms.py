"""
This module holds a prototypical, minimal Workflow Management System (WfMS) for
flowrep recipes.

Intended for use in tests and documentation, and as an example implementation to which
fully-fledged WfMS can refer.
"""

from __future__ import annotations

import math
from collections.abc import Collection
from typing import Any, cast

from pyiron_snippets import retrieve

from flowrep import base_models, edge_models, subgraph_validation, transformers
from flowrep.parsers import label_helpers
from flowrep.prospective import (
    atomic_recipe,
    constant_recipe,
    for_recipe,
    helper_models,
    if_recipe,
    try_recipe,
    union_types,
    while_recipe,
    workflow_recipe,
)
from flowrep.retrospective import datastructures


def _unsupported_recipe(recipe: Any) -> TypeError:
    return TypeError(f"Unsupported recipe type: {type(recipe).__name__}")


def run_recipe(
    recipe: union_types.RecipeDiscrimination, **kwargs: Any
) -> datastructures.NodeData:
    """
    Execute a flowrep recipe, returning a populated :class:`LiveNode`.

    All inputs are passed as keyword arguments matching the recipe's input port names.
    Inputs backed by a python default may be omitted; anything else must be supplied.
    """
    if not isinstance(recipe, base_models.NodeRecipe):
        # Guard before binding, which needs the recipe's input labels
        raise _unsupported_recipe(recipe)
    kwargs = variadic_to_inputs(recipe, **kwargs)
    match recipe:
        case atomic_recipe.AtomicRecipe():
            return _run_atomic(recipe, **kwargs)
        case constant_recipe.ConstantRecipe():
            return _run_constant(recipe, **kwargs)
        case workflow_recipe.WorkflowRecipe():
            return _run_workflow(recipe, **kwargs)
        case for_recipe.ForEachRecipe():
            return _run_for(recipe, **kwargs)
        case if_recipe.IfRecipe():
            return _run_if(recipe, **kwargs)
        case try_recipe.TryRecipe():
            return _run_try(recipe, **kwargs)
        case while_recipe.WhileRecipe():
            return _run_while(recipe, **kwargs)
        case _:
            raise _unsupported_recipe(recipe)


# ---------------------------------------------------------------------------
# Atomic
# ---------------------------------------------------------------------------


def _run_atomic(
    recipe: atomic_recipe.AtomicRecipe, **kwargs: Any
) -> datastructures.AtomicData:
    node = datastructures.AtomicData.from_recipe(recipe)
    _populate_input_ports(node, kwargs)
    result = _call_atomic(node)
    _store_atomic_outputs(node, result)
    return node


def _run_constant(
    recipe: constant_recipe.ConstantRecipe, **kwargs: Any
) -> datastructures.ConstantData:
    # A constant has no inputs; its output value is fixed and pre-filled by from_recipe.
    return datastructures.ConstantData.from_recipe(recipe)


def _call_atomic(node: datastructures.AtomicData) -> Any:
    """
    Invoke the underlying function, respecting positional-only parameter kinds.

    Values are drawn from the input data ports; if a port has no value, its
    default is used.  A :class:`ValueError` is raised when neither is available.
    """
    recipe = node.recipe
    assert isinstance(recipe, atomic_recipe.AtomicRecipe)

    positional: list[Any] = []
    keyword: dict[str, Any] = {}

    for name in recipe.inputs:
        port = node.input_ports[name]
        val = (
            port.value
            if not isinstance(port.value, datastructures.NotData)
            else port.default
        )
        if isinstance(val, datastructures.NotData):
            raise ValueError(f"Input port '{name}' has no value and no default")

        kind = recipe.reference.restricted_input_kinds.get(name)
        if kind == base_models.RestrictedParamKind.POSITIONAL_ONLY:
            positional.append(val)
        else:
            keyword[name] = val

    return node.function(*positional, **keyword)


def _store_atomic_outputs(node: datastructures.AtomicData, result: Any) -> None:
    recipe = node.recipe
    assert isinstance(recipe, atomic_recipe.AtomicRecipe)
    output_names = list(node.output_ports.keys())

    if len(output_names) == 1:
        node.output_ports[output_names[0]].value = result
    else:
        for name, val in zip(output_names, result, strict=True):
            node.output_ports[name].value = val


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


def _run_workflow(
    recipe: workflow_recipe.WorkflowRecipe, **kwargs: Any
) -> datastructures.DagData:
    node = datastructures.DagData.from_recipe(recipe)
    _populate_input_ports(node, kwargs)

    for child_label in _topo_sort_children(recipe):
        child_recipe = recipe.nodes[child_label]
        child_inputs = _gather_child_inputs(
            child_label,
            child_recipe.inputs,
            recipe.input_edges,
            recipe.edges,
            node,
        )
        child_node = run_recipe(child_recipe, **child_inputs)
        node.nodes[child_label] = child_node  # Overwrite with _executed_ child

    _populate_outputs_from_edges(node, recipe.output_edges)
    return node


def _topo_sort_children(recipe: workflow_recipe.WorkflowRecipe) -> list[str]:
    """Kahn's algorithm over sibling edges; deterministic tie-breaking by label."""
    return subgraph_validation.topological_sort(
        recipe.nodes,
        [(source.node, target.node) for target, source in recipe.edges.items()],
        tie_breaker=lambda label: label,
        cycle_message=(
            "Cycle detected in workflow edges. This should have been caught by "
            "the underlying recipe validation. Please raise a GitHub issue "
            "reporting how you got here!"
        ),
    )


def _gather_child_inputs(
    child_label: base_models.Label,
    child_inputs: base_models.Labels,
    input_edges: edge_models.InputEdges,
    edges: edge_models.Edges,
    parent: datastructures.CompositeData,
) -> dict[str, Any]:
    """
    Resolve input values for a child node from the parent's input ports and sibling
    output ports, according to the given edges.

    Ports not covered by any edge are omitted -- the child's own defaults (if any)
    will be used downstream.

    Taking the edges explicitly rather than reading them off a recipe is what lets a
    for-node, whose edges only exist at runtime, share this resolution path with a
    static workflow.
    """
    inputs: dict[str, Any] = {}

    for port in child_inputs:
        th = edge_models.TargetHandle(node=child_label, port=port)

        if th in input_edges:
            inputs[port] = parent.input_ports[input_edges[th].port].get_data()
        elif th in edges:
            sibling_source = edges[th]
            sibling = parent.nodes[sibling_source.node]
            inputs[port] = sibling.output_ports[sibling_source.port].value
        # else: port has a default on the child, _call_atomic will handle it

    return inputs


def _populate_outputs_from_edges(
    node: datastructures.CompositeData,
    output_edges: edge_models.OutputEdges,
) -> None:
    """Fill the composite's own output ports by following its output edges."""
    for target, source in output_edges.items():
        if source.node is None:
            val = node.input_ports[source.port].get_data()
        else:
            val = node.nodes[source.node].output_ports[source.port].value
        node.output_ports[target.port].value = val


# ---------------------------------------------------------------------------
# For
# ---------------------------------------------------------------------------


def _iterated_value(
    node: datastructures.ForEachData, for_port: str, body_port: str
) -> Collection:
    """
    The value to scatter over one iterated port.

    A for-node cannot fall back on a default here the way an atomic child can -- there
    is nothing to iterate -- so an unfilled port is reported directly, rather than
    leaking a bare ``NotData`` into ``itertools.product``.
    """
    value = node.input_ports[for_port].value
    if isinstance(value, datastructures.NotData):
        raise ValueError(
            f"Iterated input '{for_port}' (body port '{body_port}') has no value to "
            f"iterate over"
        )
    return cast(Collection, value)


def _scatter_label(parent_port: base_models.Label) -> base_models.Label:
    return f"scatter_{parent_port}"


def _aggregate_label(output_port: base_models.Label) -> base_models.Label:
    return f"aggregate_{output_port}"


def _body_to_parent_ports(
    recipe: for_recipe.ForEachRecipe, body_ports: base_models.Labels
) -> dict[base_models.Label, base_models.Label]:
    """Map iterated body port names to the for-node input ports that feed them."""
    body_label = recipe.body_node.label
    return {
        port: recipe.input_edges[
            edge_models.TargetHandle(node=body_label, port=port)
        ].port
        for port in body_ports
    }


def _nested_strides(
    total_steps: int, nested_lengths: dict[base_models.Label, int]
) -> dict[base_models.Label, int]:
    """
    Per-port strides for mixed-radix decomposition of the body index.

    Nested ports are the outer dimensions, in ``nested_ports`` order; zipped ports
    occupy the innermost dimension (stride 1) and are not represented here.
    """
    if total_steps == 0:
        return {}  # No body indices exist, so there is nothing to decompose

    strides: dict[base_models.Label, int] = {}
    running = total_steps
    for parent_port, length in nested_lengths.items():
        running //= length
        strides[parent_port] = running
    return strides


def _guard_generated_labels(
    recipe: for_recipe.ForEachRecipe,
    scattered: dict[base_models.Label, int],
    total_steps: int,
) -> None:
    """
    The for-node invents labels for its scatter, body and aggregate instances, and
    nothing in the recipe validators knows about them. Two generated labels coinciding
    would silently drop a node from the record; a body node named ``aggregate_ys``
    sitting beside a generated ``aggregate_ys_0`` would merely be unreadable. Refuse
    both, rather than corrupting the record or handing back a graph nobody can follow.
    """
    body_label = recipe.body_node.label
    generated = (
        [_scatter_label(port) for port in scattered]
        + [body_label]
        + [label_helpers.index_label(body_label, i) for i in range(total_steps)]
        + [_aggregate_label(port) for port in recipe.outputs]
    )
    seen: set[base_models.Label] = set()
    for label in generated:
        if label in seen:
            raise ValueError(
                f"For-node label collision: '{label}' would name two different "
                f"instances. Rename the body node or the colliding port."
            )
        seen.add(label)


def _guard_distinct_iterated_sources(
    nested_map: dict[base_models.Label, base_models.Label],
    zipped_map: dict[base_models.Label, base_models.Label],
) -> None:
    """
    One scatter node is built per *parent* port, so two iterated body ports fed by the
    same parent port would share a scatter and therefore an index -- collapsing what
    used to be two independent axes into one. Refuse rather than silently change what
    the loop iterates over.
    """
    parents = list(nested_map.values()) + list(zipped_map.values())
    duplicated = {port for port in parents if parents.count(port) > 1}
    if duplicated:
        raise ValueError(
            f"Iterated body ports must draw from distinct for-node inputs, but "
            f"{sorted(duplicated)} feeds more than one. Duplicate the input port to "
            f"iterate over the same data on two axes."
        )


def _run_for(
    recipe: for_recipe.ForEachRecipe, **kwargs: Any
) -> datastructures.ForEachData:
    """
    Execute a for-node as a real DAG: scatter nodes fan the iterated inputs out across
    body instances, and aggregate nodes gather their outputs back into lists.

    Nested ports drive a Cartesian product; zipped ports are iterated in lockstep.
    Broadcast (non-iterated) inputs are passed unchanged to every body instance.
    Transferred outputs collect the per-iteration value of a scattered input, so they
    are sourced from the scatter node rather than from a body output.

    Execution follows the recorded edges rather than running alongside them, so the
    record cannot drift from what actually happened.
    """
    node = datastructures.ForEachData.from_recipe(recipe)
    _populate_input_ports(node, kwargs)

    body_label = recipe.body_node.label
    body_recipe = recipe.body_node.recipe

    nested_map = _body_to_parent_ports(recipe, recipe.nested_ports)
    zipped_map = _body_to_parent_ports(recipe, recipe.zipped_ports)
    _guard_distinct_iterated_sources(nested_map, zipped_map)

    # Note that beyond insisting the value arrived at all, we simply take the length of
    # iterated input values, and let the user pay the price if runtime data is
    # non-compliant.
    nested_lengths = {
        parent_port: len(_iterated_value(node, parent_port, body_port))
        for body_port, parent_port in nested_map.items()
    }
    zipped_lengths = {
        parent_port: len(_iterated_value(node, parent_port, body_port))
        for body_port, parent_port in zipped_map.items()
    }
    if len(set(zipped_lengths.values())) > 1:
        raise ValueError("Zipped inputs must have equal lengths")

    zip_len = next(iter(zipped_lengths.values()), 1)
    total_steps = math.prod(nested_lengths.values()) * zip_len
    strides = _nested_strides(total_steps, nested_lengths)

    # An empty iterated port means zero body steps. Scatter nodes would then need zero
    # outputs, which no atomic recipe can express -- and nothing would consume them
    # anyway -- so only the aggregators get built, and they collect empty lists.
    scattered = {**nested_lengths, **zipped_lengths} if total_steps > 0 else {}
    _guard_generated_labels(recipe, scattered, total_steps)

    for parent_port, length in scattered.items():
        node.nodes[_scatter_label(parent_port)] = run_recipe(
            transformers.Transform1toN(length).recipe,
            **{
                transformers.Transform1toN.input_label: node.input_ports[
                    parent_port
                ].value
            },
        )

    node.input_edges = _for_input_edges(recipe, scattered, total_steps)
    node.edges = _for_edges(
        recipe,
        {**nested_map, **zipped_map},
        nested_lengths,
        strides,
        zip_len,
        total_steps,
    )
    node.output_edges = _for_output_edges(recipe)

    for i in range(total_steps):
        instance_label = label_helpers.index_label(body_label, i)
        node.nodes[instance_label] = run_recipe(
            body_recipe,
            **_gather_child_inputs(
                instance_label,
                body_recipe.inputs,
                node.input_edges,
                node.edges,
                node,
            ),
        )

    for output_port in recipe.outputs:
        aggregate_recipe = transformers.TransformNto1(total_steps).recipe
        label = _aggregate_label(output_port)
        node.nodes[label] = run_recipe(
            aggregate_recipe,
            **_gather_child_inputs(
                label,
                aggregate_recipe.inputs,
                node.input_edges,
                node.edges,
                node,
            ),
        )

    _populate_outputs_from_edges(node, node.output_edges)

    return node


def _for_input_edges(
    recipe: for_recipe.ForEachRecipe,
    scattered: dict[base_models.Label, int],
    total_steps: int,
) -> edge_models.InputEdges:
    """
    Parent inputs feed the scatter nodes, and broadcast inputs feed every body
    instance directly. Iterated ports are deliberately absent: they reach the bodies
    through a scatter node, which is the whole point of building one.
    """
    body_label = recipe.body_node.label
    input_edges: edge_models.InputEdges = {
        edge_models.TargetHandle(
            node=_scatter_label(parent_port),
            port=transformers.Transform1toN.input_label,
        ): edge_models.InputSource(port=parent_port)
        for parent_port in scattered
    }
    for target, source in recipe.input_edges.items():
        if target.port in recipe.iterated_ports:
            continue
        for i in range(total_steps):
            input_edges[
                edge_models.TargetHandle(
                    node=label_helpers.index_label(body_label, i), port=target.port
                )
            ] = edge_models.InputSource(port=source.port)
    return input_edges


def _for_edges(
    recipe: for_recipe.ForEachRecipe,
    iterated_map: dict[base_models.Label, base_models.Label],
    nested_lengths: dict[base_models.Label, int],
    strides: dict[base_models.Label, int],
    zip_len: int,
    total_steps: int,
) -> edge_models.Edges:
    """Scatters to bodies, and bodies (or scatters) to aggregators."""
    body_label = recipe.body_node.label

    def scatter_source(
        parent_port: base_models.Label, i: int
    ) -> edge_models.SourceHandle:
        if parent_port in nested_lengths:
            index = (i // strides[parent_port]) % nested_lengths[parent_port]
        else:
            index = i % zip_len
        return edge_models.SourceHandle(
            node=_scatter_label(parent_port),
            port=transformers.Transform1toN.output_label(index),
        )

    edges: edge_models.Edges = {
        edge_models.TargetHandle(
            node=label_helpers.index_label(body_label, i), port=body_port
        ): scatter_source(parent_port, i)
        for body_port, parent_port in iterated_map.items()
        for i in range(total_steps)
    }

    for target, source in recipe.output_edges.items():
        aggregate = _aggregate_label(target.port)
        for i in range(total_steps):
            handle = edge_models.TargetHandle(
                node=aggregate, port=transformers.TransformNto1.input_label(i)
            )
            if source.node is None:
                # Transferred output: the scattered input element itself
                edges[handle] = scatter_source(source.port, i)
            else:
                edges[handle] = edge_models.SourceHandle(
                    node=label_helpers.index_label(body_label, i), port=source.port
                )
    return edges


def _for_output_edges(recipe: for_recipe.ForEachRecipe) -> edge_models.OutputEdges:
    return {
        edge_models.OutputTarget(port=port): edge_models.SourceHandle(
            node=_aggregate_label(port),
            port=transformers.TransformNto1.output_label,
        )
        for port in recipe.outputs
    }


# ---------------------------------------------------------------------------
# While
# ---------------------------------------------------------------------------


def _record_while_child_edges(
    node: datastructures.WhileData,
    recipe: while_recipe.WhileRecipe,
    child_label: base_models.Label,
    iteration: int,
) -> None:
    """
    Record where one condition or body instance's inputs actually came from.

    Iteration 0 draws everything from the while-node's own inputs. Later iterations
    draw the looped ports from the previous body instance, and the rest -- ports the
    body never writes back -- from the while-node's inputs, still.
    """
    body_label = recipe.case.body.label
    looped = (
        recipe.body_condition_edges
        if child_label == recipe.case.condition.label
        else recipe.body_body_edges
    )
    instance = label_helpers.index_label(child_label, iteration)

    for target, source in recipe.input_edges.items():
        if target.node != child_label:
            continue
        retargeted = edge_models.TargetHandle(node=instance, port=target.port)
        if iteration > 0 and target in looped:
            node.edges[retargeted] = edge_models.SourceHandle(
                node=label_helpers.index_label(body_label, iteration - 1),
                port=looped[target].port,
            )
        else:
            node.input_edges[retargeted] = edge_models.InputSource(port=source.port)


def _record_while_output_edges(
    node: datastructures.WhileData,
    recipe: while_recipe.WhileRecipe,
    body_runs: int,
) -> None:
    """
    Record where the while-node's outputs actually came from: the last body instance,
    or -- if the body never ran -- the while-node's own inputs passed straight
    through, which validation guarantees is possible since outputs are a subset of
    inputs.
    """
    body_label = recipe.case.body.label
    for target, source in recipe.output_edges.items():
        if body_runs > 0:
            node.output_edges[target] = edge_models.SourceHandle(
                node=label_helpers.index_label(body_label, body_runs - 1),
                port=source.port,
            )
        else:
            node.output_edges[target] = edge_models.InputSource(port=target.port)


def _run_while(
    recipe: while_recipe.WhileRecipe, **kwargs: Any
) -> datastructures.WhileData:
    """
    Execute a while-node by repeatedly evaluating a condition and running a body.

    On each iteration the body outputs (which share names with a subset of inputs)
    feed back into the next condition/body evaluation.  If the condition is false on
    the first check, outputs are sourced from the initial input values.
    """
    node = datastructures.WhileData.from_recipe(recipe)
    _populate_input_ports(node, kwargs)

    cond_label = recipe.case.condition.label
    body_label = recipe.case.body.label
    cond_recipe = recipe.case.condition.recipe
    body_recipe = recipe.case.body.recipe

    # Working copy of current values — starts from inputs, body outputs update it
    current: dict[str, Any] = {
        name: node.input_ports[name].value for name in recipe.inputs
    }

    iteration = 0
    while True:
        # --- condition ---
        _record_while_child_edges(node, recipe, cond_label, iteration)
        cond_kwargs = _gather_dynamic_child_inputs(
            cond_label, recipe.input_edges, current
        )
        cond_node = run_recipe(cond_recipe, **cond_kwargs)
        node.nodes[label_helpers.index_label(cond_label, iteration)] = cond_node

        if not _evaluate_condition(recipe.case, cond_node):
            break

        # --- body ---
        _record_while_child_edges(node, recipe, body_label, iteration)
        body_kwargs = _gather_dynamic_child_inputs(
            body_label, recipe.input_edges, current
        )
        body_node = run_recipe(body_recipe, **body_kwargs)
        node.nodes[label_helpers.index_label(body_label, iteration)] = body_node

        # Feed body outputs back into current values
        for target, source in recipe.output_edges.items():
            current[target.port] = body_node.output_ports[source.port].value

        iteration += 1

    _record_while_output_edges(node, recipe, iteration)

    for name in recipe.outputs:
        node.output_ports[name].value = current[name]

    return node


# ---------------------------------------------------------------------------
# If
# ---------------------------------------------------------------------------


def _run_if(recipe: if_recipe.IfRecipe, **kwargs: Any) -> datastructures.IfData:
    """
    Execute an if-node by walking cases until a condition evaluates positively,
    then executing the matching body (or the else case).

    Output ports that have no source from the executed branch remain NOT_DATA.
    """
    node = datastructures.IfData.from_recipe(recipe)
    _populate_input_ports(node, kwargs)

    for case in recipe.cases:
        # --- condition ---
        cond_kwargs = _gather_dynamic_child_inputs(
            case.condition.label, recipe.input_edges, node
        )
        cond_node = run_recipe(case.condition.recipe, **cond_kwargs)
        node.nodes[case.condition.label] = cond_node

        if _evaluate_condition(case, cond_node):
            _execute_if_branch(node, recipe, case.body)
            _record_executed_input_edges(node, recipe.input_edges)
            return node

    # No case matched — try else
    if recipe.else_case is not None:
        _execute_if_branch(node, recipe, recipe.else_case)

    _record_executed_input_edges(node, recipe.input_edges)
    return node


def _execute_if_branch(
    node: datastructures.IfData,
    recipe: if_recipe.IfRecipe,
    branch: helper_models.LabeledRecipe,
) -> None:
    branch_kwargs = _gather_dynamic_child_inputs(branch.label, recipe.input_edges, node)
    branch_node = run_recipe(branch.recipe, **branch_kwargs)
    node.nodes[branch.label] = branch_node

    _populate_prospective_outputs(node, recipe.prospective_output_edges, branch.label)


# ---------------------------------------------------------------------------
# Try
# ---------------------------------------------------------------------------


def _run_try(recipe: try_recipe.TryRecipe, **kwargs: Any) -> datastructures.TryData:
    """
    Execute a try-node: run the try body and, on exception, walk exception cases
    for a matching handler.  If no handler matches, the exception propagates.
    """
    node = datastructures.TryData.from_recipe(recipe)
    _populate_input_ports(node, kwargs)

    try_kwargs = _gather_dynamic_child_inputs(
        recipe.try_node.label, recipe.input_edges, node
    )

    try:
        try_node = run_recipe(recipe.try_node.recipe, **try_kwargs)
        node.nodes[recipe.try_node.label] = try_node
        _populate_prospective_outputs(
            node, recipe.prospective_output_edges, recipe.try_node.label
        )
        _record_executed_input_edges(node, recipe.input_edges)
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
                handler_node = run_recipe(case.body.recipe, **handler_kwargs)
                node.nodes[case.body.label] = handler_node
                _populate_prospective_outputs(
                    node, recipe.prospective_output_edges, case.body.label
                )
                _record_executed_input_edges(node, recipe.input_edges)
                return node
        raise


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _populate_input_ports(
    node: datastructures.NodeData, values: dict[str, Any]
) -> None:
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
    source: datastructures.NodeData | dict[str, Any],
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
    cond_node: datastructures.NodeData,
) -> bool:
    if case.condition_output is not None:
        return bool(cond_node.output_ports[case.condition_output].value)
    output_name = next(iter(cond_node.output_ports))
    return bool(cond_node.output_ports[output_name].value)


def _record_executed_input_edges(
    node: datastructures.IfData | datastructures.TryData,
    input_edges: edge_models.InputEdges,
) -> None:
    """
    Record input edges for the children that actually ran -- the conditions evaluated
    plus the branch or handler chosen. The recipe describes every branch that might
    have run; the record describes the one that did.
    """
    for target, source in input_edges.items():
        if target.node in node.nodes:
            node.input_edges[target] = source


def _populate_prospective_outputs(
    node: datastructures.IfData | datastructures.TryData,
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
                node.output_edges[target] = source
                break


def variadic_to_inputs(recipe: base_models.NodeRecipe, /, *args, **kwargs):
    """
    Bind ``*args`` and ``**kwargs`` onto ``recipe.inputs``, as a helper for
    ``NodeRecipe.__call__`` implementations and the generic recipe runner.

    Every input must be filled except those in ``recipe.inputs_with_defaults``, which
    only recipes backed by an underlying python function have any of. Nothing else can
    supply a value after the fact, so an unfilled input is simply a missing one --
    which is exactly the invariant
    :func:`subgraph_validation.validate_nodes_are_fully_sourced` already holds children
    to, so validated recipes bind their own children by construction.

    Binding failures raise :class:`TypeError`, mirroring python's own behaviour for
    bad call signatures. (Deliberately not :class:`ValueError`: recipes catch
    exceptions by type, and a try-recipe handling ``ValueError`` must not be able to
    swallow its caller's mistake and quietly return partial data.)
    """
    who = f"{type(recipe).__name__}()"
    if len(args) > len(recipe.inputs):
        raise TypeError(
            f"One of your {who} calls takes {len(recipe.inputs)} inputs but "
            f"{len(args)} positional arguments were given -- its inputs are "
            f"{recipe.inputs}"
        )
    inputs = {}
    for label, val in zip(recipe.inputs, args, strict=False):
        inputs[label] = val
    for label, val in kwargs.items():
        if label in inputs:
            raise TypeError(
                f"One of your {who} calls got multiple values for input '{label}' -- "
                f"as a positional arg ({inputs[label]}) and as a kwarg ({val})"
            )
        if label in recipe.inputs:
            inputs[label] = val
        else:
            raise TypeError(
                f"One of your {who} calls got an unexpected input '{label}' -- its "
                f"inputs are {recipe.inputs}"
            )
    missing = [
        label
        for label in recipe.inputs
        if label not in inputs and label not in recipe.inputs_with_defaults
    ]
    if missing:
        raise TypeError(
            f"One of your {who} calls is missing {len(missing)} required "
            f"input: {missing}"
        )
    return inputs


def data_to_return(data: datastructures.NodeData):
    """A helper for ``NodeRecipe.__call__`` implementations"""
    returns = tuple(p.value for p in data.output_ports.values())
    if len(returns) == 1:
        return returns[0]
    else:
        return returns
