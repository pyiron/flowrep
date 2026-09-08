"""
Build a graphviz-free :class:`~flowrep.drawing.model.DrawGraph` from a
prospective :class:`~flowrep.base_models.NodeRecipe`.

Free of any graphviz import, so building a drawing works in a bare install.
Flow-control recipe types (``for_each``, ``while``, ``if``, ``try``) expand into
their condition and body nodes, subject to the ``depth`` limit, just like a
``WorkflowRecipe`` expands into its own nodes.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

from flowrep import base_models, edge_models, lexical
from flowrep.drawing import model, style
from flowrep.prospective import (
    atomic_recipe,
    constant_recipe,
    for_recipe,
    helper_models,
    if_recipe,
    try_recipe,
    while_recipe,
    workflow_recipe,
)


def build(recipe: base_models.NodeRecipe, depth: int = 1) -> model.DrawGraph:
    """Build a drawing of *recipe*, expanding composites down to *depth*.

    The root always expands (if it is a composite); ``depth`` counts further
    generations below the root's children. Raises ``ValueError`` if ``depth``
    is negative.
    """
    if depth < 0:
        raise ValueError(f"depth must be >= 0, got {depth}")
    return _build(recipe, path="", label=recipe.type.value, depth=depth)


def _build(
    recipe: base_models.NodeRecipe, path: str, label: str, depth: int
) -> model.DrawNode:
    """Build a single node, recursing into children when it is an expanded composite."""
    inputs, outputs = _ports(recipe)
    children: tuple[model.DrawNode, ...] = ()
    edges: tuple[model.DrawEdge, ...] = ()
    groups: tuple[model.DrawGroup, ...] = ()
    match recipe:
        case atomic_recipe.AtomicRecipe() | constant_recipe.ConstantRecipe():
            pass
        case workflow_recipe.WorkflowRecipe():
            if depth >= 0:
                children = _build_children(recipe.nodes, path, depth)
                edges = _workflow_edges(recipe)
        case for_recipe.ForEachRecipe():
            if depth >= 0:
                children, edges = _build_for_each(recipe, path, depth)
        case while_recipe.WhileRecipe():
            if depth >= 0:
                children, edges = _build_while(recipe, path, depth)
        case if_recipe.IfRecipe():
            if depth >= 0:
                children, edges, groups = _build_if(recipe, path, depth)
        case try_recipe.TryRecipe():
            if depth >= 0:
                children, edges, groups = _build_try(recipe, path, depth)
        case _:
            raise TypeError(f"Unrecognized recipe type: {recipe}")
    return model.DrawNode(
        path=path,
        label=label,
        kind=recipe.type,
        subtitle=style.subtitle_for(recipe),
        inputs=inputs,
        outputs=outputs,
        children=children,
        edges=edges,
        groups=groups,
    )


def _ports(
    recipe: base_models.NodeRecipe,
) -> tuple[tuple[model.DrawPort, ...], tuple[model.DrawPort, ...]]:
    """Convert a recipe's port labels to draw ports.

    Prospectively there is no annotation to show, so ``hint`` is always
    ``None``. ``badge`` starts ``None`` here; flow-control builders overwrite
    it afterwards on the specific ports that carry a badge (an iterated body
    input, or a condition's evaluated output).
    """
    defaults = set(recipe.inputs_with_defaults)
    inputs = tuple(
        model.DrawPort(label=label, has_default=label in defaults)
        for label in recipe.inputs
    )
    outputs = tuple(model.DrawPort(label=label) for label in recipe.outputs)
    return inputs, outputs


def _child_path(parent_path: str, label: str) -> str:
    """The lexical path of a child node given its parent's path."""
    return lexical.join(parent_path, label)


def _build_children(
    nodes: Mapping[base_models.Label, base_models.NodeRecipe],
    parent_path: str,
    depth: int,
) -> tuple[model.DrawNode, ...]:
    """Build every child of a composite, one generation shallower."""
    return tuple(
        _build(
            child_recipe, _child_path(parent_path, child_label), child_label, depth - 1
        )
        for child_label, child_recipe in nodes.items()
    )


def _workflow_edges(
    recipe: workflow_recipe.WorkflowRecipe,
) -> tuple[model.DrawEdge, ...]:
    """All edges of a workflow recipe: parent-in, sibling, and parent-out."""
    return (
        _convert_input_edges(recipe.input_edges)
        + _convert_sibling_edges(recipe.edges)
        + _convert_output_edges(recipe.output_edges)
    )


def _convert_input_edges(edges: edge_models.InputEdges) -> tuple[model.DrawEdge, ...]:
    """A parent input source, drawn to a real child target."""
    return tuple(
        model.DrawEdge(
            source=model.PortRef("", base_models.IOTypes.INPUTS, source.port),
            target=model.PortRef(target.node, base_models.IOTypes.INPUTS, target.port),
        )
        for target, source in edges.items()
    )


def _convert_sibling_edges(edges: edge_models.Edges) -> tuple[model.DrawEdge, ...]:
    """A real child source, drawn to a real child target."""
    return tuple(
        model.DrawEdge(
            source=model.PortRef(source.node, base_models.IOTypes.OUTPUTS, source.port),
            target=model.PortRef(target.node, base_models.IOTypes.INPUTS, target.port),
        )
        for target, source in edges.items()
    )


def _convert_output_edges(
    edges: edge_models.OutputEdges, *, conditional: bool = False
) -> tuple[model.DrawEdge, ...]:
    """A real child source or a parent-input passthrough, drawn to a parent output."""
    return tuple(
        model.DrawEdge(
            source=_output_edge_source(source),
            target=model.PortRef("", base_models.IOTypes.OUTPUTS, target.port),
            conditional=conditional,
        )
        for target, source in edges.items()
    )


def _output_edge_source(
    source: edge_models.SourceHandle | edge_models.InputSource,
) -> model.PortRef:
    """The source endpoint for an output edge: a child output, or a parent passthrough.

    A ``None`` node is what marks a handle as referring to the enclosing node's
    own IO rather than to a child, so that -- not the handle's class -- is the
    thing to branch on.
    """
    if source.node is None:
        return model.PortRef("", base_models.IOTypes.INPUTS, source.port)
    return model.PortRef(source.node, base_models.IOTypes.OUTPUTS, source.port)


def _badge_port(
    ports: tuple[model.DrawPort, ...], label: str, badge: str
) -> tuple[model.DrawPort, ...]:
    """Rebuild *ports*, tagging the one named *label* with *badge*."""
    return tuple(
        dataclasses.replace(port, badge=badge) if port.label == label else port
        for port in ports
    )


def _evaluated_output(case: helper_models.ConditionalCase) -> str:
    """The condition output that decides the case: explicit, or the sole one."""
    return case.condition_output or case.condition.recipe.outputs[0]


def _flatten_prospective_output_edges(
    edges: Mapping[edge_models.OutputTarget, list[edge_models.SourceHandle]],
) -> tuple[model.DrawEdge, ...]:
    """Every candidate source of a fan-in output, each drawn ``conditional=True``.

    Reuses :func:`_convert_output_edges` per candidate rather than duplicating
    its endpoint logic.
    """
    result: tuple[model.DrawEdge, ...] = ()
    for target, sources in edges.items():
        for source in sources:
            result += _convert_output_edges({target: source}, conditional=True)
    return result


def _build_for_each(
    recipe: for_recipe.ForEachRecipe, path: str, depth: int
) -> tuple[tuple[model.DrawNode, ...], tuple[model.DrawEdge, ...]]:
    """One badged body child, plus its input and (possibly transferred) output edges."""
    label = recipe.body_node.label
    child = _build(
        recipe.body_node.recipe, _child_path(path, label), f"{label}_n", depth - 1
    )
    for port_label, badge in (
        *((port, "nested") for port in recipe.nested_ports),
        *((port, "zipped") for port in recipe.zipped_ports),
    ):
        child = dataclasses.replace(
            child, inputs=_badge_port(child.inputs, port_label, badge)
        )
    edges = _convert_input_edges(recipe.input_edges) + _convert_output_edges(
        recipe.output_edges
    )
    return (child,), edges


def _build_while(
    recipe: while_recipe.WhileRecipe, path: str, depth: int
) -> tuple[tuple[model.DrawNode, ...], tuple[model.DrawEdge, ...]]:
    """Condition and body children, the recipe's inferred back-edges, and the
    conditional input-to-output fallback that stands in when the body never runs."""
    case = recipe.case
    cond = _build(
        case.condition.recipe,
        _child_path(path, case.condition.label),
        f"{case.condition.label}_i",
        depth - 1,
    )
    cond = dataclasses.replace(
        cond, outputs=_badge_port(cond.outputs, _evaluated_output(case), "test")
    )
    body = _build(
        case.body.recipe,
        _child_path(path, case.body.label),
        f"{case.body.label}_i",
        depth - 1,
    )
    edges = (
        _convert_input_edges(recipe.input_edges)
        + _convert_output_edges(recipe.output_edges)
        + _convert_sibling_edges(recipe.body_body_edges)
        + _convert_sibling_edges(recipe.body_condition_edges)
        + _fallback_edges(recipe.outputs)
    )
    return (cond, body), edges


def _fallback_edges(outputs: base_models.Labels) -> tuple[model.DrawEdge, ...]:
    """A conditional parent-input-to-parent-output edge per while output.

    Valid because ``WhileRecipe`` validates ``outputs`` as a subset of
    ``inputs``: each of these inputs is the fallback value if the body never
    executes.
    """
    return tuple(
        model.DrawEdge(
            source=model.PortRef("", base_models.IOTypes.INPUTS, label),
            target=model.PortRef("", base_models.IOTypes.OUTPUTS, label),
            conditional=True,
        )
        for label in outputs
    )


def _build_if(
    recipe: if_recipe.IfRecipe, path: str, depth: int
) -> tuple[
    tuple[model.DrawNode, ...], tuple[model.DrawEdge, ...], tuple[model.DrawGroup, ...]
]:
    """Every case's condition and body, plus the else case, grouped by case."""
    children: list[model.DrawNode] = []
    groups: list[model.DrawGroup] = []
    for i, case in enumerate(recipe.cases):
        cond, body = _build_conditional_case(case, path, depth)
        children.extend((cond, body))
        groups.append(
            model.DrawGroup(f"case {i}", (case.condition.label, case.body.label))
        )
    if recipe.else_case is not None:
        else_case = recipe.else_case
        children.append(
            _build(
                else_case.recipe,
                _child_path(path, else_case.label),
                else_case.label,
                depth - 1,
            )
        )
        groups.append(model.DrawGroup("else", (else_case.label,)))
    edges = _convert_input_edges(
        recipe.input_edges
    ) + _flatten_prospective_output_edges(recipe.prospective_output_edges)
    return tuple(children), edges, tuple(groups)


def _build_conditional_case(
    case: helper_models.ConditionalCase, parent_path: str, depth: int
) -> tuple[model.DrawNode, model.DrawNode]:
    """A case's condition (badged) and body, as plain (undecorated) children."""
    cond = _build(
        case.condition.recipe,
        _child_path(parent_path, case.condition.label),
        case.condition.label,
        depth - 1,
    )
    cond = dataclasses.replace(
        cond, outputs=_badge_port(cond.outputs, _evaluated_output(case), "test")
    )
    body = _build(
        case.body.recipe,
        _child_path(parent_path, case.body.label),
        case.body.label,
        depth - 1,
    )
    return cond, body


def _build_try(
    recipe: try_recipe.TryRecipe, path: str, depth: int
) -> tuple[
    tuple[model.DrawNode, ...], tuple[model.DrawEdge, ...], tuple[model.DrawGroup, ...]
]:
    """The try node, plus one body per exception case, grouped accordingly."""
    try_node = recipe.try_node
    children = [
        _build(
            try_node.recipe,
            _child_path(path, try_node.label),
            try_node.label,
            depth - 1,
        )
    ]
    groups = [model.DrawGroup("try", (try_node.label,))]
    for case in recipe.exception_cases:
        children.append(
            _build(
                case.body.recipe,
                _child_path(path, case.body.label),
                case.body.label,
                depth - 1,
            )
        )
        groups.append(model.DrawGroup(_except_label(case), (case.body.label,)))
    edges = _convert_input_edges(
        recipe.input_edges
    ) + _flatten_prospective_output_edges(recipe.prospective_output_edges)
    return tuple(children), edges, tuple(groups)


def _except_label(case: helper_models.ExceptionCase) -> str:
    """``"except "`` followed by each caught exception's bare name, comma-joined."""
    names = ", ".join(
        exception.fully_qualified_name.rsplit(".", 1)[-1]
        for exception in case.exceptions
    )
    return f"except {names}"
