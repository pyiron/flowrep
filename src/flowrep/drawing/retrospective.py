"""
Build a graphviz-free :class:`~flowrep.drawing.model.DrawGraph` from a
retrospective :class:`~flowrep.retrospective.datastructures.NodeData`.

Free of any graphviz import, so building a drawing works in a bare install.
Every :class:`~flowrep.retrospective.datastructures.CompositeData` -- a run
``DagData`` as much as a run ``ForEachData``, ``IfData``, ``TryData`` or
``WhileData`` -- reads its ``.nodes``, ``.input_edges``, ``.edges`` and
``.output_edges`` identically. There is no per-flow-control-type branching;
the only dispatch is leaf versus composite.

This module shares nothing with :mod:`flowrep.drawing.prospective` but the IR
(:mod:`flowrep.drawing.model`) and the shared label formatting in
:mod:`flowrep.drawing.style`.
"""

from __future__ import annotations

from collections.abc import Mapping

from flowrep import base_models, edge_models, lexical
from flowrep.drawing import model, style
from flowrep.retrospective import datastructures


def build(data: datastructures.NodeData, depth: int = 0) -> model.DrawGraph:
    """Build a drawing of *data*, expanding composites down to *depth*.

    The root always expands (if it is a composite); ``depth`` counts further
    generations below the root's children. Raises ``ValueError`` if ``depth``
    is negative.
    """
    if depth < 0:
        raise ValueError(f"depth must be >= 0, got {depth}")
    return _build(data, path="", label=data.recipe.type.value, depth=depth)


def _build(
    data: datastructures.NodeData, path: str, label: str, depth: int
) -> model.DrawNode:
    """Build a single node, recursing into children when it is an expanded composite."""
    inputs, outputs = _ports(data)
    children: tuple[model.DrawNode, ...] = ()
    edges: tuple[model.DrawEdge, ...] = ()
    if isinstance(data, datastructures.CompositeData) and depth >= 0:
        children = _build_children(data.nodes, path, depth)
        edges = (
            _convert_input_edges(data.input_edges)
            + _convert_sibling_edges(data.edges)
            + _convert_output_edges(data.output_edges)
        )
    return model.DrawNode(
        path=path,
        label=label,
        kind=data.recipe.type,
        subtitle=style.subtitle_for(data.recipe),
        inputs=inputs,
        outputs=outputs,
        children=children,
        edges=edges,
        note=_note(children, edges),
    )


def _note(
    children: tuple[model.DrawNode, ...], edges: tuple[model.DrawEdge, ...]
) -> str | None:
    """A user-facing note when a composite expanded with children but no edges.

    The toy WfMS records actualized edges for the flow-control nodes it runs, but the
    retrospective format does not oblige every WfMS to, so a composite can still
    arrive with children and no wiring. Say so, rather than drawing a set of
    unconnected boxes that reads as a rendering fault.
    """
    if children and not edges:
        return "(no recorded edges)"
    return None


def _ports(
    data: datastructures.NodeData,
) -> tuple[tuple[model.DrawPort, ...], tuple[model.DrawPort, ...]]:
    """Convert a node's data ports to draw ports, annotation hints and all."""
    inputs = tuple(
        model.DrawPort(
            label=label,
            hint=style.format_annotation(port.annotation),
            has_default=port.default is not datastructures.NOT_DATA,
        )
        for label, port in data.input_ports.items()
    )
    outputs = tuple(
        model.DrawPort(label=label, hint=style.format_annotation(port.annotation))
        for label, port in data.output_ports.items()
    )
    return inputs, outputs


def _child_path(parent_path: str, label: str) -> str:
    """The lexical path of a child node given its parent's path."""
    return lexical.join(parent_path, label)


def _build_children(
    nodes: Mapping[base_models.Label, datastructures.NodeData],
    parent_path: str,
    depth: int,
) -> tuple[model.DrawNode, ...]:
    """Build every child of a composite, one generation shallower."""
    return tuple(
        _build(
            child_data, _child_path(parent_path, child_label), child_label, depth - 1
        )
        for child_label, child_data in nodes.items()
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


def _convert_output_edges(edges: edge_models.OutputEdges) -> tuple[model.DrawEdge, ...]:
    """A real child source or a parent-input passthrough, drawn to a parent output."""
    return tuple(
        model.DrawEdge(
            source=_output_edge_source(source),
            target=model.PortRef("", base_models.IOTypes.OUTPUTS, target.port),
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
