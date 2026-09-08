"""
Render a graphviz-free :class:`~flowrep.drawing.model.DrawGraph` into a
``graphviz.Digraph``.

This is the only module in :mod:`flowrep.drawing` that imports ``graphviz``.
Building a :class:`~flowrep.drawing.model.DrawGraph`
(:mod:`flowrep.drawing.prospective`, :mod:`flowrep.drawing.retrospective`)
works in a bare install; only calling :func:`render` requires the optional
``graphviz`` package and a Graphviz ``dot`` binary.
"""

from __future__ import annotations

from pyiron_snippets import import_alarm

from flowrep import base_models, lexical
from flowrep.drawing import model, style

with import_alarm.ImportAlarm(
    "This tool requires the 'graphviz' package. Install it with "
    "`pip install flowrep[drawing]` (which also needs the Graphviz `dot` "
    "binary from your system package manager) or "
    "`conda install -c conda-forge python-graphviz` (which bundles it).",
    raise_exception=True,
) as _import_alarm:
    import graphviz

GRAPH_ATTR = {
    "rankdir": "LR",
    "compound": "true",
    "bgcolor": "white",
    "nodesep": "0.3",
    "ranksep": "0.5",
}
NODE_ATTR = {"shape": "plaintext"}

_TITLE_FONT_SIZE = "8"
_BADGE_FONT_SIZE = "7"
_GROUP_FONT_SIZE = "9"
_SPACER_WIDTH = "30"


@_import_alarm
def render(graph: model.DrawGraph) -> graphviz.Digraph:
    """Render *graph* -- and everything nested inside it -- as a ``Digraph``."""
    digraph = graphviz.Digraph(graph_attr=dict(GRAPH_ATTR), node_attr=dict(NODE_ATTR))
    _emit_node(digraph, graph)
    return digraph


def _escape(text: str) -> str:
    """Escape ``&``, ``<`` and ``>`` for use inside an HTML-like label."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _emit_node(context: graphviz.Digraph, node: model.DrawNode) -> None:
    """Draw *node* -- a leaf box, or a cluster containing its children -- into *context*."""
    if node.is_leaf:
        _emit_leaf(context, node)
    else:
        _emit_composite(context, node)


def _emit_leaf(context: graphviz.Digraph, node: model.DrawNode) -> None:
    """Draw *node* as a single ``plaintext`` node holding an HTML-like table."""
    context.node(node.path, label=_leaf_label(node))


def _leaf_label(node: model.DrawNode) -> str:
    """The HTML-like ``<TABLE>`` label for a leaf node box."""
    fill, line = style.NODE_PALETTE[node.kind]
    rows = [f'<TR><TD COLSPAN="3">{_title_cell(node)}</TD></TR>']
    row_count = max(len(node.inputs), len(node.outputs))
    for i in range(row_count):
        left = _port_cell(node.inputs[i], "i") if i < len(node.inputs) else "<TD></TD>"
        right = (
            _port_cell(node.outputs[i], "o") if i < len(node.outputs) else "<TD></TD>"
        )
        rows.append(f'<TR>{left}<TD WIDTH="{_SPACER_WIDTH}"></TD>{right}</TR>')
    table = (
        f'<TABLE STYLE="ROUNDED" BGCOLOR="{fill}" COLOR="{line}" BORDER="2" '
        f'CELLBORDER="0" CELLSPACING="5">{"".join(rows)}</TABLE>'
    )
    return f"<{table}>"


def _title_cell(node: model.DrawNode) -> str:
    """The bold, wrapped label plus the optional monospace subtitle."""
    lines = style.wrap_label(node.label)
    html = f"<B>{'<BR/>'.join(_escape(line) for line in lines)}</B>"
    if node.subtitle is not None:
        html += (
            f'<BR/><FONT FACE="monospace" POINT-SIZE="{_TITLE_FONT_SIZE}">'
            f"{_escape(node.subtitle)}</FONT>"
        )
    return html


def _port_content(port: model.DrawPort) -> str:
    """The (possibly italicised, possibly badged) label content of a port cell."""
    label = _escape(style.truncate_right(port.label, style.PORT_LABEL_MAX))
    if port.has_default:
        label = f"<I>{label}</I>"
    extra = port.badge if port.badge is not None else port.hint
    if extra is not None:
        label += (
            f'<BR/><FONT FACE="monospace" POINT-SIZE="{_BADGE_FONT_SIZE}" '
            f'COLOR="{style.HINT_COLOUR}">{_escape(extra)}</FONT>'
        )
    return label


def _port_cell(port: model.DrawPort, prefix: str) -> str:
    """A single rounded, grey-filled port cell, italicised and badged as needed."""
    return (
        f'<TD PORT="{prefix}_{port.label}" BGCOLOR="{style.IO_FILL}" '
        f'COLOR="{style.IO_LINE}" STYLE="ROUNDED" BORDER="1" CELLPADDING="3">'
        f"{_port_content(port)}</TD>"
    )


def _emit_composite(context: graphviz.Digraph, node: model.DrawNode) -> None:
    """Draw *node* as a filled cluster: its own IO boxes, its children, its edges."""
    with context.subgraph(name=f"cluster_{node.path}") as sub:
        fill, line = style.NODE_PALETTE[node.kind]
        sub.attr(
            label=_cluster_label(node),
            style="rounded,filled",
            fillcolor=fill,
            color=line,
            penwidth="2",
            margin="12",
        )
        _emit_io_boxes(sub, node, node.inputs, base_models.IOTypes.INPUTS)
        _emit_io_boxes(sub, node, node.outputs, base_models.IOTypes.OUTPUTS)
        _emit_children(sub, node)
        _emit_edges(sub, node)


def _cluster_label(node: model.DrawNode) -> str:
    """The cluster's own title, mirroring a leaf's title cell, plus an optional note."""
    html = _title_cell(node)
    if node.note is not None:
        html += f"<BR/><I>{_escape(node.note)}</I>"
    return f"<{html}>"


def _emit_io_boxes(
    context: graphviz.Digraph,
    node: model.DrawNode,
    ports: tuple[model.DrawPort, ...],
    io_type: base_models.IOTypes,
) -> None:
    """One small box per port in *ports*, rank-aligned together when there is more than one."""
    if not ports:
        return
    if len(ports) > 1:
        with context.subgraph() as rank_group:
            rank_group.attr(rank="same")
            for port in ports:
                _emit_io_box(rank_group, node, port, io_type)
    else:
        _emit_io_box(context, node, ports[0], io_type)


def _emit_io_box(
    context: graphviz.Digraph,
    node: model.DrawNode,
    port: model.DrawPort,
    io_type: base_models.IOTypes,
) -> None:
    """A composite's own IO port, drawn as a single-cell grey box just inside the wall."""
    box_id = model.PortRef(node.path, io_type, port.label).lexical_path
    table = (
        f'<TABLE STYLE="ROUNDED" BGCOLOR="{style.IO_FILL}" COLOR="{style.IO_LINE}" '
        f'BORDER="1" CELLBORDER="0" CELLSPACING="0">'
        f'<TR><TD PORT="p" CELLPADDING="3">{_port_content(port)}</TD></TR></TABLE>'
    )
    context.node(box_id, label=f"<{table}>")


def _emit_children(context: graphviz.Digraph, node: model.DrawNode) -> None:
    """Draw every child, nesting grouped ones inside a dashed group cluster.

    Groups are emitted in reverse declaration order: under ``rankdir=LR``,
    Graphviz stacks same-rank clusters bottom-up in declaration order, so
    emitting the first-declared group last places it at the top, matching
    evaluation order.
    """
    by_label = {lexical.split(child.path)[-1]: child for child in node.children}
    grouped_paths: set[str] = set()
    for index, group in reversed(list(enumerate(node.groups))):
        group_name = lexical.join(node.path, f"group_{index}")
        with context.subgraph(name=f"cluster_{group_name}") as group_sub:
            group_sub.attr(
                label=f"<<I>{_escape(group.label)}</I>>",
                fontsize=_GROUP_FONT_SIZE,
                style="rounded,dashed",
                color=style.GROUP_LINE,
            )
            for member in group.members:
                child = by_label[member]
                _emit_node(group_sub, child)
                grouped_paths.add(child.path)
    for child in node.children:
        if child.path not in grouped_paths:
            _emit_node(context, child)


def _emit_edges(context: graphviz.Digraph, node: model.DrawNode) -> None:
    """Draw every edge of *node*, cutting the ones that would close a cycle."""
    children_by_path = {child.path: child for child in node.children}
    back_edges = _back_edges(node.edges)
    for edge in node.edges:
        tail = _endpoint(node, children_by_path, edge.source, compass="e")
        head = _endpoint(node, children_by_path, edge.target, compass="w")
        attrs: dict[str, str] = {}
        if edge.conditional:
            attrs["style"] = "dashed"
            attrs["color"] = style.CONDITIONAL_LINE
        if edge in back_edges:
            attrs["constraint"] = "false"
        context.edge(tail, head, **attrs)


def _endpoint(
    composite: model.DrawNode,
    children_by_path: dict[str, model.DrawNode],
    ref: model.PortRef,
    *,
    compass: str,
) -> str:
    """The ``node:port:compass`` string addressing *ref* from inside *composite*.

    An empty ``node_path`` addresses *composite*'s own IO box. Otherwise it
    addresses a real child: a leaf child's own port cell, or -- when the
    child is itself an expanded composite -- that child's own IO box, which
    is as close as Graphviz gets to piercing a cluster wall.
    """
    if not ref.node_path:
        box_id = model.PortRef(composite.path, ref.io_type, ref.port).lexical_path
        return f"{box_id}:p:{compass}"
    child_path = lexical.join(composite.path, ref.node_path)
    child = children_by_path[child_path]
    if child.is_leaf:
        prefix = "i" if ref.io_type is base_models.IOTypes.INPUTS else "o"
        return f"{child_path}:{prefix}_{ref.port}:{compass}"
    box_id = model.PortRef(child_path, ref.io_type, ref.port).lexical_path
    return f"{box_id}:p:{compass}"


def _back_edges(edges: tuple[model.DrawEdge, ...]) -> set[model.DrawEdge]:
    """Depth-first cycle detection over *edges*, keyed by sibling node path.

    ``while`` back-edges make a cluster's own edge set cyclic, which would
    otherwise destroy ``rankdir=LR`` layout. This is a rendering-local layout
    concern, not represented in the IR.

    A real child collapses its input and output pins into one node identity,
    since the dependency that matters here is "does this child's output feed
    back into one of its own ancestors". A composite's own IO does not
    collapse that way: its inputs and outputs are the source and sink of the
    whole cluster, and treating them as one node would flag every ordinary
    input-to-output path as a cycle.
    """
    adjacency: dict[str, list[tuple[str, model.DrawEdge]]] = {}
    for edge in edges:
        adjacency.setdefault(_node_key(edge.source), []).append(
            (_node_key(edge.target), edge)
        )
    all_nodes = {_node_key(edge.source) for edge in edges} | {
        _node_key(edge.target) for edge in edges
    }
    visited: set[str] = set()
    in_progress: set[str] = set()
    back: set[model.DrawEdge] = set()

    def visit(current: str) -> None:
        visited.add(current)
        in_progress.add(current)
        for target, edge in adjacency.get(current, []):
            if target in in_progress:
                back.add(edge)
            elif target not in visited:
                visit(target)
        in_progress.discard(current)

    for start in all_nodes:
        if start not in visited:
            visit(start)
    return back


def _node_key(ref: model.PortRef) -> str:
    """The sibling-level node identity of *ref*, for cycle detection.

    A real child's own label, ignoring which of its ports is referenced; a
    composite's own input and its own output are distinct pseudo-nodes.
    """
    if ref.node_path:
        return ref.node_path
    return f"<{ref.io_type}>"
