"""
A graphviz-free intermediate representation of a drawable graph.

Builders (:mod:`flowrep.drawing.prospective`, :mod:`flowrep.drawing.retrospective`)
produce this; :mod:`flowrep.drawing.render` consumes it. Keeping the two apart
means topology logic is testable without the optional drawing dependency.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

from flowrep import base_models, lexical


@dataclasses.dataclass(frozen=True)
class DrawPort:
    """A single IO port as it should appear inside a node box."""

    label: str
    hint: str | None = None
    has_default: bool = False
    badge: str | None = None


@dataclasses.dataclass(frozen=True)
class PortRef:
    """An edge endpoint. An empty ``node_path`` means the enclosing node's own IO."""

    node_path: str
    io_type: base_models.IOTypes
    port: str

    @property
    def lexical_path(self) -> str:
        return lexical.port_path(self.node_path, self.io_type, self.port)


@dataclasses.dataclass(frozen=True)
class DrawEdge:
    source: PortRef
    target: PortRef
    conditional: bool = False
    """One of several candidate sources; exactly one actualizes at runtime."""


@dataclasses.dataclass(frozen=True)
class DrawGroup:
    """A purely visual grouping of sibling nodes. Has no lexical path."""

    label: str
    members: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class DrawNode:
    path: str
    """Lexical path from the drawing root; empty for the root itself."""
    label: str
    """Displayed name, which may differ from the path tail (e.g. ``body_n``)."""
    kind: base_models.RecipeElementType
    subtitle: str | None
    inputs: tuple[DrawPort, ...]
    outputs: tuple[DrawPort, ...]
    children: tuple[DrawNode, ...]
    edges: tuple[DrawEdge, ...]
    groups: tuple[DrawGroup, ...] = ()
    note: str | None = None

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def walk(self) -> Iterator[DrawNode]:
        """Yield this node, then every descendant, depth-first."""
        yield self
        for child in self.children:
            yield from child.walk()


DrawGraph = DrawNode
