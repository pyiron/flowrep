"""
The public drawing callables.

Named ``interface`` rather than ``draw`` so the module does not shadow the
:func:`draw` function re-exported alongside it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flowrep import base_models
from flowrep.drawing import prospective, render, retrospective
from flowrep.retrospective import datastructures

if TYPE_CHECKING:
    import graphviz

PROSPECTIVE_DEPTH = 1
RETROSPECTIVE_DEPTH = 0


def draw_prospective(
    graph: base_models.NodeRecipe, depth: int = PROSPECTIVE_DEPTH
) -> graphviz.Digraph:
    """Draw a prospective recipe, expanding ``depth`` generations of subgraph."""
    return render.render(prospective.build(graph, depth=depth))


def draw_retrospective(
    graph: datastructures.NodeData, depth: int = RETROSPECTIVE_DEPTH
) -> graphviz.Digraph:
    """Draw a retrospective data object, expanding ``depth`` generations."""
    return render.render(retrospective.build(graph, depth=depth))


def draw(
    graph: base_models.NodeRecipe | datastructures.NodeData, depth: int | None = None
) -> graphviz.Digraph:
    """Draw either a recipe or a data object, dispatching on type.

    When ``depth`` is None the default of the dispatched-to drawer applies.
    """
    if isinstance(graph, base_models.NodeRecipe):
        return draw_prospective(
            graph, depth=PROSPECTIVE_DEPTH if depth is None else depth
        )
    if isinstance(graph, datastructures.NodeData):
        return draw_retrospective(
            graph, depth=RETROSPECTIVE_DEPTH if depth is None else depth
        )
    raise TypeError(
        f"Can only draw a {base_models.NodeRecipe.__name__} or a "
        f"{datastructures.NodeData.__name__}, but got {type(graph).__name__}: {graph!r}"
    )
