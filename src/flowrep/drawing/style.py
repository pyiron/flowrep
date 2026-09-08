"""
Visual constants and label formatting for graph drawings.

Deliberately free of any graphviz import so it can be exercised without the
optional dependency.
"""

from __future__ import annotations

import typing

from flowrep import base_models

ELLIPSIS = "…"

NODE_PALETTE: dict[base_models.RecipeElementType, tuple[str, str]] = {
    base_models.RecipeElementType.ATOMIC: ("#dbeafe", "#1d4ed8"),
    base_models.RecipeElementType.WORKFLOW: ("#dcfce7", "#15803d"),
    base_models.RecipeElementType.CONSTANT: ("#ccfbf1", "#0f766e"),
    base_models.RecipeElementType.FOR_EACH: ("#fef3c7", "#b45309"),
    base_models.RecipeElementType.WHILE: ("#fce7f3", "#be185d"),
    base_models.RecipeElementType.IF: ("#f3e8ff", "#7e22ce"),
    base_models.RecipeElementType.TRY: ("#fee2e2", "#b91c1c"),
}

IO_FILL = "#e5e7eb"
IO_LINE = "#4b5563"
CONDITIONAL_LINE = "#6b7280"
GROUP_LINE = "#9ca3af"
SUBTITLE_COLOUR = "#374151"
HINT_COLOUR = "#6b7280"

NODE_LABEL_WRAP = 20
SUBTITLE_MAX = 28
PORT_LABEL_MAX = 14
ANNOTATION_MAX = 16


def truncate_right(text: str, limit: int) -> str:
    """Trim the tail, marking the cut with a trailing ellipsis."""
    if len(text) <= limit:
        return text
    return text[: limit - 1] + ELLIPSIS


def truncate_left(text: str, limit: int) -> str:
    """Trim the head, marking the cut with a leading ellipsis.

    Used for dotted paths, where the qualname at the end is the informative part.
    """
    if len(text) <= limit:
        return text
    return ELLIPSIS + text[-(limit - 1) :]


def wrap_label(text: str, width: int = NODE_LABEL_WRAP) -> list[str]:
    """Wrap an identifier, preferring breaks just after an underscore."""
    lines: list[str] = []
    remaining = text
    while len(remaining) > width:
        cut = remaining.rfind("_", 0, width + 1)
        cut = cut + 1 if cut > 0 else width
        lines.append(remaining[:cut])
        remaining = remaining[cut:]
    lines.append(remaining)
    return lines


def format_annotation(annotation: object | None) -> str | None:
    """Render a type hint compactly, or ``None`` when there is nothing to show."""
    if annotation is None:
        return None
    if isinstance(annotation, type) and typing.get_origin(annotation) is None:
        rendered = annotation.__name__
    else:
        rendered = str(annotation).replace("typing.", "")
    return truncate_right(rendered, ANNOTATION_MAX)


def subtitle_for(recipe: base_models.NodeRecipe) -> str | None:
    """The monospace line beneath a node's label, or ``None`` when there is none.

    Dispatches on ``recipe.type`` rather than importing the recipe classes, so
    this module stays free of any dependence on :mod:`flowrep.prospective`.
    """
    if recipe.type is base_models.RecipeElementType.CONSTANT:
        return truncate_right(repr(getattr(recipe, "constant", None)), SUBTITLE_MAX)
    fully_qualified_name = getattr(recipe, "fully_qualified_name", None)
    if fully_qualified_name is None:
        return None
    return truncate_left(fully_qualified_name, SUBTITLE_MAX)
