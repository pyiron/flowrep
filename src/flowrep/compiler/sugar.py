from __future__ import annotations

import keyword
from typing import Any, cast

from flowrep import edge_models
from flowrep.prospective import (
    atomic_recipe,
    constant_recipe,
    std,
    union_types,
    workflow_recipe,
)

OBJ_PORT = "obj"
NAME_PORT = "name"
ATTR_PORT = "attr"

# `LabeledRecipe.recipe` is a discriminated union; the cast narrows it for mypy.
_GETATTR_FQN = cast(
    atomic_recipe.AtomicRecipe, std.getattr_.recipe
).reference.info.fully_qualified_name


def is_std_getattr(node: union_types.RecipeDiscrimination) -> bool:
    """True if *node* is the standard-library attribute-access recipe.

    Matched on the referenced function's fully qualified name rather than on
    ``VersionInfo`` equality, so a recipe serialised under a different flowrep
    version is still recognised.
    """
    return (
        isinstance(node, atomic_recipe.AtomicRecipe)
        and node.reference.info.fully_qualified_name == _GETATTR_FQN
        and node.inputs == [OBJ_PORT, NAME_PORT]
        and node.outputs == [ATTR_PORT]
    )


def is_attribute_syntax(value: Any) -> bool:
    """True if *value* can be written after a ``.`` in Python source.

    Deliberately laxer than :func:`base_models.is_valid_label`, which also
    excludes the reserved port names ``inputs``/``outputs``. Those are perfectly
    legal attributes, and ``dc.inputs`` must compile back to ``dc.inputs``.
    """
    return (
        isinstance(value, str) and value.isidentifier() and not keyword.iskeyword(value)
    )


def attribute_name(label: str, recipe: workflow_recipe.WorkflowRecipe) -> str | None:
    """The identifier fed to *label*'s ``name`` port by a constant peer, else None."""
    source = recipe.edges.get(edge_models.TargetHandle(node=label, port=NAME_PORT))
    if source is None:
        return None
    peer = recipe.nodes.get(source.node)
    if not isinstance(peer, constant_recipe.ConstantRecipe):
        return None
    constant = peer.constant
    if isinstance(constant, str) and is_attribute_syntax(constant):
        return constant
    return None
