from __future__ import annotations

import keyword
from typing import Any, cast

from flowrep import edge_models, std
from flowrep.prospective import (
    atomic_recipe,
    constant_recipe,
    union_types,
    workflow_recipe,
)

# `LabeledRecipe.recipe` is a discriminated union; the cast narrows it for mypy.
_GETATTR = cast(atomic_recipe.AtomicRecipe, std.get_attr.flowrep_recipe)  # type: ignore[attr-defined]
_GETATTR_FQN = _GETATTR.reference.info.fully_qualified_name

# Port names are derived from the recipe, never spelled out: editing `std.get_attr` or
# `std.getitem` must not require editing strings anywhere else in the source. The two
# object ports differ -- `obj` for get_attr, `a` for getitem, which inherits
# `operator.getitem(a, b)`'s signature -- so each is named for its own link kind.
ATTR_OBJ_PORT, NAME_PORT = _GETATTR.inputs
(ATTR_PORT,) = _GETATTR.outputs

# `LabeledRecipe.recipe` is a discriminated union; the cast narrows it for mypy.
_GETITEM = cast(atomic_recipe.AtomicRecipe, std.getitem.flowrep_recipe)  # type: ignore[attr-defined]
_GETITEM_FQN = _GETITEM.reference.info.fully_qualified_name

ITEM_OBJ_PORT, KEY_PORT = _GETITEM.inputs
(ITEM_PORT,) = _GETITEM.outputs


def is_std_getattr(node: union_types.RecipeDiscrimination) -> bool:
    """True if *node* is the standard-library attribute-access recipe.

    Matched on the referenced function's fully qualified name rather than on
    ``VersionInfo`` equality, so a recipe serialised under a different flowrep
    version is still recognised.
    """
    return (
        isinstance(node, atomic_recipe.AtomicRecipe)
        and node.reference.info.fully_qualified_name == _GETATTR_FQN
        and node.inputs == _GETATTR.inputs
        and node.outputs == _GETATTR.outputs
    )


def is_std_getitem(node: union_types.RecipeDiscrimination) -> bool:
    """True if *node* is the standard-library item-access recipe.

    Matched on the referenced function's fully qualified name rather than on
    ``VersionInfo`` equality, so a recipe serialised under a different flowrep
    version is still recognised.
    """
    return (
        isinstance(node, atomic_recipe.AtomicRecipe)
        and node.reference.info.fully_qualified_name == _GETITEM_FQN
        and node.inputs == _GETITEM.inputs
        and node.outputs == _GETITEM.outputs
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
