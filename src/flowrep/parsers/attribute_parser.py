"""The ``.attr`` link of a data-access chain (see :mod:`chain_parser`).

``obj.attr`` becomes one ``std.get_attr`` node whose ``name`` input is fed by a
``ConstantRecipe`` peer carrying the attribute name.
"""

from __future__ import annotations

import ast

from flowrep.parsers import constant_parser, symbol_scope
from flowrep.prospective import atomic_recipe, union_types


def _link(node: ast.expr) -> ast.Attribute:
    """Narrow a dispatched link; ``chain_parser`` only routes Attributes here."""
    assert isinstance(node, ast.Attribute)
    return node


class AttributeHandler:
    """One ``std.get_attr`` node per ``.attr`` link, plus its name constant."""

    @property
    def recipe(self) -> atomic_recipe.AtomicRecipe:
        # Imported lazily: `flowrep.std` imports the parsers, so a module-level
        # import here would cycle.
        from flowrep import std

        return std.get_attr.flowrep_recipe  # type: ignore[attr-defined,no-any-return]

    def port_base(self, link: ast.expr) -> str:
        return _link(link).attr

    def feed_key(
        self,
        link: ast.expr,
        symbol_map: symbol_scope.SymbolScope,
        nodes: union_types.Recipes,
        label: str,
        port: str,
    ) -> None:
        constant_parser.inject_constant(
            nodes, symbol_map, _link(link).attr, label, port
        )


HANDLER = AttributeHandler()
