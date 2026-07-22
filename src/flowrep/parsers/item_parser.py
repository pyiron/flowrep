"""The ``[key]`` link of a data-access chain (see :mod:`chain_parser`).

``d[k]`` becomes one ``std.getitem`` node. Its key comes from a symbol (``d[i]``) or a
``ConstantRecipe`` peer (``d[0]``, ``d["mass"]``).

Labels come from :func:`chain_parser._label_base`, so an injected node carries exactly
the label -- and its constant peer exactly the numbering -- that parsing
``std.getitem(d, k)`` would produce. That makes ``d[k]`` and ``std.getitem(d, k)`` the
*same recipe*, which is why item access round-trips through the compiler with no
desugaring: the compiler emits the call form, and re-parsing it lands back here.
"""

from __future__ import annotations

import ast

from flowrep.parsers import constant_parser, symbol_scope
from flowrep.prospective import atomic_recipe, union_types


def _link(node: ast.expr) -> ast.Subscript:
    """Narrow a dispatched link; ``chain_parser`` only routes Subscripts here."""
    assert isinstance(node, ast.Subscript)
    return node


class ItemHandler:
    """One ``std.getitem`` node per ``[key]`` link, plus any key constant."""

    @property
    def recipe(self) -> atomic_recipe.AtomicRecipe:
        # Imported lazily: `flowrep.std` imports the parsers, so a module-level
        # import here would cycle.
        from flowrep import std

        return std.getitem.flowrep_recipe  # type: ignore[attr-defined,no-any-return]

    def port_base(self, link: ast.expr) -> str:
        (item_port,) = self.recipe.outputs
        return item_port

    def feed_key(
        self,
        link: ast.expr,
        symbol_map: symbol_scope.SymbolScope,
        nodes: union_types.Recipes,
        label: str,
        port: str,
    ) -> None:
        key = _link(link).slice
        if isinstance(key, ast.Slice):
            raise ValueError(
                f"Workflow python definitions cannot parse slices yet: "
                f"'{ast.unparse(link)}'. A slice needs a `std.slice` node to carry its "
                f"start/stop/step, and that node does not exist yet. Index and key "
                f"access -- `d[0]`, `d['mass']`, `d[k]` -- are supported. In the "
                f"meantime, you could make a slice node for your application."
            )
        if isinstance(key, ast.Name):
            symbol_map.consume(key.id, label, port)
            return
        is_literal, value = constant_parser.try_parse_constant(key)
        if not is_literal:
            raise ValueError(
                f"An item key must be a symbol or a literal constant, but "
                f"'{ast.unparse(link)}' keys on {type(key).__name__}. Bind the key to "
                f"a symbol first -- e.g. `k = {ast.unparse(key)}` -- and then "
                f"`{ast.unparse(_link(link).value)}[k]`."
            )
        constant_parser.inject_constant(nodes, symbol_map, value, label, port)


HANDLER = ItemHandler()
