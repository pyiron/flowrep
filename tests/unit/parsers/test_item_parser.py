import ast
import unittest

from flowrep import edge_models, std
from flowrep.parsers import item_parser, symbol_scope
from flowrep.prospective import constant_recipe


def _make_source(node: str, port: str) -> edge_models.SourceHandle:
    return edge_models.SourceHandle(node=node, port=port)


def _expr(src: str) -> ast.expr:
    stmt = ast.parse(src).body[0]
    assert isinstance(stmt, ast.Expr)
    return stmt.value


class TestHandlerTracksTheRecipe(unittest.TestCase):
    """Labels and ports are read off std.getitem, never spelled out."""

    def test_recipe_is_std_getitem(self):
        self.assertIs(item_parser.HANDLER.recipe, std.getitem.flowrep_recipe)

    def test_label_base_matches_the_std_call_form(self):
        """`d[k]` must mint the label `std.getitem(d, k)` would: that identity is what
        lets the compiler round-trip item access without desugaring it."""
        self.assertEqual(item_parser.HANDLER.label_base(_expr("d[0]")), "getitem")

    def test_port_base_is_the_recipes_output_port(self):
        (item_port,) = std.getitem.flowrep_recipe.outputs
        self.assertEqual(item_parser.HANDLER.port_base(_expr("d[0]")), item_port)


class TestFeedKey(unittest.TestCase):
    def _scope(self) -> symbol_scope.SymbolScope:
        return symbol_scope.SymbolScope(
            {"d": _make_source("identity_0", "x"), "i": _make_source("identity_1", "x")}
        )

    def test_symbol_key_consumes_the_symbol(self):
        scope = self._scope()
        nodes: dict = {}
        item_parser.HANDLER.feed_key(_expr("d[i]"), scope, nodes, "getitem_0", "b")
        self.assertEqual(nodes, {})
        self.assertEqual(
            scope.edges[edge_models.TargetHandle(node="getitem_0", port="b")],
            _make_source("identity_1", "x"),
        )

    def test_constant_key_injects_a_constant_peer(self):
        scope = self._scope()
        nodes: dict = {}
        item_parser.HANDLER.feed_key(_expr("d['mass']"), scope, nodes, "getitem_0", "b")
        self.assertEqual(set(nodes), {"constant_0"})
        self.assertIsInstance(nodes["constant_0"], constant_recipe.ConstantRecipe)
        self.assertEqual(nodes["constant_0"].constant, "mass")

    def test_integer_key_injects_a_constant_peer(self):
        scope = self._scope()
        nodes: dict = {}
        item_parser.HANDLER.feed_key(_expr("d[0]"), scope, nodes, "getitem_0", "b")
        self.assertEqual(nodes["constant_0"].constant, 0)

    def test_slice_raises_and_names_the_future_node(self):
        with self.assertRaises(ValueError) as ctx:
            item_parser.HANDLER.feed_key(
                _expr("d[1:2]"), self._scope(), {}, "getitem_0", "b"
            )
        message = str(ctx.exception)
        self.assertIn("slice", message.lower())
        self.assertIn("std.slice", message)

    def test_chain_key_raises_and_names_the_bind_workaround(self):
        with self.assertRaises(ValueError) as ctx:
            item_parser.HANDLER.feed_key(
                _expr("d[dc.key]"), self._scope(), {}, "getitem_0", "b"
            )
        message = str(ctx.exception)
        self.assertIn("k = dc.key", message)
        self.assertIn("d[k]", message)


if __name__ == "__main__":
    unittest.main()
