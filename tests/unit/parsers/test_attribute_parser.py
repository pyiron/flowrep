import ast
import unittest

from flowrep import edge_models, std
from flowrep.parsers import attribute_parser, chain_parser, symbol_scope


def _make_source(node: str, port: str) -> edge_models.SourceHandle:
    return edge_models.SourceHandle(node=node, port=port)


def _expr(src: str) -> ast.expr:
    stmt = ast.parse(src).body[0]
    assert isinstance(stmt, ast.Expr)
    return stmt.value


class TestInjectionUsesTheRecipePorts(unittest.TestCase):
    """The injected edges must target whatever ports std.getattr_ actually declares."""

    def test_edges_and_handle_track_the_recipe(self):
        obj_port, name_port = std.get_attr.flowrep_recipe.inputs
        (attr_port,) = std.get_attr.flowrep_recipe.outputs
        scope = symbol_scope.SymbolScope(
            {"dc": _make_source("MyDataclass_0", "instance")}
        )
        nodes: dict = {}

        handle = chain_parser.inject_chain(_expr("dc.a"), scope, nodes)

        self.assertEqual(handle, _make_source("getattr_a_0", attr_port))
        self.assertIn(
            edge_models.TargetHandle(node="getattr_a_0", port=obj_port), scope.edges
        )
        self.assertIn(
            edge_models.TargetHandle(node="getattr_a_0", port=name_port), scope.edges
        )


class TestHandlerTracksTheRecipe(unittest.TestCase):
    """Labels and ports are read off std.get_attr, never spelled out."""

    def test_recipe_is_std_get_attr(self):
        self.assertIs(attribute_parser.HANDLER.recipe, std.get_attr.flowrep_recipe)

    def test_label_base_embeds_the_attribute_name(self):
        self.assertEqual(
            attribute_parser.HANDLER.label_base(_expr("dc.a")), "getattr_a"
        )

    def test_port_base_is_the_attribute_name(self):
        self.assertEqual(attribute_parser.HANDLER.port_base(_expr("dc.a")), "a")

    def test_feed_key_injects_the_name_constant(self):
        scope = symbol_scope.SymbolScope(
            {"dc": _make_source("MyDataclass_0", "instance")}
        )
        nodes: dict = {}
        attribute_parser.HANDLER.feed_key(
            _expr("dc.a"), scope, nodes, "getattr_a_0", "name"
        )
        self.assertEqual(set(nodes), {"constant_0"})
        self.assertEqual(nodes["constant_0"].constant, "a")
        self.assertEqual(
            scope.edges[edge_models.TargetHandle(node="getattr_a_0", port="name")],
            _make_source("constant_0", "constant"),
        )


if __name__ == "__main__":
    unittest.main()
