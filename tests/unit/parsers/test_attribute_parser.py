import ast
import unittest

from flowrep import edge_models
from flowrep.parsers import attribute_parser, symbol_scope
from flowrep.prospective import constant_recipe, std


def _make_input(port: str) -> edge_models.InputSource:
    return edge_models.InputSource(port=port)


def _make_source(node: str, port: str) -> edge_models.SourceHandle:
    return edge_models.SourceHandle(node=node, port=port)


def _expr(src: str) -> ast.expr:
    stmt = ast.parse(src).body[0]
    assert isinstance(stmt, ast.Expr)
    return stmt.value


class TestIsDataAttribute(unittest.TestCase):
    def test_true_for_single_access(self):
        scope = symbol_scope.SymbolScope(
            {"dc": _make_source("MyDataclass_0", "instance")}
        )
        self.assertTrue(attribute_parser.is_data_attribute(_expr("dc.a"), scope))

    def test_true_for_chain(self):
        scope = symbol_scope.SymbolScope(
            {"dc": _make_source("MyDataclass_0", "instance")}
        )
        self.assertTrue(attribute_parser.is_data_attribute(_expr("dc.a.val"), scope))

    def test_false_for_bare_name(self):
        scope = symbol_scope.SymbolScope(
            {"dc": _make_source("MyDataclass_0", "instance")}
        )
        self.assertFalse(attribute_parser.is_data_attribute(_expr("dc"), scope))

    def test_false_when_root_not_a_symbol(self):
        scope = symbol_scope.SymbolScope({})
        self.assertFalse(attribute_parser.is_data_attribute(_expr("np.pi"), scope))

    def test_false_when_chain_contains_call(self):
        scope = symbol_scope.SymbolScope({"f": _make_source("f_0", "output_0")})
        self.assertFalse(attribute_parser.is_data_attribute(_expr("f(x).a"), scope))

    def test_false_when_chain_contains_subscript(self):
        scope = symbol_scope.SymbolScope(
            {"dc": _make_source("MyDataclass_0", "instance")}
        )
        self.assertFalse(attribute_parser.is_data_attribute(_expr("dc[0].a"), scope))

    def test_symbol_scope_shadows_object_scope(self):
        """`os.path` is a data attribute access when `os` is a bound symbol."""
        scope = symbol_scope.SymbolScope({"os": _make_input("os")})
        self.assertTrue(attribute_parser.is_data_attribute(_expr("os.path"), scope))


class TestInjectAttributeChain(unittest.TestCase):
    def test_single_link_adds_getattr_and_constant(self):
        scope = symbol_scope.SymbolScope(
            {"dc": _make_source("MyDataclass_0", "instance")}
        )
        nodes = {}
        node = _expr("dc.a")
        assert isinstance(node, ast.Attribute)
        handle = attribute_parser.inject_attribute_chain(node, scope, nodes)

        self.assertEqual(set(nodes), {"getattr_a_0", "constant_0"})
        self.assertIs(nodes["getattr_a_0"], std.get_attr.flowrep_recipe)
        self.assertIsInstance(nodes["constant_0"], constant_recipe.ConstantRecipe)
        self.assertEqual(nodes["constant_0"].constant, "a")
        self.assertEqual(
            scope.edges,
            {
                edge_models.TargetHandle(node="getattr_a_0", port="obj"): _make_source(
                    "MyDataclass_0", "instance"
                ),
                edge_models.TargetHandle(node="getattr_a_0", port="name"): _make_source(
                    "constant_0", "constant"
                ),
            },
        )
        self.assertEqual(handle, _make_source("getattr_a_0", "attr"))

    def test_chain_of_two_links(self):
        scope = symbol_scope.SymbolScope(
            {"dc": _make_source("MyDataclass_0", "instance")}
        )
        nodes = {}
        node = _expr("dc.a.val")
        assert isinstance(node, ast.Attribute)
        handle = attribute_parser.inject_attribute_chain(node, scope, nodes)

        self.assertEqual(
            list(nodes), ["getattr_a_0", "constant_0", "getattr_val_0", "constant_1"]
        )
        self.assertEqual(
            scope.edges[edge_models.TargetHandle(node="getattr_val_0", port="obj")],
            _make_source("getattr_a_0", "attr"),
        )
        self.assertEqual(handle, _make_source("getattr_val_0", "attr"))

    def test_root_is_workflow_input_creates_input_edge(self):
        scope = symbol_scope.SymbolScope({"comp": _make_input("comp")})
        nodes = {}
        node = _expr("comp.val")
        assert isinstance(node, ast.Attribute)
        attribute_parser.inject_attribute_chain(node, scope, nodes)

        self.assertEqual(
            scope.input_edges,
            {
                edge_models.TargetHandle(node="getattr_val_0", port="obj"): _make_input(
                    "comp"
                )
            },
        )

    def test_label_collision_increments_suffix(self):
        scope = symbol_scope.SymbolScope(
            {"dc": _make_source("MyDataclass_0", "instance")}
        )
        nodes = {}
        node1 = _expr("dc.a")
        node2 = _expr("dc.a")
        assert isinstance(node1, ast.Attribute) and isinstance(node2, ast.Attribute)
        attribute_parser.inject_attribute_chain(node1, scope, nodes)
        handle2 = attribute_parser.inject_attribute_chain(node2, scope, nodes)
        self.assertEqual(handle2.node, "getattr_a_1")

    def test_accumulator_root_raises(self):
        scope = symbol_scope.SymbolScope({}, available_accumulators={"acc"})
        node = _expr("acc.a")
        assert isinstance(node, ast.Attribute)
        with self.assertRaises(ValueError) as ctx:
            attribute_parser.inject_attribute_chain(node, scope, {})
        self.assertIn("accumulator", str(ctx.exception).lower())


class TestRejectMethodCall(unittest.TestCase):
    def test_raises_for_method_call_on_symbol(self):
        scope = symbol_scope.SymbolScope(
            {"dc": _make_source("MyDataclass_0", "instance")}
        )
        call = _expr("dc.method(x)")
        assert isinstance(call, ast.Call)
        with self.assertRaises(ValueError) as ctx:
            attribute_parser.reject_method_call(call, scope)
        self.assertIn("method", str(ctx.exception).lower())

    def test_noop_for_module_call(self):
        scope = symbol_scope.SymbolScope(
            {"dc": _make_source("MyDataclass_0", "instance")}
        )
        call = _expr("library.my_add(x)")
        assert isinstance(call, ast.Call)
        attribute_parser.reject_method_call(call, scope)  # does not raise

    def test_noop_for_plain_call(self):
        scope = symbol_scope.SymbolScope({})
        call = _expr("f(x)")
        assert isinstance(call, ast.Call)
        attribute_parser.reject_method_call(call, scope)  # does not raise


class TestInjectionUsesTheRecipePorts(unittest.TestCase):
    """The injected edges must target whatever ports std.getattr_ actually declares."""

    def test_edges_and_handle_track_the_recipe(self):
        obj_port, name_port = std.get_attr.flowrep_recipe.inputs
        (attr_port,) = std.get_attr.flowrep_recipe.outputs
        scope = symbol_scope.SymbolScope(
            {"dc": _make_source("MyDataclass_0", "instance")}
        )
        nodes: dict = {}

        handle = attribute_parser.inject_attribute_chain(_expr("dc.a"), scope, nodes)

        self.assertEqual(handle, _make_source("getattr_a_0", attr_port))
        self.assertIn(
            edge_models.TargetHandle(node="getattr_a_0", port=obj_port), scope.edges
        )
        self.assertIn(
            edge_models.TargetHandle(node="getattr_a_0", port=name_port), scope.edges
        )


class TestRejectUnboundAttribute(unittest.TestCase):
    def _scope(self) -> symbol_scope.SymbolScope:
        return symbol_scope.SymbolScope(
            {"dc": _make_source("MyDataclass_0", "instance")}
        )

    def test_raises_for_chain_on_known_symbol(self):
        with self.assertRaises(ValueError) as ctx:
            attribute_parser.reject_unbound_attribute(
                _expr("dc.a"), self._scope(), "returned from a workflow"
            )
        message = str(ctx.exception)
        self.assertIn("returned from a workflow", message)
        self.assertIn("dc.a", message)
        self.assertIn("bind", message.lower())

    def test_raises_for_chain_of_chains(self):
        with self.assertRaises(ValueError) as ctx:
            attribute_parser.reject_unbound_attribute(
                _expr("dc.a.val"), self._scope(), "returned from a workflow"
            )
        self.assertIn("dc.a.val", str(ctx.exception))

    def test_noop_for_bare_symbol(self):
        attribute_parser.reject_unbound_attribute(
            _expr("dc"), self._scope(), "returned from a workflow"
        )

    def test_noop_for_chain_on_unknown_root(self):
        attribute_parser.reject_unbound_attribute(
            _expr("numpy.pi"), self._scope(), "returned from a workflow"
        )


class TestGeneratedPort(unittest.TestCase):
    def test_takes_the_outermost_attribute(self):
        self.assertEqual(
            attribute_parser.generate_port_name(_expr("dc.a.val"), []), "val_0"
        )

    def test_dodges_taken_names(self):
        self.assertEqual(
            attribute_parser.generate_port_name(_expr("dc.val"), ["val_0", "val_1"]),
            "val_2",
        )


if __name__ == "__main__":
    unittest.main()
