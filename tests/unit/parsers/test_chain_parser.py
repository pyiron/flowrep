import ast
import unittest

from flowrep import edge_models, std
from flowrep.parsers import chain_parser, symbol_scope
from flowrep.prospective import constant_recipe


def _make_input(port: str) -> edge_models.InputSource:
    return edge_models.InputSource(port=port)


def _make_source(node: str, port: str) -> edge_models.SourceHandle:
    return edge_models.SourceHandle(node=node, port=port)


def _expr(src: str) -> ast.expr:
    stmt = ast.parse(src).body[0]
    assert isinstance(stmt, ast.Expr)
    return stmt.value


class TestIsDataAccess(unittest.TestCase):
    def test_true_for_single_access(self):
        scope = symbol_scope.SymbolScope(
            {"dc": _make_source("MyDataclass_0", "instance")}
        )
        self.assertTrue(chain_parser.is_data_access(_expr("dc.a"), scope))

    def test_true_for_chain(self):
        scope = symbol_scope.SymbolScope(
            {"dc": _make_source("MyDataclass_0", "instance")}
        )
        self.assertTrue(chain_parser.is_data_access(_expr("dc.a.val"), scope))

    def test_false_for_bare_name(self):
        scope = symbol_scope.SymbolScope(
            {"dc": _make_source("MyDataclass_0", "instance")}
        )
        self.assertFalse(chain_parser.is_data_access(_expr("dc"), scope))

    def test_false_when_root_not_a_symbol(self):
        scope = symbol_scope.SymbolScope({})
        self.assertFalse(chain_parser.is_data_access(_expr("np.pi"), scope))

    def test_false_when_chain_contains_call(self):
        scope = symbol_scope.SymbolScope({"f": _make_source("f_0", "output_0")})
        self.assertFalse(chain_parser.is_data_access(_expr("f(x).a"), scope))

    def test_true_for_mixed_chain(self):
        """Item links are part of a chain now; this assertion is the inverse of the
        one attribute-only parsing carried."""
        scope = symbol_scope.SymbolScope(
            {"dc": _make_source("MyDataclass_0", "instance")}
        )
        self.assertTrue(chain_parser.is_data_access(_expr("dc[0].a"), scope))

    def test_true_for_bare_item_access(self):
        scope = symbol_scope.SymbolScope({"d": _make_source("identity_0", "x")})
        self.assertTrue(chain_parser.is_data_access(_expr("d[0]"), scope))

    def test_false_when_item_root_not_a_symbol(self):
        scope = symbol_scope.SymbolScope({})
        self.assertFalse(chain_parser.is_data_access(_expr("d[0]"), scope))

    def test_symbol_scope_shadows_object_scope(self):
        """`os.path` is a data attribute access when `os` is a bound symbol."""
        scope = symbol_scope.SymbolScope({"os": _make_input("os")})
        self.assertTrue(chain_parser.is_data_access(_expr("os.path"), scope))


class TestInjectChain(unittest.TestCase):
    def test_single_link_adds_getattr_and_constant(self):
        scope = symbol_scope.SymbolScope(
            {"dc": _make_source("MyDataclass_0", "instance")}
        )
        nodes = {}
        node = _expr("dc.a")
        assert isinstance(node, ast.Attribute)
        handle = chain_parser.inject_chain(node, scope, nodes)

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
        handle = chain_parser.inject_chain(node, scope, nodes)

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
        chain_parser.inject_chain(node, scope, nodes)

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
        chain_parser.inject_chain(node1, scope, nodes)
        handle2 = chain_parser.inject_chain(node2, scope, nodes)
        self.assertEqual(handle2.node, "getattr_a_1")

    def test_accumulator_root_raises(self):
        scope = symbol_scope.SymbolScope({}, available_accumulators={"acc"})
        node = _expr("acc.a")
        assert isinstance(node, ast.Attribute)
        with self.assertRaises(ValueError) as ctx:
            chain_parser.inject_chain(node, scope, {})
        self.assertIn("accumulator", str(ctx.exception).lower())


class TestRejectMethodCall(unittest.TestCase):
    def test_raises_for_method_call_on_symbol(self):
        scope = symbol_scope.SymbolScope(
            {"dc": _make_source("MyDataclass_0", "instance")}
        )
        call = _expr("dc.method(x)")
        assert isinstance(call, ast.Call)
        with self.assertRaises(ValueError) as ctx:
            chain_parser.reject_method_call(call, scope)
        self.assertIn("method", str(ctx.exception).lower())

    def test_noop_for_module_call(self):
        scope = symbol_scope.SymbolScope(
            {"dc": _make_source("MyDataclass_0", "instance")}
        )
        call = _expr("std.add(x)")
        assert isinstance(call, ast.Call)
        chain_parser.reject_method_call(call, scope)  # does not raise

    def test_noop_for_plain_call(self):
        scope = symbol_scope.SymbolScope({})
        call = _expr("f(x)")
        assert isinstance(call, ast.Call)
        chain_parser.reject_method_call(call, scope)  # does not raise


class TestRejectUnboundAccess(unittest.TestCase):
    def _scope(self) -> symbol_scope.SymbolScope:
        return symbol_scope.SymbolScope(
            {"dc": _make_source("MyDataclass_0", "instance")}
        )

    def test_raises_for_chain_on_known_symbol(self):
        with self.assertRaises(ValueError) as ctx:
            chain_parser.reject_unbound_access(
                _expr("dc.a"), self._scope(), "returned from a workflow"
            )
        message = str(ctx.exception)
        self.assertIn("returned from a workflow", message)
        self.assertIn("dc.a", message)
        self.assertIn("bind", message.lower())

    def test_raises_for_chain_of_chains(self):
        with self.assertRaises(ValueError) as ctx:
            chain_parser.reject_unbound_access(
                _expr("dc.a.val"), self._scope(), "returned from a workflow"
            )
        self.assertIn("dc.a.val", str(ctx.exception))

    def test_noop_for_bare_symbol(self):
        chain_parser.reject_unbound_access(
            _expr("dc"), self._scope(), "returned from a workflow"
        )

    def test_noop_for_chain_on_unknown_root(self):
        chain_parser.reject_unbound_access(
            _expr("numpy.pi"), self._scope(), "returned from a workflow"
        )


class TestGeneratedPort(unittest.TestCase):
    def test_takes_the_outermost_attribute(self):
        self.assertEqual(
            chain_parser.generate_port_name(_expr("dc.a.val"), []), "val_0"
        )

    def test_dodges_taken_names(self):
        self.assertEqual(
            chain_parser.generate_port_name(_expr("dc.val"), ["val_0", "val_1"]),
            "val_2",
        )


class TestInjectMixedChain(unittest.TestCase):
    def test_links_inject_root_outward_with_interleaved_constants(self):
        scope = symbol_scope.SymbolScope({"holder": _make_input("holder")})
        nodes: dict = {}
        handle = chain_parser.inject_chain(
            _expr("holder.sub['item']['subitem'].val"), scope, nodes
        )
        self.assertEqual(
            list(nodes),
            [
                "getattr_sub_0",
                "constant_0",
                "getitem_0",
                "constant_1",
                "getitem_1",
                "constant_2",
                "getattr_val_0",
                "constant_3",
            ],
        )
        self.assertEqual(handle, _make_source("getattr_val_0", "attr"))

    def test_each_link_reads_from_the_previous(self):
        scope = symbol_scope.SymbolScope({"holder": _make_input("holder")})
        nodes: dict = {}
        chain_parser.inject_chain(_expr("holder.sub['item']"), scope, nodes)
        self.assertEqual(
            scope.edges[edge_models.TargetHandle(node="getitem_0", port="a")],
            _make_source("getattr_sub_0", "attr"),
        )

    def test_item_label_numbering_matches_the_std_call_form(self):
        scope = symbol_scope.SymbolScope({"d": _make_source("identity_0", "x")})
        nodes: dict = {}
        handle = chain_parser.inject_chain(_expr("d[0]"), scope, nodes)
        self.assertEqual(list(nodes), ["getitem_0", "constant_0"])
        self.assertEqual(handle, _make_source("getitem_0", "item"))


class TestGeneratedPortForItems(unittest.TestCase):
    def test_item_port_uses_the_item_base(self):
        self.assertEqual(chain_parser.generate_port_name(_expr("d[0]"), []), "item_0")

    def test_item_port_dodges_taken_names(self):
        self.assertEqual(
            chain_parser.generate_port_name(_expr("d['k']"), ["item_0", "item_1"]),
            "item_2",
        )

    def test_outermost_link_wins_in_a_mixed_chain(self):
        self.assertEqual(
            chain_parser.generate_port_name(_expr("d['k'].val"), []), "val_0"
        )
        self.assertEqual(
            chain_parser.generate_port_name(_expr("d.sub['k']"), []), "item_0"
        )


class TestDependencySymbols(unittest.TestCase):
    def test_attribute_chain_depends_only_on_its_root(self):
        self.assertEqual(chain_parser.dependency_symbols(_expr("x.val")), {"x"})

    def test_item_chain_depends_on_root_and_key(self):
        self.assertEqual(chain_parser.dependency_symbols(_expr("d[k]")), {"d", "k"})

    def test_constant_key_adds_no_dependency(self):
        self.assertEqual(chain_parser.dependency_symbols(_expr("d['k']")), {"d"})

    def test_mixed_chain_collects_every_key(self):
        self.assertEqual(
            chain_parser.dependency_symbols(_expr("d.sub[i]['c'][j].val")),
            {"d", "i", "j"},
        )

    def test_rootless_chain_has_no_dependencies(self):
        self.assertEqual(chain_parser.dependency_symbols(_expr("f(x)[k]")), {"k"})


if __name__ == "__main__":
    unittest.main()
