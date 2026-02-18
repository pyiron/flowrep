import unittest

from flowrep.models import edge_models
from flowrep.models.nodes import atomic_model, helper_models
from flowrep.models.parsers.symbol_scope import SymbolScope


def _make_input(port: str) -> edge_models.InputSource:
    return edge_models.InputSource(port=port)


def _make_source(node: str, port: str) -> edge_models.SourceHandle:
    return edge_models.SourceHandle(node=node, port=port)


def _make_labeled_node(label: str, outputs: list[str]) -> helper_models.LabeledNode:
    return helper_models.LabeledNode(
        label=label,
        node=atomic_model.AtomicNode(
            fully_qualified_name="test.module.func",
            inputs=["x"],
            outputs=outputs,
            unpack_mode="tuple",
        ),
    )


class TestSymbolScopeMapping(unittest.TestCase):
    def setUp(self):
        self.scope = SymbolScope(
            {"a": _make_input("a"), "b": _make_source("node_0", "out")}
        )

    def test_getitem_hit(self):
        self.assertEqual(self.scope["a"], _make_input("a"))
        self.assertEqual(self.scope["b"], _make_source("node_0", "out"))

    def test_getitem_miss(self):
        with self.assertRaises(KeyError) as ctx:
            self.scope["missing"]
        self.assertIn("missing", str(ctx.exception))
        self.assertIn("Available", str(ctx.exception))

    def test_iter(self):
        self.assertEqual(list(self.scope), ["a", "b"])

    def test_len(self):
        self.assertEqual(len(self.scope), 2)

    def test_contains(self):
        # Mapping.__contains__ delegates to __getitem__
        self.assertIn("a", self.scope)
        self.assertNotIn("z", self.scope)


class TestSymbolScopeRegister(unittest.TestCase):
    def test_register_success(self):
        scope = SymbolScope({"a": _make_input("a")})
        child = _make_labeled_node("node_0", ["x", "y"])
        scope.register(["p", "q"], child)

        self.assertEqual(len(scope), 3)
        self.assertEqual(scope["p"], _make_source("node_0", "x"))
        self.assertEqual(scope["q"], _make_source("node_0", "y"))

    def test_register_reassignment(self):
        scope = SymbolScope({"a": _make_input("a")})
        child = _make_labeled_node("node_0", ["out"])
        scope.register(["a"], child)
        self.assertIn("a", scope.reassigned_symbols)


class TestSymbolScopeFork(unittest.TestCase):
    def test_fork_with_remap(self):
        scope = SymbolScope({"xs": _make_input("xs"), "const": _make_input("const")})
        child = scope.fork_scope({"xs": "x"})

        self.assertEqual(len(child), 2)
        self.assertEqual(child["x"], _make_input("x"))
        self.assertEqual(child["const"], _make_input("const"))
        self.assertNotIn("xs", child)

    def test_fork_without_remap(self):
        scope = SymbolScope({"a": _make_input("a")})
        child = scope.fork_scope()

        self.assertEqual(len(child), 1)
        self.assertEqual(child["a"], _make_input("a"))

    def test_fork_does_not_mutate_parent(self):
        scope = SymbolScope({"xs": _make_input("xs")})
        child = scope.fork_scope({"xs": "x"})

        # Parent unchanged
        self.assertIn("xs", scope)
        self.assertNotIn("x", scope)
        # Child has remapped symbol
        self.assertIn("x", child)
        self.assertNotIn("xs", child)

    def test_fork_all_sources_become_input_sources(self):
        """Even SourceHandle values in the parent become InputSources in the fork,
        since the child scope treats them as its own inputs."""
        scope = SymbolScope({"a": _make_source("node_0", "out")})
        child = scope.fork_scope()

        result = child["a"]
        self.assertIsInstance(result, edge_models.InputSource)
        self.assertEqual(result.port, "a")


class TestSymbolScopeErrors(unittest.TestCase):
    """Cover all ValueError/KeyError raise paths in SymbolScope."""

    # --- register ---

    def test_register_length_mismatch_raises(self):
        scope = SymbolScope({})
        child = _make_labeled_node("node_0", ["x", "y"])
        with self.assertRaises(ValueError) as ctx:
            scope.register(["only_one"], child)
        self.assertIn("Cannot map", str(ctx.exception))

    def test_register_overshadows_declared_accumulator_raises(self):
        scope = SymbolScope({})
        scope.register_accumulator("acc")
        child = _make_labeled_node("node_0", ["out"])
        with self.assertRaises(ValueError) as ctx:
            scope.register(["acc"], child)
        self.assertIn("accumulator", str(ctx.exception).lower())

    def test_register_overshadows_available_accumulator_raises(self):
        scope = SymbolScope({}, available_accumulators={"acc"})
        child = _make_labeled_node("node_0", ["out"])
        with self.assertRaises(ValueError) as ctx:
            scope.register(["acc"], child)
        self.assertIn("accumulator", str(ctx.exception).lower())

    # --- register_accumulator ---

    def test_register_accumulator_already_in_sources_raises(self):
        scope = SymbolScope({"x": _make_input("x")})
        with self.assertRaises(ValueError) as ctx:
            scope.register_accumulator("x")
        self.assertIn("already in symbol scope", str(ctx.exception))

    def test_register_accumulator_duplicate_declared_raises(self):
        scope = SymbolScope({})
        scope.register_accumulator("acc")
        with self.assertRaises(ValueError) as ctx:
            scope.register_accumulator("acc")
        self.assertIn("already declared", str(ctx.exception))

    def test_register_accumulator_already_available_raises(self):
        scope = SymbolScope({}, available_accumulators={"acc"})
        with self.assertRaises(ValueError) as ctx:
            scope.register_accumulator("acc")
        self.assertIn("already available", str(ctx.exception))

    # --- consume ---

    def test_consume_unknown_symbol_raises(self):
        scope = SymbolScope({"a": _make_input("a")})
        with self.assertRaises(KeyError) as ctx:
            scope.consume("missing", "node_0", "port")
        self.assertIn("missing", str(ctx.exception))

    # --- produce ---

    def test_produce_unknown_symbol_raises(self):
        scope = SymbolScope({})
        with self.assertRaises(KeyError) as ctx:
            scope.produce("out", "missing")
        self.assertIn("missing", str(ctx.exception))

    def test_produce_duplicate_output_port_raises(self):
        scope = SymbolScope({"a": _make_source("node_0", "x")})
        scope.produce("out", "a")
        with self.assertRaises(ValueError) as ctx:
            scope.produce("out", "a")
        self.assertIn("already produced", str(ctx.exception))

    # --- use_accumulator ---

    def test_use_accumulator_not_available_raises(self):
        scope = SymbolScope({})
        with self.assertRaises(ValueError) as ctx:
            scope.use_accumulator("missing", "x")
        self.assertIn(
            "not found among available accumulator symbols", str(ctx.exception)
        )

    def test_use_accumulator_declared_but_not_available_raises(self):
        """Declared accumulators are for child scopes, not for local use."""
        scope = SymbolScope({})
        scope.register_accumulator("acc")
        with self.assertRaises(ValueError) as ctx:
            scope.use_accumulator("acc", "x")
        self.assertIn(
            "not found among available accumulator symbols", str(ctx.exception)
        )


if __name__ == "__main__":
    unittest.main()
