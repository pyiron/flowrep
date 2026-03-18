import ast
import sys
import types
import unittest

from flowrep.models.parsers import object_scope


def add(x: float = 2.0, y: float = 1) -> float:
    return x + y


class Outer:
    class Inner:
        @staticmethod
        def nested_func(a, b):
            return a, b


class TestScopeProxy(unittest.TestCase):
    def test_basic_access(self):
        d = {"foo": 1, "bar": 2}
        proxy = object_scope.ScopeProxy(d)
        self.assertEqual(proxy.foo, 1)
        self.assertEqual(proxy.bar, 2)

    def test_missing_key_raises_attribute_error(self):
        proxy = object_scope.ScopeProxy({})
        with self.assertRaises(AttributeError):
            _ = proxy.nonexistent


class TestGetScope(unittest.TestCase):
    def test_returns_module_globals(self):
        scope = object_scope.get_scope(add)
        # Should have module-level names
        self.assertIs(scope.add, add)
        self.assertIs(scope.Outer, Outer)
        self.assertIs(scope.object_scope, object_scope)

    def test_includes_builtins(self):
        scope = object_scope.get_scope(add)
        self.assertIs(scope.len, len)
        self.assertIs(scope.int, int)
        self.assertIs(scope.ValueError, ValueError)

    def test_none_module_fallback_via_dunder_module(self):
        """When inspect.getmodule returns None but __module__ is set, fall back."""
        mod = types.ModuleType("_test_dynamic_mod")
        mod.__dict__["sentinel"] = object()
        sys.modules["_test_dynamic_mod"] = mod
        try:
            func = types.FunctionType(
                (lambda: None).__code__,
                {},
                "_test_func",
            )
            # Manually set __module__ but keep the function out of a real module
            # so inspect.getmodule() returns None.
            func.__module__ = "_test_dynamic_mod"
            scope = object_scope.get_scope(func)
            self.assertIs(scope.sentinel, mod.__dict__["sentinel"])
        finally:
            del sys.modules["_test_dynamic_mod"]

    def test_no_resolvable_module_raises_value_error(self):
        """When neither inspect.getmodule nor __module__ resolves, raise ValueError."""
        func = types.FunctionType(
            (lambda: None).__code__,
            {},
            "_orphan_func",
        )
        func.__module__ = None  # type: ignore[assignment]
        with self.assertRaises(ValueError):
            object_scope.get_scope(func)


class TestResolveSymbolToObject(unittest.TestCase):
    def test_simple_name(self):
        scope = object_scope.ScopeProxy({"add": add})

        node = ast.Name(id="add")
        result = object_scope.resolve_symbol_to_object(node, scope)
        self.assertIs(result, add)

    def test_attribute_chain(self):
        scope = object_scope.ScopeProxy({"Outer": Outer})

        # Outer.Inner.nested_func
        node = ast.Attribute(
            value=ast.Attribute(value=ast.Name(id="Outer"), attr="Inner"),
            attr="nested_func",
        )
        result = object_scope.resolve_symbol_to_object(node, scope)
        self.assertIs(result, Outer.Inner.nested_func)

    def test_missing_attribute_raises(self):
        scope = object_scope.ScopeProxy({"Outer": Outer})

        node = ast.Attribute(value=ast.Name(id="Outer"), attr="NonExistent")
        with self.assertRaises(ValueError):
            object_scope.resolve_symbol_to_object(node, scope)

    def test_unrecognized_node_raises(self):
        scope = object_scope.ScopeProxy({})
        node = ast.Constant(value=42)
        with self.assertRaises(TypeError):
            object_scope.resolve_symbol_to_object(node, scope)

    def test_resolve_attribute_to_object(self):
        scope = object_scope.ScopeProxy({"ast": ast})
        f = object_scope.resolve_attribute_to_object("ast.literal_eval", scope)
        self.assertIs(f, ast.literal_eval)


if __name__ == "__main__":
    unittest.main()
