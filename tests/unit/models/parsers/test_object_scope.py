import ast
import sys
import types
import unittest
from unittest.mock import patch

from flowrep.models.parsers import object_scope


def add(x: float = 2.0, y: float = 1) -> float:
    return x + y


identity = lambda x: x  # noqa: E731


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

    def test_sys_modules_fallback_when_getmodule_returns_none(self):
        """Cover line 53: sys.modules.get(module_name) is reached when inspect.getmodule
        returns None but the object's __module__ is registered in sys.modules."""
        mod = types.ModuleType("_test_fallback_mod")
        mod.__dict__["marker"] = object()
        sys.modules["_test_fallback_mod"] = mod
        try:
            func = types.FunctionType(
                (lambda: None).__code__,
                {},
                "_test_func",
            )
            func.__module__ = "_test_fallback_mod"
            # Patch inspect.getmodule to return None, simulating objects (e.g.
            # C-extension types) where the module cannot be determined from the
            # object directly, so the fallback via sys.modules is exercised.
            with patch.object(object_scope.inspect, "getmodule", return_value=None):
                scope = object_scope.get_scope(func)
            self.assertIs(scope.marker, mod.__dict__["marker"])
        finally:
            del sys.modules["_test_fallback_mod"]

    def test_builtin_type(self):
        """get_scope works for a builtin type such as ``int``."""
        scope = object_scope.get_scope(int)
        # The builtins module is always merged in, so int and len must be present.
        self.assertIs(scope.int, int)
        self.assertIs(scope.len, len)

    def test_builtin_function(self):
        """get_scope works for a builtin function such as ``len``."""
        scope = object_scope.get_scope(len)
        self.assertIs(scope.len, len)
        self.assertIs(scope.int, int)

    def test_user_defined_class(self):
        """get_scope works for a user-defined class object."""
        scope = object_scope.get_scope(Outer)
        # Module-level names from this test module should be visible.
        self.assertIs(scope.Outer, Outer)
        self.assertIs(scope.add, add)

    def test_static_method(self):
        """get_scope works for a static method."""
        scope = object_scope.get_scope(Outer.Inner.nested_func)
        self.assertIs(scope.Outer, Outer)
        self.assertIs(scope.add, add)

    def test_lambda(self):
        """get_scope works for a module-level lambda."""
        scope = object_scope.get_scope(identity)
        self.assertIs(scope.identity, identity)
        self.assertIs(scope.add, add)

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
