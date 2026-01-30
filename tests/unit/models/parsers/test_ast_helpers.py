import ast
import unittest

from flowrep.models.parsers import ast_helpers


class TestEnsureFunction(unittest.TestCase):
    def test_rejects_class(self):
        class MyClass:
            pass

        with self.assertRaises(TypeError) as ctx:
            ast_helpers.ensure_function(MyClass, "@atomic")
        self.assertIn("@atomic can only decorate functions", str(ctx.exception))
        self.assertIn("type", str(ctx.exception))

    def test_rejects_callable_instance(self):
        class Callable:
            def __call__(self):
                pass

        with self.assertRaises(TypeError) as ctx:
            ast_helpers.ensure_function(Callable(), "@atomic")
        self.assertIn("@atomic can only decorate functions", str(ctx.exception))
        self.assertIn("Callable", str(ctx.exception))

    def test_accepts_lambda(self):
        # Lambdas are FunctionType, so this should pass _ensure_function
        # (they fail later in parse_atomic due to source unavailability)
        f = lambda: None  # noqa: E731
        ast_helpers.ensure_function(f, "@atomic")

    def test_rejects_builtin(self):
        with self.assertRaises(TypeError) as ctx:
            ast_helpers.ensure_function(len, "@atomic")
        self.assertIn("@atomic can only decorate functions", str(ctx.exception))
        self.assertIn("builtin_function_or_method", str(ctx.exception))

    def test_custom_decorator_name(self):
        class Foo:
            pass

        with self.assertRaises(TypeError) as ctx:
            ast_helpers.ensure_function(Foo(), "@custom")
        self.assertIn("@custom can only decorate functions", str(ctx.exception))


class TestGetFunctionDefinition(unittest.TestCase):
    def test_valid_single_function(self):
        source = "def func(): pass"
        tree = ast.parse(source)
        func_def = ast_helpers.get_function_definition(tree)
        self.assertIsInstance(func_def, ast.FunctionDef)
        self.assertEqual(func_def.name, "func")

    def test_multiple_statements_raises_error(self):
        source = "def func1(): pass\ndef func2(): pass"
        tree = ast.parse(source)
        with self.assertRaises(ValueError):
            ast_helpers.get_function_definition(tree)

    def test_non_function_raises_error(self):
        source = "x = 1"
        tree = ast.parse(source)
        with self.assertRaises(ValueError):
            ast_helpers.get_function_definition(tree)


if __name__ == "__main__":
    unittest.main()
