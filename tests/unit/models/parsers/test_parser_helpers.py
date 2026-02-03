import ast
import unittest

from flowrep.models.parsers import parser_helpers


class TestEnsureFunction(unittest.TestCase):
    def test_rejects_class(self):
        class MyClass:
            pass

        with self.assertRaises(TypeError) as ctx:
            parser_helpers.ensure_function(MyClass, "@atomic")
        self.assertIn("@atomic can only decorate functions", str(ctx.exception))
        self.assertIn("type", str(ctx.exception))

    def test_rejects_callable_instance(self):
        class Callable:
            def __call__(self):
                pass

        with self.assertRaises(TypeError) as ctx:
            parser_helpers.ensure_function(Callable(), "@atomic")
        self.assertIn("@atomic can only decorate functions", str(ctx.exception))
        self.assertIn("Callable", str(ctx.exception))

    def test_accepts_lambda(self):
        # Lambdas are FunctionType, so this should pass _ensure_function
        # (they fail later in parse_atomic due to source unavailability)
        f = lambda: None  # noqa: E731
        parser_helpers.ensure_function(f, "@atomic")

    def test_rejects_builtin(self):
        with self.assertRaises(TypeError) as ctx:
            parser_helpers.ensure_function(len, "@atomic")
        self.assertIn("@atomic can only decorate functions", str(ctx.exception))
        self.assertIn("builtin_function_or_method", str(ctx.exception))

    def test_custom_decorator_name(self):
        class Foo:
            pass

        with self.assertRaises(TypeError) as ctx:
            parser_helpers.ensure_function(Foo(), "@custom")
        self.assertIn("@custom can only decorate functions", str(ctx.exception))


class TestGetFunctionDefinition(unittest.TestCase):
    def test_valid_single_function(self):
        source = "def func(): pass"
        tree = ast.parse(source)
        func_def = parser_helpers.get_function_definition(tree)
        self.assertIsInstance(func_def, ast.FunctionDef)
        self.assertEqual(func_def.name, "func")

    def test_multiple_statements_raises_error(self):
        source = "def func1(): pass\ndef func2(): pass"
        tree = ast.parse(source)
        with self.assertRaises(ValueError):
            parser_helpers.get_function_definition(tree)

    def test_non_function_raises_error(self):
        source = "x = 1"
        tree = ast.parse(source)
        with self.assertRaises(ValueError):
            parser_helpers.get_function_definition(tree)


class TestGetSourceCode(unittest.TestCase):
    def test_returns_source_for_normal_function(self):
        def my_func():
            return 42

        source = parser_helpers.get_source_code(my_func)
        self.assertIn("def my_func", source)
        self.assertIn("return 42", source)

    def test_lambda_raises_error(self):
        func = lambda x: x * 2  # noqa: E731

        with self.assertRaises(ValueError) as ctx:
            parser_helpers.get_source_code(func)
        self.assertIn("lambda", str(ctx.exception))

    def test_dynamically_defined_function_raises_error(self):
        exec_globals = {}
        exec("def dynamic_func(x): return x", exec_globals)
        func = exec_globals["dynamic_func"]

        with self.assertRaises(ValueError) as ctx:
            parser_helpers.get_source_code(func)
        self.assertIn("source code unavailable", str(ctx.exception))

    def test_builtin_function_raises_error(self):
        with self.assertRaises(ValueError) as ctx:
            parser_helpers.get_source_code(len)
        self.assertIn("source code unavailable", str(ctx.exception))

    def test_dedents_source(self):
        # Nested function to ensure indentation exists
        def outer():
            def inner():
                return 1

            return inner

        func = outer()
        source = parser_helpers.get_source_code(func)
        # Should start at column 0, not be indented
        self.assertTrue(source.startswith("def inner"))


class TestGetAstFunctionNode(unittest.TestCase):
    def test_returns_function_def_node(self):
        def my_func(x, y):
            return x + y

        node = parser_helpers.get_ast_function_node(my_func)
        self.assertIsInstance(node, ast.FunctionDef)
        self.assertEqual(node.name, "my_func")
        self.assertEqual(len(node.args.args), 2)

    def test_lambda_raises_error(self):
        func = lambda: None  # noqa: E731

        with self.assertRaises(ValueError) as ctx:
            parser_helpers.get_ast_function_node(func)
        self.assertIn("lambda", str(ctx.exception))

    def test_dynamically_defined_function_raises_error(self):
        exec_globals = {}
        exec("def f(): pass", exec_globals)

        with self.assertRaises(ValueError) as ctx:
            parser_helpers.get_ast_function_node(exec_globals["f"])
        self.assertIn("source code unavailable", str(ctx.exception))


class TestResolveSymbolsToStrings(unittest.TestCase):
    def test_single_name(self):
        node = ast.Name(id="foo")
        result = parser_helpers.resolve_symbols_to_strings(node)
        self.assertEqual(result, ["foo"])

    def test_tuple_of_names(self):
        node = ast.Tuple(elts=[ast.Name(id="a"), ast.Name(id="b")])
        result = parser_helpers.resolve_symbols_to_strings(node)
        self.assertEqual(result, ["a", "b"])

    def test_non_name_raises(self):
        node = ast.Constant(value=42)
        with self.assertRaises(TypeError):
            parser_helpers.resolve_symbols_to_strings(node)


if __name__ == "__main__":
    unittest.main()
