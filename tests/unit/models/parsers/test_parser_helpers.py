import ast
import unittest

from flowrep.models import edge_models
from flowrep.models.nodes import atomic_model, helper_models
from flowrep.models.parsers import parser_helpers, symbol_scope


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


class TestConsumeCallArguments(unittest.TestCase):
    """Tests for consume_call_arguments function."""

    def _make_labeled_node(
        self, label: str, inputs: list[str], outputs: list[str] | None = None
    ) -> helper_models.LabeledNode:
        outputs = outputs or ["output_0"]
        return helper_models.LabeledNode(
            label=label,
            node=atomic_model.AtomicNode(
                fully_qualified_name="test.module.func",
                inputs=inputs,
                outputs=outputs,
            ),
        )

    def _parse_call(self, call_str: str) -> ast.Call:
        return ast.parse(call_str, mode="eval").body

    def _make_scope(self, symbols: list[str]) -> symbol_scope.SymbolScope:
        return symbol_scope.SymbolScope(
            {s: edge_models.InputSource(port=s) for s in symbols}
        )

    def _consumed_pairs(self, scope: symbol_scope.SymbolScope) -> list[tuple[str, str]]:
        """Extract (symbol, consumer_port) pairs from scope consumptions."""
        return [(c.symbol, c.consumer_port) for c in scope._consumptions]

    def test_single_positional_arg(self):
        scope = self._make_scope(["x"])
        call = self._parse_call("func(x)")
        node = self._make_labeled_node("func_0", inputs=["a"])

        parser_helpers.consume_call_arguments(scope, call, node)

        self.assertEqual(self._consumed_pairs(scope), [("x", "a")])

    def test_multiple_positional_args(self):
        scope = self._make_scope(["x", "y", "z"])
        call = self._parse_call("func(x, y, z)")
        node = self._make_labeled_node("func_0", inputs=["a", "b", "c"])

        parser_helpers.consume_call_arguments(scope, call, node)

        self.assertEqual(
            self._consumed_pairs(scope), [("x", "a"), ("y", "b"), ("z", "c")]
        )

    def test_single_keyword_arg(self):
        scope = self._make_scope(["x"])
        call = self._parse_call("func(a=x)")
        node = self._make_labeled_node("func_0", inputs=["a"])

        parser_helpers.consume_call_arguments(scope, call, node)

        self.assertEqual(self._consumed_pairs(scope), [("x", "a")])

    def test_multiple_keyword_args(self):
        scope = self._make_scope(["x", "y"])
        call = self._parse_call("func(a=x, b=y)")
        node = self._make_labeled_node("func_0", inputs=["a", "b"])

        parser_helpers.consume_call_arguments(scope, call, node)

        self.assertEqual(self._consumed_pairs(scope), [("x", "a"), ("y", "b")])

    def test_mixed_positional_and_keyword(self):
        scope = self._make_scope(["x", "y"])
        call = self._parse_call("func(x, b=y)")
        node = self._make_labeled_node("func_0", inputs=["a", "b"])

        parser_helpers.consume_call_arguments(scope, call, node)

        self.assertEqual(self._consumed_pairs(scope), [("x", "a"), ("y", "b")])

    def test_keyword_args_out_of_order(self):
        scope = self._make_scope(["x", "y"])
        call = self._parse_call("func(b=y, a=x)")
        node = self._make_labeled_node("func_0", inputs=["a", "b"])

        parser_helpers.consume_call_arguments(scope, call, node)

        # Keywords consumed in call order, not definition order
        self.assertEqual(self._consumed_pairs(scope), [("y", "b"), ("x", "a")])

    def test_no_args(self):
        scope = self._make_scope([])
        call = self._parse_call("func()")
        node = self._make_labeled_node("func_0", inputs=[])

        parser_helpers.consume_call_arguments(scope, call, node)

        self.assertEqual(self._consumed_pairs(scope), [])

    def test_literal_positional_raises_type_error(self):
        scope = self._make_scope([])
        call = self._parse_call("func(42)")
        node = self._make_labeled_node("func_0", inputs=["a"])

        with self.assertRaises(TypeError) as ctx:
            parser_helpers.consume_call_arguments(scope, call, node)

        self.assertIn("symbolic input", str(ctx.exception))
        self.assertIn("func_0", str(ctx.exception))

    def test_literal_keyword_raises_type_error(self):
        scope = self._make_scope([])
        call = self._parse_call("func(a=42)")
        node = self._make_labeled_node("func_0", inputs=["a"])

        with self.assertRaises(TypeError) as ctx:
            parser_helpers.consume_call_arguments(scope, call, node)

        self.assertIn("symbolic input", str(ctx.exception))

    def test_expression_positional_raises_type_error(self):
        scope = self._make_scope(["x", "y"])
        call = self._parse_call("func(x + y)")
        node = self._make_labeled_node("func_0", inputs=["a"])

        with self.assertRaises(TypeError) as ctx:
            parser_helpers.consume_call_arguments(scope, call, node)

        self.assertIn("symbolic input", str(ctx.exception))

    def test_nested_call_raises_type_error(self):
        scope = self._make_scope(["x"])
        call = self._parse_call("func(other_func(x))")
        node = self._make_labeled_node("func_0", inputs=["a"])

        with self.assertRaises(TypeError) as ctx:
            parser_helpers.consume_call_arguments(scope, call, node)

        self.assertIn("symbolic input", str(ctx.exception))

    def test_attribute_access_raises_type_error(self):
        scope = self._make_scope([])
        call = self._parse_call("func(obj.attr)")
        node = self._make_labeled_node("func_0", inputs=["a"])

        with self.assertRaises(TypeError) as ctx:
            parser_helpers.consume_call_arguments(scope, call, node)

        self.assertIn("symbolic input", str(ctx.exception))

    def test_subscript_raises_type_error(self):
        scope = self._make_scope([])
        call = self._parse_call("func(arr[0])")
        node = self._make_labeled_node("func_0", inputs=["a"])

        with self.assertRaises(TypeError) as ctx:
            parser_helpers.consume_call_arguments(scope, call, node)

        self.assertIn("symbolic input", str(ctx.exception))

    def test_string_literal_keyword_raises_type_error(self):
        scope = self._make_scope([])
        call = self._parse_call("func(a='hello')")
        node = self._make_labeled_node("func_0", inputs=["a"])

        with self.assertRaises(TypeError) as ctx:
            parser_helpers.consume_call_arguments(scope, call, node)

        self.assertIn("symbolic input", str(ctx.exception))

    def test_partial_consumption_before_error(self):
        """Valid args are consumed before hitting an invalid one."""
        scope = self._make_scope(["x"])
        call = self._parse_call("func(x, 42)")
        node = self._make_labeled_node("func_0", inputs=["a", "b"])

        with self.assertRaises(TypeError):
            parser_helpers.consume_call_arguments(scope, call, node)

        # First arg was consumed before the error
        self.assertEqual(self._consumed_pairs(scope), [("x", "a")])

    def test_preserves_underscore_names(self):
        scope = self._make_scope(["_private", "__dunder__"])
        call = self._parse_call("func(_private, __dunder__)")
        node = self._make_labeled_node("func_0", inputs=["a", "b"])

        parser_helpers.consume_call_arguments(scope, call, node)

        self.assertEqual(
            self._consumed_pairs(scope), [("_private", "a"), ("__dunder__", "b")]
        )

    def test_consumed_source_is_recorded(self):
        """Verify the full SymbolConsumption, not just the symbol/port pair."""
        scope = self._make_scope(["x"])
        call = self._parse_call("func(x)")
        node = self._make_labeled_node("func_0", inputs=["a"])

        parser_helpers.consume_call_arguments(scope, call, node)

        c = scope._consumptions[0]
        self.assertEqual(c.symbol, "x")
        self.assertEqual(c.consumer_node, "func_0")
        self.assertEqual(c.consumer_port, "a")
        self.assertEqual(c.source, edge_models.InputSource(port="x"))

    def test_generates_input_edges(self):
        """Consuming InputSource symbols produces correct input_edges."""
        scope = self._make_scope(["x", "y"])
        call = self._parse_call("func(x, b=y)")
        node = self._make_labeled_node("func_0", inputs=["a", "b"])

        parser_helpers.consume_call_arguments(scope, call, node)

        self.assertEqual(
            scope.input_edges,
            {
                edge_models.TargetHandle(
                    node="func_0", port="a"
                ): edge_models.InputSource(port="x"),
                edge_models.TargetHandle(
                    node="func_0", port="b"
                ): edge_models.InputSource(port="y"),
            },
        )

    def test_generates_sibling_edges(self):
        """Consuming SourceHandle symbols produces correct sibling edges."""
        scope = symbol_scope.SymbolScope(
            {"x": edge_models.SourceHandle(node="upstream_0", port="out")}
        )
        call = self._parse_call("func(x)")
        node = self._make_labeled_node("func_0", inputs=["a"])

        parser_helpers.consume_call_arguments(scope, call, node)

        self.assertEqual(scope.input_edges, {})
        self.assertEqual(
            scope.edges,
            {
                edge_models.TargetHandle(
                    node="func_0", port="a"
                ): edge_models.SourceHandle(node="upstream_0", port="out"),
            },
        )
