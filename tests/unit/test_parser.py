import ast
import dataclasses
import inspect
import textwrap
import unittest

from flowrep import model, parser


class TestAtomicDecorator(unittest.TestCase):
    def test_atomic_without_args(self):
        @parser.atomic
        def simple_func(x, y):
            return x + y

        self.assertTrue(hasattr(simple_func, "recipe"))
        self.assertIsInstance(simple_func.recipe, model.AtomicNode)
        self.assertEqual(simple_func(2, 3), 5)
        self.assertEqual(simple_func.recipe, parser.parse_atomic(simple_func))

    def test_atomic_with_unpack_mode(self):
        @parser.atomic(unpack_mode=model.UnpackMode.NONE)
        def func_no_unpack(x):
            return x

        self.assertEqual(func_no_unpack.recipe.unpack_mode, model.UnpackMode.NONE)
        self.assertEqual(
            func_no_unpack.recipe,
            parser.parse_atomic(func_no_unpack, unpack_mode=model.UnpackMode.NONE),
        )

    def test_atomic_preserves_function_behavior(self):
        @parser.atomic
        def add(a, b):
            return a, b

        result = add(1, 2)
        self.assertEqual(result, (1, 2))


class TestParseAtomic(unittest.TestCase):
    def test_basic_function(self):
        def my_func(x, y):
            return x, y

        node = parser.parse_atomic(my_func)
        self.assertEqual(node.inputs, ["x", "y"])
        self.assertEqual(node.outputs, ["x", "y"])
        self.assertTrue(node.fully_qualified_name.endswith("my_func"))

    def test_unpack_mode_none(self):
        def single_output(x):
            return x * 2, x + 1

        node = parser.parse_atomic(single_output, unpack_mode=model.UnpackMode.NONE)
        self.assertEqual(
            node.outputs,
            ["output_0"],
            msg="Return tuple should be condensed to a single output port",
        )
        self.assertEqual(node.unpack_mode, model.UnpackMode.NONE)


class TestAtomicWithOutputLabels(unittest.TestCase):
    def test_atomic_with_explicit_output_labels(self):
        @parser.atomic("a", "b")
        def func(x, y):
            return x, y

        self.assertEqual(func.recipe.outputs, ["a", "b"])
        self.assertEqual(func.recipe.inputs, ["x", "y"])

    def test_atomic_with_output_labels_and_unpack_mode(self):
        @parser.atomic("result", unpack_mode=model.UnpackMode.NONE)
        def func(x):
            return x, x + 1

        self.assertEqual(func.recipe.outputs, ["result"])
        self.assertEqual(func.recipe.unpack_mode, model.UnpackMode.NONE)

    def test_atomic_no_args_infers_labels(self):
        @parser.atomic
        def func(x):
            result = x * 2
            return result

        self.assertEqual(func.recipe.outputs, ["result"])

    def test_atomic_empty_parens_infers_labels(self):
        @parser.atomic()
        def func(x):
            a = x
            b = x + 1
            return a, b

        self.assertEqual(func.recipe.outputs, ["a", "b"])

    def test_atomic_only_unpack_mode_infers_labels(self):
        @parser.atomic(unpack_mode=model.UnpackMode.TUPLE)
        def func():
            x = 1
            y = 2
            return x, y

        self.assertEqual(func.recipe.outputs, ["x", "y"])

    def test_atomic_wrong_number_of_labels_raises(self):
        with self.assertRaises(ValueError) as ctx:

            @parser.atomic("only_one")
            def func():
                return 1, 2

        self.assertIn("Expected 2 output labels", str(ctx.exception))

    def test_atomic_with_three_labels(self):
        @parser.atomic("first", "second", "third")
        def func():
            return 1, 2, 3

        self.assertEqual(func.recipe.outputs, ["first", "second", "third"])


class TestParseAtomicWithOutputLabels(unittest.TestCase):
    def test_explicit_labels_override_inferred(self):
        def func():
            x = 1
            y = 2
            return x, y

        node = parser.parse_atomic(func, "custom_x", "custom_y")
        self.assertEqual(node.outputs, ["custom_x", "custom_y"])

    def test_explicit_labels_with_unpack_none(self):
        def func():
            return 1, 2

        node = parser.parse_atomic(func, "single", unpack_mode=model.UnpackMode.NONE)
        self.assertEqual(node.outputs, ["single"])

    def test_explicit_labels_mismatch_raises(self):
        def func():
            return 1, 2, 3

        with self.assertRaises(ValueError) as ctx:
            parser.parse_atomic(func, "only", "two")

        self.assertIn("Expected 3 output labels", str(ctx.exception))

    def test_no_explicit_labels_falls_back_to_inferred(self):
        def func():
            result = 42
            return result

        node = parser.parse_atomic(func)
        self.assertEqual(node.outputs, ["result"])

    def test_explicit_labels_with_dataclass(self):
        @dataclasses.dataclass
        class Result:
            x: int
            y: int

        def func() -> Result:
            return Result(1, 2)

        # Should use dataclass fields, not explicit labels
        node = parser.parse_atomic(func, unpack_mode=model.UnpackMode.DATACLASS)
        self.assertEqual(node.outputs, ["x", "y"])

    def test_explicit_labels_with_dataclass_wrong_count_raises(self):
        @dataclasses.dataclass
        class Result:
            x: int
            y: int

        def func() -> Result:
            return Result(1, 2)

        with self.assertRaises(ValueError) as ctx:
            parser.parse_atomic(
                func, "only_one", unpack_mode=model.UnpackMode.DATACLASS
            )

        self.assertIn("Expected 2 output labels", str(ctx.exception))


class TestAtomicEdgeCases(unittest.TestCase):
    def test_atomic_preserves_function_with_labels(self):
        @parser.atomic("out1", "out2")
        def add(a, b):
            return a + b, a - b

        self.assertEqual(add(5, 3), (8, 2))
        self.assertListEqual(add.recipe.outputs, ["out1", "out2"])

    def test_atomic_single_label_tuple_return(self):
        @parser.atomic("value")
        def func():
            x = 42
            return (x,)

        self.assertListEqual(func.recipe.outputs, ["value"])

    def test_atomic_labels_with_multiple_returns(self):
        @parser.atomic("out1", "out2")
        def func(flag):
            if flag:
                return 1, 2
            else:
                return 3, 4

        self.assertEqual(func.recipe.outputs, ["out1", "out2"])


class TestGetInputLabels(unittest.TestCase):
    def test_simple_params(self):
        def func(a, b, c):
            pass

        labels = parser._get_input_labels(func)
        self.assertEqual(labels, ["a", "b", "c"])

    def test_no_params(self):
        def func():
            pass

        labels = parser._get_input_labels(func)
        self.assertEqual(labels, [])

    def test_varargs_raises_error(self):
        def func(*args):
            pass

        with self.assertRaises(ValueError) as ctx:
            parser._get_input_labels(func)
        self.assertIn("*args", str(ctx.exception))

    def test_kwargs_raises_error(self):
        def func(**kwargs):
            pass

        with self.assertRaises(ValueError) as ctx:
            parser._get_input_labels(func)
        self.assertIn("**kwargs", str(ctx.exception))


class TestDefaultOutputLabel(unittest.TestCase):
    def test_label_format(self):
        self.assertEqual(parser.default_output_label(0), "output_0")
        self.assertEqual(parser.default_output_label(5), "output_5")


class TestGetOutputLabels(unittest.TestCase):
    def test_unpack_mode_none(self):
        def func():
            return 42

        labels = parser._get_output_labels(func, model.UnpackMode.NONE)
        self.assertEqual(labels, ["output_0"])

    def test_invalid_unpack_mode(self):
        def func():
            pass

        with self.assertRaises(TypeError):
            parser._get_output_labels(func, "invalid")


class TestParseTupleReturnLabels(unittest.TestCase):
    def test_single_return_named(self):
        def func():
            result = 42
            return result

        labels = parser._parse_tuple_return_labels(func)
        self.assertEqual(labels, ["result"])

    def test_tuple_return_named(self):
        def func():
            x = 1
            y = 2
            return x, y

        labels = parser._parse_tuple_return_labels(func)
        self.assertEqual(labels, ["x", "y"])

    def test_tuple_return_mixed(self):
        def func():
            x = 1
            return x, 2

        labels = parser._parse_tuple_return_labels(func)
        self.assertEqual(labels, ["x", "output_1"])

    def test_multiple_returns_consistent(self):
        def func(flag):
            if flag:
                a = 1
                b = 2
                return a, b
            else:
                a = 3
                b = 4
                return a, b

        labels = parser._parse_tuple_return_labels(func)
        self.assertEqual(labels, ["a", "b"])

    def test_multiple_returns_inconsistent_names(self):
        def func(flag):
            if flag:
                x = 1
                y = 2
                return x, y
            else:
                a = 3
                b = 4
                return a, b

        labels = parser._parse_tuple_return_labels(func)
        self.assertEqual(labels, ["output_0", "output_1"])

    def test_multiple_returns_different_lengths_raises_error(self):
        def func(flag):
            if flag:
                return 1, 2
            else:
                return 3

        with self.assertRaises(ValueError) as ctx:
            parser._parse_tuple_return_labels(func)
        self.assertIn("same number of elements", str(ctx.exception))

    def test_lambda_raises_error(self):
        func = lambda x: x * 2  # noqa: E731

        with self.assertRaises(ValueError) as ctx:
            parser._parse_tuple_return_labels(func)
        self.assertIn("lambda", str(ctx.exception))

    def test_dynamically_defined_function_raises_error(self):
        # exec'd functions don't have source available
        exec_globals = {}
        exec("def dynamic_func(x): return x, x + 1", exec_globals)
        func = exec_globals["dynamic_func"]

        with self.assertRaises(ValueError) as ctx:
            parser._parse_tuple_return_labels(func)
        self.assertIn("source code unavailable", str(ctx.exception))

    def test_builtin_function_raises_error(self):
        # Built-in functions have no source
        with self.assertRaises(ValueError) as ctx:
            parser._parse_tuple_return_labels(len)
        self.assertIn("source code unavailable", str(ctx.exception))


class TestExtractReturnLabels(unittest.TestCase):
    def test_no_return(self):
        def func():
            pass

        source = textwrap.dedent(inspect.getsource(func))
        tree = ast.parse(source)
        func_node = parser._get_function_definition(tree)
        labels = parser._extract_return_labels(func_node)
        self.assertEqual(labels, [()])

    def test_single_value_return(self):
        def func():
            x = 1
            return x

        source = textwrap.dedent(inspect.getsource(func))
        tree = ast.parse(source)
        func_node = parser._get_function_definition(tree)
        labels = parser._extract_return_labels(func_node)
        self.assertEqual(labels, [("x",)])

    def test_tuple_return(self):
        def func():
            a = 1
            b = 2
            return a, b

        source = textwrap.dedent(inspect.getsource(func))
        tree = ast.parse(source)
        func_node = parser._get_function_definition(tree)
        labels = parser._extract_return_labels(func_node)
        self.assertEqual(labels, [("a", "b")])


class TestParseDataclassReturnLabels(unittest.TestCase):
    def test_valid_dataclass_return(self):
        @dataclasses.dataclass
        class Result:
            x: int
            y: int

        def func() -> Result:
            return Result(1, 2)

        labels = parser._parse_dataclass_return_labels(func)
        self.assertEqual(labels, ["x", "y"])

    def test_missing_return_annotation_raises_error(self):
        def func():
            return dataclasses.make_dataclass("Result", [("x", int)])()

        with self.assertRaises(ValueError) as ctx:
            parser._parse_dataclass_return_labels(func)
        self.assertIn("return type annotation", str(ctx.exception))

    def test_non_dataclass_annotation_raises_error(self):
        def func() -> int:
            return 42

        with self.assertRaises(ValueError) as ctx:
            parser._parse_dataclass_return_labels(func)
        self.assertIn("dataclass", str(ctx.exception))

    def test_multiple_returns(self):
        @dataclasses.dataclass
        class Result:
            x: int

        def func(flag) -> Result:
            if flag:
                return Result(1)
            return Result(2)

        labels = parser._parse_dataclass_return_labels(func)
        self.assertEqual(labels, ["x"])

    def test_dangerous_returns(self):
        @dataclasses.dataclass
        class Result:
            x: int

        def func(flag) -> Result:
            if flag:
                return 42
            return Result(2)

        labels = parser._parse_dataclass_return_labels(func)
        self.assertEqual(
            labels,
            ["x"],
            msg="THIS IS THE DEVELOPER'S PROBLEM. We can't stop them from lying in "
            "their return statements.",
        )

    def test_inconsistent_returns(self):
        @dataclasses.dataclass
        class Result:
            x: int

        def func(flag) -> Result:
            if flag:
                return Result(1), Result(2)
            return Result(3)

        with self.assertRaises(ValueError) as ctx:
            parser._parse_dataclass_return_labels(func)
        self.assertIn("same number of elements", str(ctx.exception).lower())


class TestGetFunctionDefinition(unittest.TestCase):
    def test_valid_single_function(self):
        source = "def func(): pass"
        tree = ast.parse(source)
        func_def = parser._get_function_definition(tree)
        self.assertIsInstance(func_def, ast.FunctionDef)
        self.assertEqual(func_def.name, "func")

    def test_multiple_statements_raises_error(self):
        source = "def func1(): pass\ndef func2(): pass"
        tree = ast.parse(source)
        with self.assertRaises(ValueError):
            parser._get_function_definition(tree)

    def test_non_function_raises_error(self):
        source = "x = 1"
        tree = ast.parse(source)
        with self.assertRaises(ValueError):
            parser._get_function_definition(tree)


if __name__ == "__main__":
    unittest.main()
