import ast
import dataclasses
import inspect
import textwrap
import unittest
from typing import Annotated

from flowrep import model, parser
from flowrep.parser import OutputMeta


class TestOutputMeta(unittest.TestCase):
    def test_from_annotation_with_model_instance(self):
        meta = OutputMeta(label="test")
        result = OutputMeta.from_annotation(meta)
        self.assertEqual(result, meta)

    def test_from_annotation_with_dict(self):
        result = OutputMeta.from_annotation({"label": "test"})
        self.assertIsNotNone(result)
        self.assertEqual(result.label, "test")

    def test_from_annotation_ignores_extra_keys(self):
        result = OutputMeta.from_annotation(
            {
                "label": "test",
                "units": "meters",
                "uri": "http://example.org/distance",
                "arbitrary": "garbage",
            }
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.label, "test")

    def test_from_annotation_with_no_label(self):
        result = OutputMeta.from_annotation({"units": "meters"})
        self.assertIsNotNone(result)
        self.assertIsNone(result.label)

    def test_from_annotation_with_empty_dict(self):
        result = OutputMeta.from_annotation({})
        self.assertIsNotNone(result)
        self.assertIsNone(result.label)

    def test_from_annotation_with_non_dict_non_model(self):
        self.assertIsNone(OutputMeta.from_annotation("string"))
        self.assertIsNone(OutputMeta.from_annotation(42))
        self.assertIsNone(OutputMeta.from_annotation(["list"]))

    def test_model_direct_construction(self):
        meta = OutputMeta(label="result")
        self.assertEqual(meta.label, "result")

    def test_model_default_label_is_none(self):
        meta = OutputMeta()
        self.assertIsNone(meta.label)


class TestOutputMetaFromAnnotationExceptions(unittest.TestCase):
    """Tests for exception handling in OutputMeta.from_annotation."""

    def test_dict_with_wrong_label_type_returns_none(self):
        """Pydantic validation fails if label is not str."""
        result = OutputMeta.from_annotation({"label": 123})
        self.assertIsNone(result)

    def test_dict_with_label_as_list_returns_none(self):
        result = OutputMeta.from_annotation({"label": ["not", "a", "string"]})
        self.assertIsNone(result)

    def test_dict_with_label_as_dict_returns_none(self):
        result = OutputMeta.from_annotation({"label": {"nested": "dict"}})
        self.assertIsNone(result)


class TestParseReturnLabelWithoutUnpackingExceptions(unittest.TestCase):
    """Tests for exception handling in _parse_return_label_without_unpacking."""

    def test_unresolvable_forward_reference_returns_default(self):
        """get_type_hints fails on unresolvable forward refs."""
        # Use exec to create a function with an unresolvable forward reference
        exec_globals = {}
        exec(
            "from typing import Annotated\n"
            "def func() -> 'NonExistentType':\n"
            "    return 42\n",
            exec_globals,
        )
        func = exec_globals["func"]

        labels = parser._parse_return_label_without_unpacking(func)
        self.assertEqual(labels, ["output_0"])

    def test_annotation_referencing_undefined_name_returns_default(self):
        """Annotation with undefined name in Annotated."""
        exec_globals = {"Annotated": Annotated}
        exec(
            "def func() -> Annotated['UndefinedClass', {'label': 'x'}]:\n"
            "    return 42\n",
            exec_globals,
        )
        func = exec_globals["func"]

        labels = parser._parse_return_label_without_unpacking(func)
        self.assertEqual(labels, ["output_0"])

    def test_no_return_annotation_returns_default(self):
        def func():
            return 42

        labels = parser._parse_return_label_without_unpacking(func)
        self.assertEqual(labels, ["output_0"])

    def test_annotated_without_label_returns_default(self):
        def func() -> Annotated[float, "just a string, no label"]:
            return 42.0

        labels = parser._parse_return_label_without_unpacking(func)
        self.assertEqual(labels, ["output_0"])


class TestGetAnnotatedOutputLabelsExceptions(unittest.TestCase):
    """Tests for exception handling in _get_annotated_output_labels."""

    def test_unresolvable_forward_reference_returns_none(self):
        exec_globals = {}
        exec(
            "from typing import Annotated\n"
            "def func() -> 'NonExistentType':\n"
            "    return 42\n",
            exec_globals,
        )
        func = exec_globals["func"]

        labels = parser._get_annotated_output_labels(func)
        self.assertIsNone(labels)

    def test_tuple_with_unresolvable_element_returns_none(self):
        exec_globals = {"Annotated": Annotated, "tuple": tuple}
        exec(
            "def func() -> tuple['UndefinedA', 'UndefinedB']:\n" "    return 1, 2\n",
            exec_globals,
        )
        func = exec_globals["func"]

        labels = parser._get_annotated_output_labels(func)
        self.assertIsNone(labels)

    def test_complex_invalid_annotation_returns_none(self):
        """Deeply nested invalid annotation."""
        exec_globals = {"Annotated": Annotated, "tuple": tuple}
        exec(
            "def func() -> Annotated[tuple['Missing', int], {'label': 'x'}]:\n"
            "    return 1, 2\n",
            exec_globals,
        )
        func = exec_globals["func"]

        labels = parser._get_annotated_output_labels(func)
        self.assertIsNone(labels)

    def test_annotated_with_invalid_label_type_returns_none(self):
        """Label exists but has wrong type - should not extract it."""

        def func() -> Annotated[float, {"label": 999}]:
            return 42.0

        labels = parser._get_annotated_output_labels(func)
        # from_annotation returns None for invalid label type,
        # so no valid label is found
        self.assertIsNone(labels)

    def test_tuple_with_some_invalid_labels_partial_result(self):
        """Mix of valid and invalid label types."""

        def func() -> (
            tuple[
                Annotated[int, {"label": "valid"}],
                Annotated[int, {"label": 123}],  # invalid type
            ]
        ):
            return 1, 2

        labels = parser._get_annotated_output_labels(func)
        # First is valid, second fails validation -> None
        self.assertEqual(labels, ["valid", None])


class TestEnsureFunction(unittest.TestCase):
    def test_rejects_class(self):
        class MyClass:
            pass

        with self.assertRaises(TypeError) as ctx:
            parser._ensure_function(MyClass, "@atomic")
        self.assertIn("@atomic can only decorate functions", str(ctx.exception))
        self.assertIn("type", str(ctx.exception))

    def test_rejects_callable_instance(self):
        class Callable:
            def __call__(self):
                pass

        with self.assertRaises(TypeError) as ctx:
            parser._ensure_function(Callable(), "@atomic")
        self.assertIn("@atomic can only decorate functions", str(ctx.exception))
        self.assertIn("Callable", str(ctx.exception))

    def test_accepts_lambda(self):
        # Lambdas are FunctionType, so this should pass _ensure_function
        # (they fail later in parse_atomic due to source unavailability)
        f = lambda: None  # noqa: E731
        parser._ensure_function(f, "@atomic")

    def test_rejects_builtin(self):
        with self.assertRaises(TypeError) as ctx:
            parser._ensure_function(len, "@atomic")
        self.assertIn("@atomic can only decorate functions", str(ctx.exception))
        self.assertIn("builtin_function_or_method", str(ctx.exception))

    def test_custom_decorator_name(self):
        class Foo:
            pass

        with self.assertRaises(TypeError) as ctx:
            parser._ensure_function(Foo(), "@custom")
        self.assertIn("@custom can only decorate functions", str(ctx.exception))


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


class TestAtomicTypeValidation(unittest.TestCase):
    def test_rejects_class_bare_decorator(self):
        with self.assertRaises(TypeError) as ctx:

            @parser.atomic
            class MyClass:
                pass

        self.assertIn("@atomic can only decorate functions", str(ctx.exception))

    def test_rejects_class_with_args(self):
        with self.assertRaises(TypeError) as ctx:

            @parser.atomic("output")
            class MyClass:
                pass

        self.assertIn("@atomic can only decorate functions", str(ctx.exception))

    def test_rejects_callable_instance_bare(self):
        class MyCallable:
            def __call__(self):
                pass

        with self.assertRaises(TypeError) as ctx:
            parser.atomic(MyCallable())
        self.assertIn("@atomic can only decorate functions", str(ctx.exception))

    def test_rejects_callable_instance_with_args(self):
        class Callable:
            def __call__(self):
                pass

        decorator = parser.atomic("output")
        with self.assertRaises(TypeError) as ctx:
            decorator(Callable())
        self.assertIn("@atomic can only decorate functions", str(ctx.exception))


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

    def test_no_return(self):
        def func():
            pass

        labels = parser._parse_tuple_return_labels(func)
        self.assertEqual(labels, [])

    def test_implicit_none_return(self):
        def func():
            return

        labels = parser._parse_tuple_return_labels(func)
        self.assertEqual(labels, [])

    def test_explicit_none_return(self):
        def func():
            return None

        labels = parser._parse_tuple_return_labels(func)
        self.assertEqual(labels, ["output_0"])

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
        exec_globals = {}
        exec("def dynamic_func(x): return x, x + 1", exec_globals)
        func = exec_globals["dynamic_func"]

        with self.assertRaises(ValueError) as ctx:
            parser._parse_tuple_return_labels(func)
        self.assertIn("source code unavailable", str(ctx.exception))

    def test_builtin_function_raises_error(self):
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

    def test_multiple_return_statements(self):
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

    def test_multiple_returns_raises_error(self):
        @dataclasses.dataclass
        class Result:
            x: int

        def func() -> tuple[Result, Result]:
            return Result(0), Result(1)

        with self.assertRaises(ValueError) as ctx:
            parser._parse_dataclass_return_labels(func)
        self.assertIn("exactly one value", str(ctx.exception).lower())

    def test_inconsistent_returns_raises_error(self):
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


class TestExtractLabelFromAnnotated(unittest.TestCase):
    def test_extracts_label_from_dict_metadata(self):
        hint = Annotated[float, {"label": "distance"}]
        label = parser._extract_label_from_annotated(hint)
        self.assertEqual(label, "distance")

    def test_extracts_label_from_output_meta(self):
        hint = Annotated[float, OutputMeta(label="distance")]
        label = parser._extract_label_from_annotated(hint)
        self.assertEqual(label, "distance")

    def test_returns_none_for_plain_type(self):
        label = parser._extract_label_from_annotated(float)
        self.assertIsNone(label)

    def test_returns_none_for_annotated_without_label(self):
        hint = Annotated[float, "some other metadata"]
        label = parser._extract_label_from_annotated(hint)
        self.assertIsNone(label)

    def test_finds_label_among_multiple_metadata(self):
        with self.subTest("first metadata"):
            hint = Annotated[
                float, "units: meters", {"label": "distance"}, {"other": "data"}
            ]
            label = parser._extract_label_from_annotated(hint)
            self.assertEqual(label, "distance")
        with self.subTest("appears later"):
            hint = Annotated[
                float, "units: meters", {"other": "data"}, {"label": "distance"}
            ]
            label = parser._extract_label_from_annotated(hint)
            self.assertEqual(label, "distance")

    def test_finds_output_meta_among_multiple_metadata(self):
        hint = Annotated[float, "units: meters", OutputMeta(label="distance"), 42]
        label = parser._extract_label_from_annotated(hint)
        self.assertEqual(label, "distance")

    def test_uses_first_label_if_multiple_dicts(self):
        with self.subTest("as dictionary"):
            hint = Annotated[float, {"label": "first"}, {"label": "second"}]
            label = parser._extract_label_from_annotated(hint)
            self.assertEqual(label, "first")
        with self.subTest("as model"):
            hint = Annotated[
                float, OutputMeta(label="first"), OutputMeta(label="second")
            ]
            label = parser._extract_label_from_annotated(hint)
            self.assertEqual(label, "first")

    def test_dict_with_extra_keys_works(self):
        hint = Annotated[
            float, {"label": "distance", "units": "m", "uri": "http://..."}
        ]
        label = parser._extract_label_from_annotated(hint)
        self.assertEqual(label, "distance")

    def test_output_meta_none_label_returns_none(self):
        hint = Annotated[float, OutputMeta()]
        label = parser._extract_label_from_annotated(hint)
        self.assertIsNone(label)

    def test_dict_without_label_key_returns_none(self):
        hint = Annotated[float, {"units": "meters", "uri": "http://..."}]
        label = parser._extract_label_from_annotated(hint)
        self.assertIsNone(label)


class TestGetAnnotatedOutputLabels(unittest.TestCase):
    def test_single_annotated_return_dict(self):
        def func(x) -> Annotated[float, {"label": "result"}]:
            return x * 2

        labels = parser._get_annotated_output_labels(func)
        self.assertEqual(labels, ["result"])

    def test_single_annotated_return_model(self):
        def func(x) -> Annotated[float, OutputMeta(label="result")]:
            return x * 2

        labels = parser._get_annotated_output_labels(func)
        self.assertEqual(labels, ["result"])

    def test_tuple_all_annotated_dict(self):
        def func(
            x,
        ) -> tuple[
            Annotated[float, {"label": "distance"}],
            Annotated[str, {"label": "city"}],
        ]:
            return x, "somewhere"

        labels = parser._get_annotated_output_labels(func)
        self.assertEqual(labels, ["distance", "city"])

    def test_tuple_all_annotated_model(self):
        def func(x) -> tuple[
            Annotated[float, OutputMeta(label="distance")],
            Annotated[str, OutputMeta(label="city")],
        ]:
            return x, "somewhere"

        labels = parser._get_annotated_output_labels(func)
        self.assertEqual(labels, ["distance", "city"])

    def test_tuple_mixed_dict_and_model(self):
        def func(
            x,
        ) -> tuple[
            Annotated[float, {"label": "from_dict"}],
            Annotated[str, OutputMeta(label="from_model")],
        ]:
            return x, "somewhere"

        labels = parser._get_annotated_output_labels(func)
        self.assertEqual(labels, ["from_dict", "from_model"])

    def test_tuple_partial_annotation(self):
        def func(
            x,
        ) -> tuple[
            Annotated[float, {"label": "a"}], str, Annotated[int, {"label": "c"}]
        ]:
            return x, "b", 1

        labels = parser._get_annotated_output_labels(func)
        self.assertEqual(labels, ["a", None, "c"])

    def test_no_return_annotation(self):
        def func(x):
            return x

        labels = parser._get_annotated_output_labels(func)
        self.assertIsNone(labels)

    def test_plain_type_annotation(self):
        def func(x) -> float:
            return x

        labels = parser._get_annotated_output_labels(func)
        self.assertIsNone(labels)

    def test_plain_tuple_annotation(self):
        def func(x) -> tuple[float, str]:
            return x, "y"

        labels = parser._get_annotated_output_labels(func)
        self.assertIsNone(labels)

    def test_variable_length_tuple_returns_none(self):
        def func(x) -> tuple[Annotated[int, {"label": "val"}], ...]:
            return (1, 2, 3)

        labels = parser._get_annotated_output_labels(func)
        self.assertIsNone(labels)


class TestParseTupleReturnLabelsWithAnnotations(unittest.TestCase):
    def test_annotation_overrides_scraped(self):
        def func() -> Annotated[int, {"label": "custom"}]:
            result = 42
            return result

        labels = parser._parse_tuple_return_labels(func)
        self.assertEqual(labels, ["custom"])

    def test_annotation_overrides_default(self):
        def func() -> Annotated[int, {"label": "custom"}]:
            return 42

        labels = parser._parse_tuple_return_labels(func)
        self.assertEqual(labels, ["custom"])

    def test_tuple_annotation_overrides_scraped(self):
        def func() -> (
            tuple[
                Annotated[int, {"label": "first"}],
                Annotated[int, {"label": "second"}],
            ]
        ):
            x = 1
            y = 2
            return x, y

        labels = parser._parse_tuple_return_labels(func)
        self.assertEqual(labels, ["first", "second"])

    def test_partial_annotation_merges_with_scraped(self):
        def func() -> tuple[Annotated[int, {"label": "custom_a"}], int]:
            a = 1
            b = 2
            return a, b

        labels = parser._parse_tuple_return_labels(func)
        self.assertEqual(labels, ["custom_a", "b"])

    def test_partial_annotation_merges_with_default(self):
        def func() -> tuple[int, Annotated[int, {"label": "custom_b"}]]:
            return 1, 2

        labels = parser._parse_tuple_return_labels(func)
        self.assertEqual(labels, ["output_0", "custom_b"])

    def test_annotation_length_mismatch_raises(self):
        def func() -> (
            tuple[
                Annotated[int, {"label": "a"}],
                Annotated[int, {"label": "b"}],
                Annotated[int, {"label": "c"}],
            ]
        ):
            return 1, 2

        with self.assertRaises(ValueError) as ctx:
            parser._parse_tuple_return_labels(func)
        self.assertIn("3 elements", str(ctx.exception))
        self.assertIn("2 values", str(ctx.exception))

    def test_no_annotation_falls_back_to_scraped(self):
        def func():
            result = 42
            return result

        labels = parser._parse_tuple_return_labels(func)
        self.assertEqual(labels, ["result"])


class TestAtomicWithAnnotations(unittest.TestCase):
    def test_atomic_uses_annotated_labels_dict(self):
        @parser.atomic
        def func(x) -> Annotated[float, {"label": "doubled"}]:
            return x * 2

        self.assertEqual(func.recipe.outputs, ["doubled"])

    def test_atomic_uses_annotated_labels_model(self):
        @parser.atomic
        def func(x) -> Annotated[float, OutputMeta(label="doubled")]:
            return x * 2

        self.assertEqual(func.recipe.outputs, ["doubled"])

    def test_atomic_tuple_annotated(self):
        with self.subTest("dictionary annotation"):

            @parser.atomic
            def func(
                x,
            ) -> tuple[
                Annotated[float, {"label": "sum"}],
                Annotated[float, {"label": "diff"}],
            ]:
                return x + 1, x - 1

            self.assertEqual(func.recipe.outputs, ["sum", "diff"])

        with self.subTest("model annotation"):

            @parser.atomic
            def func(x) -> tuple[
                Annotated[float, OutputMeta(label="sum")],
                Annotated[float, OutputMeta(label="diff")],
            ]:
                return x + 1, x - 1

            self.assertEqual(func.recipe.outputs, ["sum", "diff"])

    def test_explicit_labels_override_annotation(self):
        @parser.atomic("override1", "override2")
        def func(
            x,
        ) -> tuple[
            Annotated[float, {"label": "annotated1"}],
            Annotated[float, {"label": "annotated2"}],
        ]:
            return x, x

        self.assertEqual(func.recipe.outputs, ["override1", "override2"])

    def test_unpack_none_uses_single_annotation(self):
        @parser.atomic(unpack_mode=model.UnpackMode.NONE)
        def func(x) -> Annotated[float, {"label": "single_result"}]:
            return x

        self.assertEqual(func.recipe.outputs, ["single_result"])

    def test_unpack_none_uses_tuple_level_annotation(self):
        @parser.atomic(unpack_mode=model.UnpackMode.NONE)
        def func(x) -> Annotated[tuple[float, str], {"label": "combined"}]:
            return x, "y"

        self.assertEqual(func.recipe.outputs, ["combined"])

    def test_unpack_none_ignores_element_annotations(self):
        @parser.atomic(unpack_mode=model.UnpackMode.NONE)
        def func(
            x,
        ) -> tuple[
            Annotated[float, {"label": "ignored1"}],
            Annotated[str, {"label": "ignored2"}],
        ]:
            return x, "y"

        self.assertEqual(func.recipe.outputs, ["output_0"])

    def test_unpack_none_tuple_with_both_annotations(self):
        @parser.atomic(unpack_mode=model.UnpackMode.NONE)
        def func(
            x,
        ) -> Annotated[
            tuple[
                Annotated[float, {"label": "element_a"}],
                Annotated[str, {"label": "element_b"}],
            ],
            {"label": "the_pair"},
        ]:
            return x, "y"

        self.assertEqual(func.recipe.outputs, ["the_pair"])

    def test_unpack_tuple_with_both_annotations(self):
        @parser.atomic(unpack_mode=model.UnpackMode.TUPLE)
        def func(
            x,
        ) -> Annotated[
            tuple[
                Annotated[float, {"label": "element_a"}],
                Annotated[str, {"label": "element_b"}],
            ],
            {"label": "ignored_outer"},
        ]:
            return x, "y"

        self.assertEqual(func.recipe.outputs, ["element_a", "element_b"])

    def test_unpack_none_no_annotation_falls_back(self):
        @parser.atomic(unpack_mode=model.UnpackMode.NONE)
        def func(x):
            return x

        self.assertEqual(func.recipe.outputs, ["output_0"])

    def test_annotation_with_extra_keys_for_other_packages(self):
        """Simulates semantikon-style annotations with extra metadata."""

        @parser.atomic
        def func(
            x,
        ) -> Annotated[
            float,
            {
                "label": "distance",
                "units": "meters",
                "uri": "http://example.org/distance",
            },
        ]:
            return x * 2

        self.assertEqual(func.recipe.outputs, ["distance"])

    def test_tuple_annotation_with_extra_keys(self):
        @parser.atomic
        def func(
            x,
        ) -> tuple[
            Annotated[float, {"label": "dist", "units": "m"}],
            Annotated[str, {"label": "city", "uri": "http://..."}],
        ]:
            return x, "somewhere"

        self.assertEqual(func.recipe.outputs, ["dist", "city"])


class TestAnnotationWithDataclass(unittest.TestCase):
    def test_dataclass_mode_ignores_annotated_wrapper(self):
        @dataclasses.dataclass
        class Result:
            x: int
            y: int

        def func() -> Annotated[Result, {"label": "ignored"}]:
            return Result(1, 2)

        node = parser.parse_atomic(func, unpack_mode=model.UnpackMode.DATACLASS)
        self.assertEqual(node.outputs, ["x", "y"])

    def test_dataclass_mode_ignores_output_meta_wrapper(self):
        @dataclasses.dataclass
        class Result:
            x: int
            y: int

        def func() -> Annotated[Result, OutputMeta(label="ignored")]:
            return Result(1, 2)

        node = parser.parse_atomic(func, unpack_mode=model.UnpackMode.DATACLASS)
        self.assertEqual(node.outputs, ["x", "y"])


if __name__ == "__main__":
    unittest.main()
