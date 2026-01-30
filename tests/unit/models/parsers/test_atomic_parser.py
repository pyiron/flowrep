import ast
import dataclasses
import inspect
import textwrap
import unittest
from typing import Annotated

from flowrep.models.nodes import atomic_model
from flowrep.models.parsers import ast_helpers, atomic_parser, label_helpers


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

        labels = atomic_parser._parse_return_label_without_unpacking(func)
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

        labels = atomic_parser._parse_return_label_without_unpacking(func)
        self.assertEqual(labels, ["output_0"])

    def test_no_return_annotation_returns_default(self):
        def func():
            return 42

        labels = atomic_parser._parse_return_label_without_unpacking(func)
        self.assertEqual(labels, ["output_0"])

    def test_annotated_without_label_returns_default(self):
        def func() -> Annotated[float, "just a string, no label"]:
            return 42.0

        labels = atomic_parser._parse_return_label_without_unpacking(func)
        self.assertEqual(labels, ["output_0"])


class TestAtomicDecorator(unittest.TestCase):
    def test_atomic_without_args(self):
        @atomic_parser.atomic
        def simple_func(x, y):
            return x + y

        self.assertTrue(hasattr(simple_func, "flowrep_recipe"))
        self.assertIsInstance(simple_func.flowrep_recipe, atomic_model.AtomicNode)
        self.assertEqual(simple_func(2, 3), 5)
        self.assertEqual(
            simple_func.flowrep_recipe, atomic_parser.parse_atomic(simple_func)
        )

    def test_atomic_with_unpack_mode(self):
        @atomic_parser.atomic(unpack_mode=atomic_model.UnpackMode.NONE)
        def func_no_unpack(x):
            return x

        self.assertEqual(
            func_no_unpack.flowrep_recipe.unpack_mode, atomic_model.UnpackMode.NONE
        )
        self.assertEqual(
            func_no_unpack.flowrep_recipe,
            atomic_parser.parse_atomic(
                func_no_unpack, unpack_mode=atomic_model.UnpackMode.NONE
            ),
        )

    def test_atomic_preserves_function_behavior(self):
        @atomic_parser.atomic
        def add(a, b):
            return a, b

        result = add(1, 2)
        self.assertEqual(result, (1, 2))


class TestParseAtomic(unittest.TestCase):
    def test_basic_function(self):
        def my_func(x, y):
            return x, y

        node = atomic_parser.parse_atomic(my_func)
        self.assertEqual(node.inputs, ["x", "y"])
        self.assertEqual(node.outputs, ["x", "y"])
        self.assertTrue(node.fully_qualified_name.endswith("my_func"))

    def test_unpack_mode_none(self):
        def single_output(x):
            return x * 2, x + 1

        node = atomic_parser.parse_atomic(
            single_output, unpack_mode=atomic_model.UnpackMode.NONE
        )
        self.assertEqual(
            node.outputs,
            ["output_0"],
            msg="Return tuple should be condensed to a single output port",
        )
        self.assertEqual(node.unpack_mode, atomic_model.UnpackMode.NONE)


class TestAtomicTypeValidation(unittest.TestCase):
    def test_rejects_class_bare_decorator(self):
        with self.assertRaises(TypeError) as ctx:

            @atomic_parser.atomic
            class MyClass:
                pass

        self.assertIn("@atomic can only decorate functions", str(ctx.exception))

    def test_rejects_class_with_args(self):
        with self.assertRaises(TypeError) as ctx:

            @atomic_parser.atomic("output")
            class MyClass:
                pass

        self.assertIn("@atomic can only decorate functions", str(ctx.exception))

    def test_rejects_callable_instance_bare(self):
        class MyCallable:
            def __call__(self):
                pass

        with self.assertRaises(TypeError) as ctx:
            atomic_parser.atomic(MyCallable())
        self.assertIn("@atomic can only decorate functions", str(ctx.exception))

    def test_rejects_callable_instance_with_args(self):
        class Callable:
            def __call__(self):
                pass

        decorator = atomic_parser.atomic("output")
        with self.assertRaises(TypeError) as ctx:
            decorator(Callable())
        self.assertIn("@atomic can only decorate functions", str(ctx.exception))


class TestAtomicWithOutputLabels(unittest.TestCase):
    def test_atomic_with_explicit_output_labels(self):
        @atomic_parser.atomic("a", "b")
        def func(x, y):
            return x, y

        self.assertEqual(func.flowrep_recipe.outputs, ["a", "b"])
        self.assertEqual(func.flowrep_recipe.inputs, ["x", "y"])

    def test_atomic_with_output_labels_and_unpack_mode(self):
        @atomic_parser.atomic("result", unpack_mode=atomic_model.UnpackMode.NONE)
        def func(x):
            return x, x + 1

        self.assertEqual(func.flowrep_recipe.outputs, ["result"])
        self.assertEqual(func.flowrep_recipe.unpack_mode, atomic_model.UnpackMode.NONE)

    def test_atomic_no_args_infers_labels(self):
        @atomic_parser.atomic
        def func(x):
            result = x * 2
            return result

        self.assertEqual(func.flowrep_recipe.outputs, ["result"])

    def test_atomic_empty_parens_infers_labels(self):
        @atomic_parser.atomic()
        def func(x):
            a = x
            b = x + 1
            return a, b

        self.assertEqual(func.flowrep_recipe.outputs, ["a", "b"])

    def test_atomic_only_unpack_mode_infers_labels(self):
        @atomic_parser.atomic(unpack_mode=atomic_model.UnpackMode.TUPLE)
        def func():
            x = 1
            y = 2
            return x, y

        self.assertEqual(func.flowrep_recipe.outputs, ["x", "y"])

    def test_atomic_wrong_number_of_labels_raises(self):
        with self.assertRaises(ValueError) as ctx:

            @atomic_parser.atomic("only_one")
            def func():
                return 1, 2

        self.assertIn("Expected 2 output labels", str(ctx.exception))

    def test_atomic_with_three_labels(self):
        @atomic_parser.atomic("first", "second", "third")
        def func():
            return 1, 2, 3

        self.assertEqual(func.flowrep_recipe.outputs, ["first", "second", "third"])


class TestParseAtomicWithOutputLabels(unittest.TestCase):
    def test_explicit_labels_override_inferred(self):
        def func():
            x = 1
            y = 2
            return x, y

        node = atomic_parser.parse_atomic(func, "custom_x", "custom_y")
        self.assertEqual(node.outputs, ["custom_x", "custom_y"])

    def test_explicit_labels_with_unpack_none(self):
        def func():
            return 1, 2

        node = atomic_parser.parse_atomic(
            func, "single", unpack_mode=atomic_model.UnpackMode.NONE
        )
        self.assertEqual(node.outputs, ["single"])

    def test_explicit_labels_mismatch_raises(self):
        def func():
            return 1, 2, 3

        with self.assertRaises(ValueError) as ctx:
            atomic_parser.parse_atomic(func, "only", "two")

        self.assertIn("Expected 3 output labels", str(ctx.exception))

    def test_no_explicit_labels_falls_back_to_inferred(self):
        def func():
            result = 42
            return result

        node = atomic_parser.parse_atomic(func)
        self.assertEqual(node.outputs, ["result"])

    def test_explicit_labels_with_dataclass(self):
        @dataclasses.dataclass
        class Result:
            x: int
            y: int

        def func() -> Result:
            return Result(1, 2)

        # Should use dataclass fields, not explicit labels
        node = atomic_parser.parse_atomic(
            func, unpack_mode=atomic_model.UnpackMode.DATACLASS
        )
        self.assertEqual(node.outputs, ["x", "y"])

    def test_explicit_labels_with_dataclass_wrong_count_raises(self):
        @dataclasses.dataclass
        class Result:
            x: int
            y: int

        def func() -> Result:
            return Result(1, 2)

        with self.assertRaises(ValueError) as ctx:
            atomic_parser.parse_atomic(
                func, "only_one", unpack_mode=atomic_model.UnpackMode.DATACLASS
            )

        self.assertIn("Expected 2 output labels", str(ctx.exception))


class TestAtomicEdgeCases(unittest.TestCase):
    def test_atomic_preserves_function_with_labels(self):
        @atomic_parser.atomic("out1", "out2")
        def add(a, b):
            return a + b, a - b

        self.assertEqual(add(5, 3), (8, 2))
        self.assertListEqual(add.flowrep_recipe.outputs, ["out1", "out2"])

    def test_atomic_single_label_tuple_return(self):
        @atomic_parser.atomic("value")
        def func():
            x = 42
            return (x,)

        self.assertListEqual(func.flowrep_recipe.outputs, ["value"])

    def test_atomic_labels_with_multiple_returns(self):
        @atomic_parser.atomic("out1", "out2")
        def func(flag):
            if flag:
                return 1, 2
            else:
                return 3, 4

        self.assertEqual(func.flowrep_recipe.outputs, ["out1", "out2"])


class TestGetOutputLabels(unittest.TestCase):
    def test_unpack_mode_none(self):
        def func():
            return 42

        labels = atomic_parser._get_output_labels(func, atomic_model.UnpackMode.NONE)
        self.assertEqual(labels, ["output_0"])

    def test_invalid_unpack_mode(self):
        def func():
            pass

        with self.assertRaises(TypeError):
            atomic_parser._get_output_labels(func, "invalid")


class TestParseTupleReturnLabels(unittest.TestCase):
    def test_single_return_named(self):
        def func():
            result = 42
            return result

        labels = atomic_parser._parse_tuple_return_labels(func)
        self.assertEqual(labels, ["result"])

    def test_tuple_return_named(self):
        def func():
            x = 1
            y = 2
            return x, y

        labels = atomic_parser._parse_tuple_return_labels(func)
        self.assertEqual(labels, ["x", "y"])

    def test_tuple_return_mixed(self):
        def func():
            x = 1
            return x, 2

        labels = atomic_parser._parse_tuple_return_labels(func)
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

        labels = atomic_parser._parse_tuple_return_labels(func)
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

        labels = atomic_parser._parse_tuple_return_labels(func)
        self.assertEqual(labels, ["output_0", "output_1"])

    def test_no_return(self):
        def func():
            pass

        labels = atomic_parser._parse_tuple_return_labels(func)
        self.assertEqual(labels, [])

    def test_implicit_none_return(self):
        def func():
            return

        labels = atomic_parser._parse_tuple_return_labels(func)
        self.assertEqual(labels, [])

    def test_explicit_none_return(self):
        def func():
            return None

        labels = atomic_parser._parse_tuple_return_labels(func)
        self.assertEqual(labels, ["output_0"])

    def test_multiple_returns_different_lengths_raises_error(self):
        def func(flag):
            if flag:
                return 1, 2
            else:
                return 3

        with self.assertRaises(ValueError) as ctx:
            atomic_parser._parse_tuple_return_labels(func)
        self.assertIn("same number of elements", str(ctx.exception))

    def test_lambda_raises_error(self):
        func = lambda x: x * 2  # noqa: E731

        with self.assertRaises(ValueError) as ctx:
            atomic_parser._parse_tuple_return_labels(func)
        self.assertIn("lambda", str(ctx.exception))

    def test_dynamically_defined_function_raises_error(self):
        exec_globals = {}
        exec("def dynamic_func(x): return x, x + 1", exec_globals)
        func = exec_globals["dynamic_func"]

        with self.assertRaises(ValueError) as ctx:
            atomic_parser._parse_tuple_return_labels(func)
        self.assertIn("source code unavailable", str(ctx.exception))

    def test_builtin_function_raises_error(self):
        with self.assertRaises(ValueError) as ctx:
            atomic_parser._parse_tuple_return_labels(len)
        self.assertIn("source code unavailable", str(ctx.exception))


class TestExtractReturnLabels(unittest.TestCase):
    def test_no_return(self):
        def func():
            pass

        source = textwrap.dedent(inspect.getsource(func))
        tree = ast.parse(source)
        func_node = ast_helpers.get_function_definition(tree)
        labels = atomic_parser._extract_return_labels(func_node)
        self.assertEqual(labels, [()])

    def test_single_value_return(self):
        def func():
            x = 1
            return x

        source = textwrap.dedent(inspect.getsource(func))
        tree = ast.parse(source)
        func_node = ast_helpers.get_function_definition(tree)
        labels = atomic_parser._extract_return_labels(func_node)
        self.assertEqual(labels, [("x",)])

    def test_tuple_return(self):
        def func():
            a = 1
            b = 2
            return a, b

        source = textwrap.dedent(inspect.getsource(func))
        tree = ast.parse(source)
        func_node = ast_helpers.get_function_definition(tree)
        labels = atomic_parser._extract_return_labels(func_node)
        self.assertEqual(labels, [("a", "b")])


class TestParseDataclassReturnLabels(unittest.TestCase):
    def test_valid_dataclass_return(self):
        @dataclasses.dataclass
        class Result:
            x: int
            y: int

        def func() -> Result:
            return Result(1, 2)

        labels = atomic_parser._parse_dataclass_return_labels(func)
        self.assertEqual(labels, ["x", "y"])

    def test_missing_return_annotation_raises_error(self):
        def func():
            return dataclasses.make_dataclass("Result", [("x", int)])()

        with self.assertRaises(ValueError) as ctx:
            atomic_parser._parse_dataclass_return_labels(func)
        self.assertIn("return type annotation", str(ctx.exception))

    def test_non_dataclass_annotation_raises_error(self):
        def func() -> int:
            return 42

        with self.assertRaises(ValueError) as ctx:
            atomic_parser._parse_dataclass_return_labels(func)
        self.assertIn("dataclass", str(ctx.exception))

    def test_multiple_return_statements(self):
        @dataclasses.dataclass
        class Result:
            x: int

        def func(flag) -> Result:
            if flag:
                return Result(1)
            return Result(2)

        labels = atomic_parser._parse_dataclass_return_labels(func)
        self.assertEqual(labels, ["x"])

    def test_dangerous_returns(self):
        @dataclasses.dataclass
        class Result:
            x: int

        def func(flag) -> Result:
            if flag:
                return 42
            return Result(2)

        labels = atomic_parser._parse_dataclass_return_labels(func)
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
            atomic_parser._parse_dataclass_return_labels(func)
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
            atomic_parser._parse_dataclass_return_labels(func)
        self.assertIn("same number of elements", str(ctx.exception).lower())


class TestParseTupleReturnLabelsWithAnnotations(unittest.TestCase):
    def test_annotation_overrides_scraped(self):
        def func() -> Annotated[int, {"label": "custom"}]:
            result = 42
            return result

        labels = atomic_parser._parse_tuple_return_labels(func)
        self.assertEqual(labels, ["custom"])

    def test_annotation_overrides_default(self):
        def func() -> Annotated[int, {"label": "custom"}]:
            return 42

        labels = atomic_parser._parse_tuple_return_labels(func)
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

        labels = atomic_parser._parse_tuple_return_labels(func)
        self.assertEqual(labels, ["first", "second"])

    def test_partial_annotation_merges_with_scraped(self):
        def func() -> tuple[Annotated[int, {"label": "custom_a"}], int]:
            a = 1
            b = 2
            return a, b

        labels = atomic_parser._parse_tuple_return_labels(func)
        self.assertEqual(labels, ["custom_a", "b"])

    def test_partial_annotation_merges_with_default(self):
        def func() -> tuple[int, Annotated[int, {"label": "custom_b"}]]:
            return 1, 2

        labels = atomic_parser._parse_tuple_return_labels(func)
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
            atomic_parser._parse_tuple_return_labels(func)
        self.assertIn("3 elements", str(ctx.exception))
        self.assertIn("2 values", str(ctx.exception))

    def test_no_annotation_falls_back_to_scraped(self):
        def func():
            result = 42
            return result

        labels = atomic_parser._parse_tuple_return_labels(func)
        self.assertEqual(labels, ["result"])


class TestAtomicWithAnnotations(unittest.TestCase):
    def test_atomic_uses_annotated_labels_dict(self):
        @atomic_parser.atomic
        def func(x) -> Annotated[float, {"label": "doubled"}]:
            return x * 2

        self.assertEqual(func.flowrep_recipe.outputs, ["doubled"])

    def test_atomic_uses_annotated_labels_model(self):
        @atomic_parser.atomic
        def func(x) -> Annotated[float, label_helpers.OutputMeta(label="doubled")]:
            return x * 2

        self.assertEqual(func.flowrep_recipe.outputs, ["doubled"])

    def test_atomic_tuple_annotated(self):
        with self.subTest("dictionary annotation"):

            @atomic_parser.atomic
            def func(
                x,
            ) -> tuple[
                Annotated[float, {"label": "sum"}],
                Annotated[float, {"label": "diff"}],
            ]:
                return x + 1, x - 1

            self.assertEqual(func.flowrep_recipe.outputs, ["sum", "diff"])

        with self.subTest("model annotation"):

            @atomic_parser.atomic
            def func(x) -> tuple[
                Annotated[float, label_helpers.OutputMeta(label="sum")],
                Annotated[float, label_helpers.OutputMeta(label="diff")],
            ]:
                return x + 1, x - 1

            self.assertEqual(func.flowrep_recipe.outputs, ["sum", "diff"])

    def test_explicit_labels_override_annotation(self):
        @atomic_parser.atomic("override1", "override2")
        def func(
            x,
        ) -> tuple[
            Annotated[float, {"label": "annotated1"}],
            Annotated[float, {"label": "annotated2"}],
        ]:
            return x, x

        self.assertEqual(func.flowrep_recipe.outputs, ["override1", "override2"])

    def test_unpack_none_uses_single_annotation(self):
        @atomic_parser.atomic(unpack_mode=atomic_model.UnpackMode.NONE)
        def func(x) -> Annotated[float, {"label": "single_result"}]:
            return x

        self.assertEqual(func.flowrep_recipe.outputs, ["single_result"])

    def test_unpack_none_uses_tuple_level_annotation(self):
        @atomic_parser.atomic(unpack_mode=atomic_model.UnpackMode.NONE)
        def func(x) -> Annotated[tuple[float, str], {"label": "combined"}]:
            return x, "y"

        self.assertEqual(func.flowrep_recipe.outputs, ["combined"])

    def test_unpack_none_ignores_element_annotations(self):
        @atomic_parser.atomic(unpack_mode=atomic_model.UnpackMode.NONE)
        def func(
            x,
        ) -> tuple[
            Annotated[float, {"label": "ignored1"}],
            Annotated[str, {"label": "ignored2"}],
        ]:
            return x, "y"

        self.assertEqual(func.flowrep_recipe.outputs, ["output_0"])

    def test_unpack_none_tuple_with_both_annotations(self):
        @atomic_parser.atomic(unpack_mode=atomic_model.UnpackMode.NONE)
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

        self.assertEqual(func.flowrep_recipe.outputs, ["the_pair"])

    def test_unpack_tuple_with_both_annotations(self):
        @atomic_parser.atomic(unpack_mode=atomic_model.UnpackMode.TUPLE)
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

        self.assertEqual(func.flowrep_recipe.outputs, ["element_a", "element_b"])

    def test_unpack_none_no_annotation_falls_back(self):
        @atomic_parser.atomic(unpack_mode=atomic_model.UnpackMode.NONE)
        def func(x):
            return x

        self.assertEqual(func.flowrep_recipe.outputs, ["output_0"])

    def test_annotation_with_extra_keys_for_other_packages(self):
        """Simulates semantikon-style annotations with extra metadata."""

        @atomic_parser.atomic
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

        self.assertEqual(func.flowrep_recipe.outputs, ["distance"])

    def test_tuple_annotation_with_extra_keys(self):
        @atomic_parser.atomic
        def func(
            x,
        ) -> tuple[
            Annotated[float, {"label": "dist", "units": "m"}],
            Annotated[str, {"label": "city", "uri": "http://..."}],
        ]:
            return x, "somewhere"

        self.assertEqual(func.flowrep_recipe.outputs, ["dist", "city"])


class TestAnnotationWithDataclass(unittest.TestCase):
    def test_dataclass_mode_ignores_annotated_wrapper(self):
        @dataclasses.dataclass
        class Result:
            x: int
            y: int

        def func() -> Annotated[Result, {"label": "ignored"}]:
            return Result(1, 2)

        node = atomic_parser.parse_atomic(
            func, unpack_mode=atomic_model.UnpackMode.DATACLASS
        )
        self.assertEqual(node.outputs, ["x", "y"])

    def test_dataclass_mode_ignores_output_meta_wrapper(self):
        @dataclasses.dataclass
        class Result:
            x: int
            y: int

        def func() -> Annotated[Result, label_helpers.OutputMeta(label="ignored")]:
            return Result(1, 2)

        node = atomic_parser.parse_atomic(
            func, unpack_mode=atomic_model.UnpackMode.DATACLASS
        )
        self.assertEqual(node.outputs, ["x", "y"])


if __name__ == "__main__":
    unittest.main()
