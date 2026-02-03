import ast
import unittest
from typing import Annotated

from flowrep.models.parsers import label_helpers


class TestOutputMeta(unittest.TestCase):
    def test_from_annotation_with_model_instance(self):
        meta = label_helpers.OutputMeta(label="test")
        result = label_helpers.OutputMeta.from_annotation(meta)
        self.assertEqual(result, meta)

    def test_from_annotation_with_dict(self):
        result = label_helpers.OutputMeta.from_annotation({"label": "test"})
        self.assertIsNotNone(result)
        self.assertEqual(result.label, "test")

    def test_from_annotation_ignores_extra_keys(self):
        result = label_helpers.OutputMeta.from_annotation(
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
        result = label_helpers.OutputMeta.from_annotation({"units": "meters"})
        self.assertIsNotNone(result)
        self.assertIsNone(result.label)

    def test_from_annotation_with_empty_dict(self):
        result = label_helpers.OutputMeta.from_annotation({})
        self.assertIsNotNone(result)
        self.assertIsNone(result.label)

    def test_from_annotation_with_non_dict_non_model(self):
        self.assertIsNone(label_helpers.OutputMeta.from_annotation("string"))
        self.assertIsNone(label_helpers.OutputMeta.from_annotation(42))
        self.assertIsNone(label_helpers.OutputMeta.from_annotation(["list"]))

    def test_model_direct_construction(self):
        meta = label_helpers.OutputMeta(label="result")
        self.assertEqual(meta.label, "result")

    def test_model_default_label_is_none(self):
        meta = label_helpers.OutputMeta()
        self.assertIsNone(meta.label)


class TestOutputMetaFromAnnotationExceptions(unittest.TestCase):
    """Tests for exception handling in atomic_parser.OutputMeta.from_annotation."""

    def test_dict_with_wrong_label_type_returns_none(self):
        """Pydantic validation fails if label is not str."""
        result = label_helpers.OutputMeta.from_annotation({"label": 123})
        self.assertIsNone(result)

    def test_dict_with_label_as_list_returns_none(self):
        result = label_helpers.OutputMeta.from_annotation(
            {"label": ["not", "a", "string"]}
        )
        self.assertIsNone(result)

    def test_dict_with_label_as_dict_returns_none(self):
        result = label_helpers.OutputMeta.from_annotation({"label": {"nested": "dict"}})
        self.assertIsNone(result)


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

        labels = label_helpers.get_annotated_output_labels(func)
        self.assertIsNone(labels)

    def test_tuple_with_unresolvable_element_returns_none(self):
        exec_globals = {"Annotated": Annotated, "tuple": tuple}
        exec(
            "def func() -> tuple['UndefinedA', 'UndefinedB']:\n" "    return 1, 2\n",
            exec_globals,
        )
        func = exec_globals["func"]

        labels = label_helpers.get_annotated_output_labels(func)
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

        labels = label_helpers.get_annotated_output_labels(func)
        self.assertIsNone(labels)

    def test_annotated_with_invalid_label_type_returns_none(self):
        """Label exists but has wrong type - should not extract it."""

        def func() -> Annotated[float, {"label": 999}]:
            return 42.0

        labels = label_helpers.get_annotated_output_labels(func)
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

        labels = label_helpers.get_annotated_output_labels(func)
        # First is valid, second fails validation -> None
        self.assertEqual(labels, ["valid", None])


class TestGetInputLabels(unittest.TestCase):
    def test_simple_params(self):
        def func(a, b, c):
            pass

        labels = label_helpers.get_input_labels(func)
        self.assertEqual(labels, ["a", "b", "c"])

    def test_no_params(self):
        def func():
            pass

        labels = label_helpers.get_input_labels(func)
        self.assertEqual(labels, [])

    def test_varargs_raises_error(self):
        def func(*args):
            pass

        with self.assertRaises(ValueError) as ctx:
            label_helpers.get_input_labels(func)
        self.assertIn("*args", str(ctx.exception))

    def test_kwargs_raises_error(self):
        def func(**kwargs):
            pass

        with self.assertRaises(ValueError) as ctx:
            label_helpers.get_input_labels(func)
        self.assertIn("**kwargs", str(ctx.exception))


class TestExtractLabelFromAnnotated(unittest.TestCase):
    def test_extracts_label_from_dict_metadata(self):
        hint = Annotated[float, {"label": "distance"}]
        label = label_helpers.extract_label_from_annotated(hint)
        self.assertEqual(label, "distance")

    def test_extracts_label_from_output_meta(self):
        hint = Annotated[float, label_helpers.OutputMeta(label="distance")]
        label = label_helpers.extract_label_from_annotated(hint)
        self.assertEqual(label, "distance")

    def test_returns_none_for_plain_type(self):
        label = label_helpers.extract_label_from_annotated(float)
        self.assertIsNone(label)

    def test_returns_none_for_annotated_without_label(self):
        hint = Annotated[float, "some other metadata"]
        label = label_helpers.extract_label_from_annotated(hint)
        self.assertIsNone(label)

    def test_finds_label_among_multiple_metadata(self):
        with self.subTest("first metadata"):
            hint = Annotated[
                float, "units: meters", {"label": "distance"}, {"other": "data"}
            ]
            label = label_helpers.extract_label_from_annotated(hint)
            self.assertEqual(label, "distance")
        with self.subTest("appears later"):
            hint = Annotated[
                float, "units: meters", {"other": "data"}, {"label": "distance"}
            ]
            label = label_helpers.extract_label_from_annotated(hint)
            self.assertEqual(label, "distance")

    def test_finds_output_meta_among_multiple_metadata(self):
        hint = Annotated[
            float, "units: meters", label_helpers.OutputMeta(label="distance"), 42
        ]
        label = label_helpers.extract_label_from_annotated(hint)
        self.assertEqual(label, "distance")

    def test_uses_first_label_if_multiple_dicts(self):
        with self.subTest("as dictionary"):
            hint = Annotated[float, {"label": "first"}, {"label": "second"}]
            label = label_helpers.extract_label_from_annotated(hint)
            self.assertEqual(label, "first")
        with self.subTest("as model"):
            hint = Annotated[
                float,
                label_helpers.OutputMeta(label="first"),
                label_helpers.OutputMeta(label="second"),
            ]
            label = label_helpers.extract_label_from_annotated(hint)
            self.assertEqual(label, "first")

    def test_dict_with_extra_keys_works(self):
        hint = Annotated[
            float, {"label": "distance", "units": "m", "uri": "http://..."}
        ]
        label = label_helpers.extract_label_from_annotated(hint)
        self.assertEqual(label, "distance")

    def test_output_meta_none_label_returns_none(self):
        hint = Annotated[float, label_helpers.OutputMeta()]
        label = label_helpers.extract_label_from_annotated(hint)
        self.assertIsNone(label)

    def test_dict_without_label_key_returns_none(self):
        hint = Annotated[float, {"units": "meters", "uri": "http://..."}]
        label = label_helpers.extract_label_from_annotated(hint)
        self.assertIsNone(label)


class TestGetAnnotatedOutputLabels(unittest.TestCase):
    def test_single_annotated_return_dict(self):
        def func(x) -> Annotated[float, {"label": "result"}]:
            return x * 2

        labels = label_helpers.get_annotated_output_labels(func)
        self.assertEqual(labels, ["result"])

    def test_single_annotated_return_model(self):
        def func(x) -> Annotated[float, label_helpers.OutputMeta(label="result")]:
            return x * 2

        labels = label_helpers.get_annotated_output_labels(func)
        self.assertEqual(labels, ["result"])

    def test_tuple_all_annotated_dict(self):
        def func(
            x,
        ) -> tuple[
            Annotated[float, {"label": "distance"}],
            Annotated[str, {"label": "city"}],
        ]:
            return x, "somewhere"

        labels = label_helpers.get_annotated_output_labels(func)
        self.assertEqual(labels, ["distance", "city"])

    def test_tuple_all_annotated_model(self):
        def func(x) -> tuple[
            Annotated[float, label_helpers.OutputMeta(label="distance")],
            Annotated[str, label_helpers.OutputMeta(label="city")],
        ]:
            return x, "somewhere"

        labels = label_helpers.get_annotated_output_labels(func)
        self.assertEqual(labels, ["distance", "city"])

    def test_tuple_mixed_dict_and_model(self):
        def func(
            x,
        ) -> tuple[
            Annotated[float, {"label": "from_dict"}],
            Annotated[str, label_helpers.OutputMeta(label="from_model")],
        ]:
            return x, "somewhere"

        labels = label_helpers.get_annotated_output_labels(func)
        self.assertEqual(labels, ["from_dict", "from_model"])

    def test_tuple_partial_annotation(self):
        def func(
            x,
        ) -> tuple[
            Annotated[float, {"label": "a"}], str, Annotated[int, {"label": "c"}]
        ]:
            return x, "b", 1

        labels = label_helpers.get_annotated_output_labels(func)
        self.assertEqual(labels, ["a", None, "c"])

    def test_no_return_annotation(self):
        def func(x):
            return x

        labels = label_helpers.get_annotated_output_labels(func)
        self.assertIsNone(labels)

    def test_plain_type_annotation(self):
        def func(x) -> float:
            return x

        labels = label_helpers.get_annotated_output_labels(func)
        self.assertIsNone(labels)

    def test_plain_tuple_annotation(self):
        def func(x) -> tuple[float, str]:
            return x, "y"

        labels = label_helpers.get_annotated_output_labels(func)
        self.assertIsNone(labels)

    def test_variable_length_tuple_returns_none(self):
        def func(x) -> tuple[Annotated[int, {"label": "val"}], ...]:
            return (1, 2, 3)

        labels = label_helpers.get_annotated_output_labels(func)
        self.assertIsNone(labels)


class TestMergeLabels(unittest.TestCase):
    def test_none_first_choice_returns_fallback(self):
        result = label_helpers.merge_labels(None, ["a", "b", "c"])
        self.assertEqual(["a", "b", "c"], result)

    def test_full_first_choice_ignores_fallback(self):
        result = label_helpers.merge_labels(["x", "y"], ["a", "b"])
        self.assertEqual(["x", "y"], result)

    def test_partial_first_choice_merges(self):
        result = label_helpers.merge_labels(["x", None, "z"], ["a", "b", "c"])
        self.assertEqual(["x", "b", "z"], result)

    def test_all_none_first_choice_returns_fallback(self):
        result = label_helpers.merge_labels([None, None], ["a", "b"])
        self.assertEqual(["a", "b"], result)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError) as ctx:
            label_helpers.merge_labels(["x", "y"], ["a", "b", "c"])
        self.assertIn("number of elements differ", str(ctx.exception))

    def test_message_prefix_included_in_error(self):
        with self.assertRaises(ValueError) as ctx:
            label_helpers.merge_labels(["x"], ["a", "b"], message_prefix="Custom: ")
        self.assertIn("Custom: ", str(ctx.exception))

    def test_empty_collections(self):
        result = label_helpers.merge_labels([], [])
        self.assertEqual([], result)


class TestExtractReturnLabels(unittest.TestCase):
    """Tests for extract_return_labels which processes a single ast.Return node."""

    def _parse_return(self, code: str) -> ast.Return:
        """Helper to parse a return statement from code."""
        tree = ast.parse(code)
        func = tree.body[0]
        for node in ast.walk(func):
            if isinstance(node, ast.Return):
                return node
        raise ValueError("No return statement found")

    def test_return_none_implicit(self):
        ret = self._parse_return("def f(): return")
        self.assertEqual(label_helpers.extract_return_labels(ret), ())

    def test_return_none_explicit(self):
        ret = self._parse_return("def f(): return None")
        self.assertEqual(label_helpers.extract_return_labels(ret), ("output_0",))

    def test_return_single_name(self):
        ret = self._parse_return("def f():\n  x = 1\n  return x")
        self.assertEqual(label_helpers.extract_return_labels(ret), ("x",))

    def test_return_single_literal(self):
        ret = self._parse_return("def f(): return 42")
        self.assertEqual(label_helpers.extract_return_labels(ret), ("output_0",))

    def test_return_tuple_all_names(self):
        ret = self._parse_return("def f():\n  a, b = 1, 2\n  return a, b")
        self.assertEqual(label_helpers.extract_return_labels(ret), ("a", "b"))

    def test_return_tuple_mixed(self):
        ret = self._parse_return("def f():\n  x = 1\n  return x, 2, 'literal'")
        self.assertEqual(
            label_helpers.extract_return_labels(ret), ("x", "output_1", "output_2")
        )

    def test_return_tuple_all_literals(self):
        ret = self._parse_return("def f(): return 1, 2, 3")
        self.assertEqual(
            label_helpers.extract_return_labels(ret),
            ("output_0", "output_1", "output_2"),
        )

    def test_return_expression(self):
        ret = self._parse_return("def f(): return x + y")
        self.assertEqual(label_helpers.extract_return_labels(ret), ("output_0",))

    def test_return_call(self):
        ret = self._parse_return("def f(): return foo()")
        self.assertEqual(label_helpers.extract_return_labels(ret), ("output_0",))

    def test_return_tuple_with_call(self):
        ret = self._parse_return("def f():\n  a = 1\n  return a, foo()")
        self.assertEqual(label_helpers.extract_return_labels(ret), ("a", "output_1"))


class TestDefaultOutputLabel(unittest.TestCase):
    def test_label_format(self):
        self.assertEqual(label_helpers.default_output_label(0), "output_0")
        self.assertEqual(label_helpers.default_output_label(5), "output_5")


class TestUniqueSuffix(unittest.TestCase):
    def test_first_suffix(self):
        result = label_helpers.unique_suffix("foo", [])
        self.assertEqual(result, "foo_0")

    def test_increments_on_collision(self):
        result = label_helpers.unique_suffix("foo", ["foo_0", "foo_1"])
        self.assertEqual(result, "foo_2")

    def test_handles_gaps(self):
        result = label_helpers.unique_suffix("foo", ["foo_0", "foo_2"])
        self.assertEqual(result, "foo_1")


if __name__ == "__main__":
    unittest.main()
