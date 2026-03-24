"""Unit tests for flowrep.models.base_models"""

import unittest
from typing import Literal

import pydantic

from flowrep.models import base_models

from flowrep_static import makers


class _ValidTestNode(base_models.NodeModel):
    """Minimal valid NodeModel subclass for testing base class behavior."""

    type: Literal[base_models.RecipeElementType.ATOMIC] = pydantic.Field(
        default=base_models.RecipeElementType.ATOMIC, frozen=True
    )


class TestLabelValidation(unittest.TestCase):
    """Tests for Label type alias and _validate_label."""

    def test_valid_identifiers(self):
        valid = ["x", "foo", "_private", "camelCase", "snake_case", "x1", "_1"]
        for label in valid:
            with self.subTest(label=label):
                node = _ValidTestNode(inputs=[label], outputs=[])
                self.assertEqual(node.inputs, [label])

    def test_python_keywords_rejected(self):
        keywords = ["for", "while", "if", "class", "def", "return", "import", "try"]
        for kw in keywords:
            with self.subTest(keyword=kw):
                with self.assertRaises(pydantic.ValidationError) as ctx:
                    _ValidTestNode(inputs=[kw], outputs=[])
                self.assertIn("valid Python identifier", str(ctx.exception))

    def test_reserved_names_rejected(self):
        for reserved in base_models.RESERVED_NAMES:
            with self.subTest(reserved=reserved):
                with self.assertRaises(pydantic.ValidationError) as ctx:
                    _ValidTestNode(inputs=[reserved], outputs=[])
                self.assertIn("valid Python identifier", str(ctx.exception))
                self.assertIn(reserved, str(ctx.exception))

    def test_invalid_identifiers_rejected(self):
        invalid = [
            ("1starts_with_digit", "starts with digit"),
            ("has-hyphen", "contains hyphen"),
            ("has space", "contains space"),
            ("", "empty string"),
            ("has.dot", "contains dot"),
        ]
        for label, reason in invalid:
            with self.subTest(label=label, reason=reason):
                with self.assertRaises(pydantic.ValidationError) as ctx:
                    _ValidTestNode(inputs=[label] if label else [""], outputs=[])
                self.assertIn("valid Python identifier", str(ctx.exception))


class TestUniqueListValidation(unittest.TestCase):
    """Tests for UniqueList and validate_unique."""

    def test_unique_elements_accepted(self):
        node = _ValidTestNode(inputs=["a", "b", "c"], outputs=["x", "y"])
        self.assertEqual(node.inputs, ["a", "b", "c"])
        self.assertEqual(node.outputs, ["x", "y"])

    def test_duplicate_inputs_rejected(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            _ValidTestNode(inputs=["x", "y", "x"], outputs=[])
        self.assertIn("unique", str(ctx.exception).lower())
        self.assertIn("x", str(ctx.exception))

    def test_duplicate_outputs_rejected(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            _ValidTestNode(inputs=[], outputs=["a", "b", "a"])
        self.assertIn("unique", str(ctx.exception).lower())
        self.assertIn("a", str(ctx.exception))

    def test_order_preserved(self):
        node = _ValidTestNode(inputs=["c", "a", "b"], outputs=["z", "x", "y"])
        self.assertEqual(node.inputs, ["c", "a", "b"])
        self.assertEqual(node.outputs, ["z", "x", "y"])

    def test_empty_list_valid(self):
        node = _ValidTestNode(inputs=[], outputs=[])
        self.assertEqual(node.inputs, [])
        self.assertEqual(node.outputs, [])

    def test_single_element_valid(self):
        node = _ValidTestNode(inputs=["only"], outputs=["one"])
        self.assertEqual(node.inputs, ["only"])
        self.assertEqual(node.outputs, ["one"])


class TestNodeModelTypeFieldConstraints(unittest.TestCase):
    """Tests that NodeModel enforces type field requirements on subclasses."""

    def test_subclass_without_type_default_rejected(self):
        with self.assertRaises(TypeError) as ctx:

            class _NoDefault(base_models.NodeModel):
                type: Literal[base_models.RecipeElementType.ATOMIC]

        self.assertIn("_NoDefault", str(ctx.exception))
        self.assertIn("default value for 'type'", str(ctx.exception))

    def test_subclass_without_frozen_type_rejected(self):
        with self.assertRaises(TypeError) as ctx:

            class _NotFrozen(base_models.NodeModel):
                type: Literal[base_models.RecipeElementType.ATOMIC] = (
                    base_models.RecipeElementType.ATOMIC
                )

        self.assertIn("_NotFrozen", str(ctx.exception))
        self.assertIn("frozen", str(ctx.exception))

    def test_subclass_inheriting_base_type_rejected(self):
        """Subclass must redefine type, not inherit base definition."""
        with self.assertRaises(TypeError) as ctx:

            class _InheritsType(base_models.NodeModel):
                extra_field: str = "foo"

        self.assertIn("_InheritsType", str(ctx.exception))
        self.assertIn("default value for 'type'", str(ctx.exception))

    def test_valid_subclass_accepted(self):
        # Should not raise
        class _Valid(base_models.NodeModel):
            type: Literal[base_models.RecipeElementType.WORKFLOW] = pydantic.Field(
                default=base_models.RecipeElementType.WORKFLOW, frozen=True
            )

        node = _Valid(inputs=[], outputs=[])
        self.assertEqual(node.type, base_models.RecipeElementType.WORKFLOW)


class TestNodeModelTypeImmutability(unittest.TestCase):
    """Tests that type field is immutable after construction."""

    def test_type_cannot_be_overridden_at_construction(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            _ValidTestNode(
                type=base_models.RecipeElementType.WORKFLOW,
                inputs=[],
                outputs=[],
            )
        exc_str = str(ctx.exception)
        self.assertIn("Input should be", exc_str)
        self.assertIn(base_models.RecipeElementType.ATOMIC.value, exc_str)

    def test_type_cannot_be_mutated_after_construction(self):
        node = _ValidTestNode(inputs=[], outputs=[])
        with self.assertRaises(pydantic.ValidationError) as ctx:
            node.type = base_models.RecipeElementType.WORKFLOW
        self.assertIn("frozen", str(ctx.exception).lower())

    def test_type_defaults_correctly(self):
        node = _ValidTestNode(inputs=[], outputs=[])
        self.assertEqual(node.type, base_models.RecipeElementType.ATOMIC)


class TestNodeModelDescription(unittest.TestCase):
    """Tests that NodeModel description is validated."""

    def test_description_is_optional(self):
        node = _ValidTestNode(inputs=[], outputs=[])
        self.assertIsNone(node.description)

    def test_description_recoverable(self):
        description = "This is a description."
        node = _ValidTestNode(inputs=[], outputs=[], description=description)
        self.assertEqual(node.description, description)


class TestRecipeElementType(unittest.TestCase):
    """Tests for RecipeElementType enum."""

    def test_all_expected_values_exist(self):
        expected = {"atomic", "workflow", "for", "while", "if", "try"}
        actual = {e.value for e in base_models.RecipeElementType}
        self.assertEqual(expected, actual)

    def test_is_str_enum(self):
        for element in base_models.RecipeElementType:
            self.assertIsInstance(element, str)
            self.assertEqual(element, element.value)


class TestIOTypes(unittest.TestCase):
    """Tests for IOTypes enum."""

    def test_all_expected_values_exist(self):
        expected = {"inputs", "outputs"}
        actual = {e.value for e in base_models.IOTypes}
        self.assertEqual(expected, actual)

    def test_reserved_names_derived_from_io_types(self):
        self.assertEqual(
            base_models.RESERVED_NAMES,
            {e.value for e in base_models.IOTypes},
        )


class TestValidLabelFunction(unittest.TestCase):
    """Tests for _valid_label helper function."""

    def test_valid_labels_return_true(self):
        valid = ["x", "foo_bar", "_private", "CamelCase"]
        for label in valid:
            with self.subTest(label=label):
                self.assertTrue(base_models.is_valid_label(label))

    def test_keywords_return_false(self):
        self.assertFalse(base_models.is_valid_label("for"))
        self.assertFalse(base_models.is_valid_label("class"))

    def test_reserved_names_return_false(self):
        for reserved in base_models.RESERVED_NAMES:
            with self.subTest(reserved=reserved):
                self.assertFalse(base_models.is_valid_label(reserved))

    def test_non_identifiers_return_false(self):
        self.assertFalse(base_models.is_valid_label("1bad"))
        self.assertFalse(base_models.is_valid_label(""))
        self.assertFalse(base_models.is_valid_label("a-b"))


class TestValidateUniqueFunction(unittest.TestCase):
    """Tests for validate_unique helper function."""

    def test_unique_list_returned_unchanged(self):
        lst = ["a", "b", "c"]
        result = base_models.validate_unique(lst)
        self.assertEqual(result, lst)
        self.assertIs(result, lst)

    def test_duplicates_raise_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            base_models.validate_unique(["a", "b", "a"])
        self.assertIn("unique", str(ctx.exception).lower())
        self.assertIn("a", str(ctx.exception))

    def test_multiple_duplicates_all_reported(self):
        with self.assertRaises(ValueError) as ctx:
            base_models.validate_unique(["a", "b", "a", "b", "c"])
        exc_str = str(ctx.exception)
        self.assertIn("a", exc_str)
        self.assertIn("b", exc_str)

    def test_empty_list_valid(self):
        self.assertEqual(base_models.validate_unique([]), [])

    def test_custom_message(self):
        msg = "custom"
        with self.assertRaises(ValueError) as ctx:
            base_models.validate_unique(["a", "b", "a", "b", "c"], message=msg)
        exc_str = str(ctx.exception)
        self.assertEqual(msg, exc_str)


class TestNodeModelSerialization(unittest.TestCase):
    """Tests for NodeModel serialization behavior."""

    def test_roundtrip(self):
        original = _ValidTestNode(inputs=["a", "b"], outputs=["x"])
        for mode in ["json", "python"]:
            with self.subTest(mode=mode):
                data = original.model_dump(mode=mode)
            restored = _ValidTestNode.model_validate(data)
            self.assertEqual(original.inputs, restored.inputs)
            self.assertEqual(original.outputs, restored.outputs)
            self.assertEqual(original.type, restored.type)

    def test_schema_generation(self):
        # Should not raise
        schema = _ValidTestNode.model_json_schema()
        self.assertIn("properties", schema)
        self.assertIn("inputs", schema["properties"])
        self.assertIn("outputs", schema["properties"])
        self.assertIn("type", schema["properties"])


class TestNodeModelMultipleSubclasses(unittest.TestCase):
    """Tests that multiple valid subclasses can coexist."""

    def test_different_type_literals_allowed(self):
        class _TypeA(base_models.NodeModel):
            type: Literal[base_models.RecipeElementType.ATOMIC] = pydantic.Field(
                default=base_models.RecipeElementType.ATOMIC, frozen=True
            )

        class _TypeB(base_models.NodeModel):
            type: Literal[base_models.RecipeElementType.WORKFLOW] = pydantic.Field(
                default=base_models.RecipeElementType.WORKFLOW, frozen=True
            )

        a = _TypeA(inputs=[], outputs=[])
        b = _TypeB(inputs=[], outputs=[])
        self.assertEqual(a.type, base_models.RecipeElementType.ATOMIC)
        self.assertEqual(b.type, base_models.RecipeElementType.WORKFLOW)


class TestPythonReference(unittest.TestCase):
    """Tests for PythonReference model."""

    def test_inputs_with_defaults_defaults_to_empty(self):
        ref = makers.make_reference()
        self.assertEqual(ref.inputs_with_defaults, [])

    def test_inputs_with_defaults_accepted(self):
        ref = makers.make_reference(inputs_with_defaults=["a", "b"])
        self.assertEqual(ref.inputs_with_defaults, ["a", "b"])

    def test_info_required(self):
        with self.assertRaises(pydantic.ValidationError):
            base_models.PythonReference()

    def test_restricted_input_kinds_defaults_to_empty(self):
        ref = makers.make_reference()
        self.assertEqual(ref.restricted_input_kinds, {})

    def test_restricted_input_kinds_accepted(self):
        ref = makers.make_reference(
            restricted_input_kinds={"a": base_models.RestrictedParamKind.KEYWORD_ONLY}
        )
        self.assertEqual(
            ref.restricted_input_kinds,
            {"a": base_models.RestrictedParamKind.KEYWORD_ONLY},
        )

    def test_restricted_input_kinds_positional_only(self):
        ref = makers.make_reference(
            restricted_input_kinds={
                "x": base_models.RestrictedParamKind.POSITIONAL_ONLY
            }
        )
        self.assertEqual(
            ref.restricted_input_kinds,
            {"x": base_models.RestrictedParamKind.POSITIONAL_ONLY},
        )

    def test_var_positional_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            makers.make_reference(
                restricted_input_kinds={
                    "args": base_models.RestrictedParamKind.VAR_POSITIONAL
                }
            )
        self.assertIn("Variadic", str(ctx.exception))
        self.assertIn("args", str(ctx.exception))

    def test_var_keyword_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            makers.make_reference(
                restricted_input_kinds={
                    "kwargs": base_models.RestrictedParamKind.VAR_KEYWORD
                }
            )
        self.assertIn("Variadic", str(ctx.exception))
        self.assertIn("kwargs", str(ctx.exception))

    def test_roundtrip(self):
        original = makers.make_reference(
            inputs_with_defaults=["x", "y"],
            restricted_input_kinds={
                "a": base_models.RestrictedParamKind.KEYWORD_ONLY,
                "b": base_models.RestrictedParamKind.POSITIONAL_ONLY,
            },
        )
        for mode in ["json", "python"]:
            with self.subTest(mode=mode):
                data = original.model_dump(mode=mode)
                restored = base_models.PythonReference.model_validate(data)
                self.assertEqual(
                    original.inputs_with_defaults, restored.inputs_with_defaults
                )
                self.assertEqual(
                    original.inputs_with_defaults, restored.inputs_with_defaults
                )
                self.assertEqual(
                    original.info.fully_qualified_name,
                    restored.info.fully_qualified_name,
                )
                self.assertEqual(
                    original.restricted_input_kinds,
                    restored.restricted_input_kinds,
                )


if __name__ == "__main__":
    unittest.main()
