"""Unit tests for flowrep.model"""

import unittest
from typing import Literal

import pydantic

from flowrep.models import base_models
from flowrep.models.nodes import atomic_model, workflow_model


class TestNodeModel(unittest.TestCase):
    """Tests for input/output uniqueness validation on NodeModel base class."""

    def test_duplicate_inputs_rejected(self):
        """Any NodeModel subclass should reject duplicate inputs."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            atomic_model.AtomicNode(
                fully_qualified_name="mod.func",
                inputs=["x", "y", "x"],  # duplicate 'x'
                outputs=["z"],
            )
        self.assertIn("unique", str(ctx.exception).lower())
        self.assertIn("x", str(ctx.exception))

    def test_duplicate_outputs_rejected(self):
        """Any NodeModel subclass should reject duplicate outputs."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            workflow_model.WorkflowNode(
                inputs=["a"],
                outputs=["x", "y", "x"],  # duplicate 'x'
                nodes={},
                input_edges={},
                edges={},
                output_edges={},
            )
        self.assertIn("unique", str(ctx.exception).lower())
        self.assertIn("x", str(ctx.exception))

    def test_unique_inputs_outputs_preserved_order(self):
        """Unique inputs/outputs should preserve declaration order."""
        node = atomic_model.AtomicNode(
            fully_qualified_name="mod.func",
            inputs=["c", "a", "b"],
            outputs=["z", "x", "y"],
        )
        # Order must be preserved for function signature mapping
        self.assertEqual(node.inputs, ["c", "a", "b"])
        self.assertEqual(node.outputs, ["z", "x", "y"])

    def test_invalid_IO_labels(self):
        test_cases = [
            ("for", "Python keyword"),
            ("while", "Python keyword"),
            *[(reserved, "reserved name") for reserved in base_models.RESERVED_NAMES],
            ("1invalid", "not an identifier"),
            ("my-var", "not an identifier"),
            ("my var", "not an identifier"),
            ("", "not an identifier"),
        ]

        for io_type in ["inputs", "outputs"]:
            for invalid_label, reason in test_cases:
                with self.subTest(io_type=io_type, label=invalid_label, reason=reason):
                    with self.assertRaises(pydantic.ValidationError) as ctx:
                        kwargs = {
                            "fully_qualified_name": "mod.func",
                            "inputs": ["x"],
                            "outputs": ["y"],
                        }
                        kwargs[io_type] = [invalid_label, "valid"]
                        atomic_model.AtomicNode(**kwargs)

                    exc_str = str(ctx.exception)
                    self.assertIn(
                        "valid Python identifier",
                        exc_str,
                        f"{io_type} with {invalid_label} ({reason}) should fail",
                    )
                    if invalid_label:  # empty string won't appear in error
                        self.assertIn(invalid_label, exc_str)

    def test_valid_IO_labels(self):
        """Valid identifiers should pass."""
        node = atomic_model.AtomicNode(
            fully_qualified_name="mod.func",
            inputs=["x", "y_1", "_private", "camelCase"],
            outputs=["result", "status_code"],
        )
        self.assertEqual(len(node.inputs), 4)
        self.assertEqual(len(node.outputs), 2)


class TestNodeTypeImmutability(unittest.TestCase):
    """Tests that the 'type' field is immutable on NodeModel subclasses."""

    def test_type_field_cannot_be_overridden_at_construction(self):
        """AtomicNode should reject type override during instantiation."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            atomic_model.AtomicNode(
                type=base_models.RecipeElementType.WORKFLOW,  # Wrong type
                fully_qualified_name="mod.func",
                inputs=["x"],
                outputs=["y"],
            )
        exc_str = str(ctx.exception)
        self.assertIn("Input should be", exc_str)
        self.assertIn(base_models.RecipeElementType.ATOMIC.value, exc_str)

    def test_type_field_cannot_be_mutated_after_construction(self):
        """AtomicNode should reject mutation of type field."""
        node = atomic_model.AtomicNode(
            fully_qualified_name="mod.func",
            inputs=["x"],
            outputs=["y"],
        )
        with self.assertRaises(pydantic.ValidationError) as ctx:
            node.type = base_models.RecipeElementType.WORKFLOW
        exc_str = str(ctx.exception)
        self.assertIn("frozen", exc_str.lower())

    def test_subclass_must_provide_type_default(self):
        """NodeModel subclasses must provide a default value for 'type'."""
        with self.assertRaises(TypeError) as ctx:

            class BadNode(base_models.NodeModel):
                type: Literal[base_models.RecipeElementType.ATOMIC]  # No default
                inputs: list[str]
                outputs: list[str]

        exc_str = str(ctx.exception)
        self.assertIn("BadNode", exc_str)
        self.assertIn("default value for 'type'", exc_str)

    def test_subclass_must_freeze_type_field(self):
        """NodeModel subclasses must mark 'type' as frozen."""
        with self.assertRaises(TypeError) as ctx:

            class BadNode(base_models.NodeModel):
                type: Literal[base_models.RecipeElementType.ATOMIC] = (
                    base_models.RecipeElementType.ATOMIC
                )  # Not frozen
                inputs: list[str]
                outputs: list[str]

        exc_str = str(ctx.exception)
        self.assertIn("BadNode", exc_str)
        self.assertIn("frozen", exc_str)

    def test_subclass_must_redefine_type_field(self):
        """NodeModel subclasses must redefine 'type' field, not inherit base definition."""
        with self.assertRaises(TypeError) as ctx:

            class BadNode(base_models.NodeModel):
                # Doesn't mention 'type' at all
                inputs: list[str]
                outputs: list[str]

        exc_str = str(ctx.exception)
        self.assertIn("BadNode", exc_str)
        self.assertIn("default value for 'type'", exc_str)


if __name__ == "__main__":
    unittest.main()
