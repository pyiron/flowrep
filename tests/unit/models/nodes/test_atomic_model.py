import unittest

import pydantic

from flowrep.models import base_models
from flowrep.models.nodes import atomic_model, union


class TestAtomicNode(unittest.TestCase):
    """Tests for AtomicNode validation."""

    def test_schema_generation(self):
        """model_json_schema() fails if forward refs aren't resolved."""
        atomic_model.AtomicNode.model_json_schema()

    def test_valid_fqn(self):
        node = atomic_model.AtomicNode(
            fully_qualified_name="module.func",
            inputs=[],
            outputs=[],
        )
        self.assertEqual(node.fully_qualified_name, "module.func")
        self.assertEqual(node.type, base_models.RecipeElementType.ATOMIC)

    def test_valid_fqn_deep(self):
        node = atomic_model.AtomicNode(
            fully_qualified_name="a.b.c.d",
            inputs=[],
            outputs=[],
        )
        self.assertEqual(node.fully_qualified_name, "a.b.c.d")

    def test_fqn_no_period(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            atomic_model.AtomicNode(
                fully_qualified_name="noDot",
                inputs=[],
                outputs=[],
            )
        self.assertIn("at least one period", str(ctx.exception))

    def test_fqn_empty_string(self):
        with self.assertRaises(pydantic.ValidationError):
            atomic_model.AtomicNode(
                fully_qualified_name="",
                inputs=[],
                outputs=[],
            )

    def test_fqn_empty_part(self):
        """e.g., 'module.' or '.func' or 'a..b'"""
        for bad in ["module.", ".func", "a..b"]:
            with self.assertRaises(
                pydantic.ValidationError, msg=f"Should reject {bad!r}"
            ):
                atomic_model.AtomicNode(
                    fully_qualified_name=bad,
                    inputs=[],
                    outputs=[],
                )


class TestAtomicNodeUnpacking(unittest.TestCase):
    """Tests for unpack_mode validation."""

    def test_default_unpack_mode(self):
        """Default unpack_mode should be 'tuple'."""
        node = atomic_model.AtomicNode(
            fully_qualified_name="mod.func",
            inputs=["x"],
            outputs=["a", "b"],
        )
        self.assertEqual(node.unpack_mode, atomic_model.UnpackMode.TUPLE)

    def test_tuple_mode_multiple_outputs(self):
        """Multiple outputs allowed with unpack_mode='tuple'."""
        node = atomic_model.AtomicNode(
            fully_qualified_name="mod.func",
            inputs=[],
            outputs=["a", "b", "c"],
            unpack_mode=atomic_model.UnpackMode.TUPLE,
        )
        self.assertEqual(len(node.outputs), 3)
        self.assertEqual(node.unpack_mode, atomic_model.UnpackMode.TUPLE)

    def test_dataclass_mode_multiple_outputs(self):
        """Multiple outputs allowed with unpack_mode='dataclass'."""
        node = atomic_model.AtomicNode(
            fully_qualified_name="mod.func",
            inputs=[],
            outputs=["a", "b", "c"],
            unpack_mode=atomic_model.UnpackMode.DATACLASS,
        )
        self.assertEqual(len(node.outputs), 3)
        self.assertEqual(node.unpack_mode, atomic_model.UnpackMode.DATACLASS)

    def test_none_mode_multiple_outputs_rejected(self):
        """Multiple outputs rejected when unpack_mode='none'."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            atomic_model.AtomicNode(
                fully_qualified_name="mod.func",
                inputs=[],
                outputs=["a", "b"],
                unpack_mode=atomic_model.UnpackMode.NONE,
            )
        self.assertIn("exactly one element", str(ctx.exception))
        self.assertIn(
            f"unpack_mode={atomic_model.UnpackMode.NONE.value}", str(ctx.exception)
        )

    def test_none_mode_single_output_valid(self):
        """Single output valid with unpack_mode='none'."""
        node = atomic_model.AtomicNode(
            fully_qualified_name="mod.func",
            inputs=["x"],
            outputs=["result"],
            unpack_mode=atomic_model.UnpackMode.NONE,
        )
        self.assertEqual(node.outputs, ["result"])
        self.assertEqual(node.unpack_mode, atomic_model.UnpackMode.NONE)

    def test_none_mode_zero_outputs_valid(self):
        """Zero outputs valid with unpack_mode='none'."""
        node = atomic_model.AtomicNode(
            fully_qualified_name="mod.func",
            inputs=[],
            outputs=[],
            unpack_mode=atomic_model.UnpackMode.NONE,
        )
        self.assertEqual(len(node.outputs), 0)
        self.assertEqual(node.unpack_mode, atomic_model.UnpackMode.NONE)

    def test_tuple_mode_zero_outputs_valid(self):
        """Zero outputs valid with unpack_mode='tuple'."""
        node = atomic_model.AtomicNode(
            fully_qualified_name="mod.func",
            inputs=["x"],
            outputs=[],
            unpack_mode=atomic_model.UnpackMode.TUPLE,
        )
        self.assertEqual(len(node.outputs), 0)

    def test_all_unpack_modes_valid_literal(self):
        """All three unpack modes should be valid."""
        for mode in ["none", "tuple", "dataclass"]:
            node = atomic_model.AtomicNode(
                fully_qualified_name="mod.func",
                inputs=[],
                outputs=["out"],
                unpack_mode=mode,
            )
            self.assertEqual(node.unpack_mode, mode)


class TestAtomicNodeSerialization(unittest.TestCase):
    def test_atomic_roundtrip(self):
        original = atomic_model.AtomicNode(
            fully_qualified_name="mod.func",
            inputs=["a"],
            outputs=["b"],
        )
        for mode in ["json", "python"]:
            with self.subTest(mode=mode):
                data = original.model_dump(mode=mode)
                restored = atomic_model.AtomicNode.model_validate(data)
                self.assertEqual(original, restored)

    def test_discriminated_union_roundtrip(self):
        """Ensure type discriminator works for polymorphic deserialization."""
        data = {
            "type": base_models.RecipeElementType.ATOMIC,
            "fully_qualified_name": "a.b",
            "inputs": ["x"],
            "outputs": ["y"],
        }
        node = pydantic.TypeAdapter(union.NodeType).validate_python(data)
        self.assertIsInstance(node, atomic_model.AtomicNode)


if __name__ == "__main__":
    unittest.main()
