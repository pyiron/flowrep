"""Unit tests for flowrep.models.nodes.atomic_model"""

import unittest

import pydantic

from flowrep.models import base_models
from flowrep.models.nodes import atomic_model


class TestAtomicNodeStructure(unittest.TestCase):
    """Tests for basic structure and schema."""

    def test_schema_generation(self):
        """model_json_schema() fails if forward refs aren't resolved."""
        atomic_model.AtomicNode.model_json_schema()

    def test_type_defaults_to_atomic(self):
        node = atomic_model.AtomicNode(
            fully_qualified_name="mod.func",
            inputs=[],
            outputs=[],
        )
        self.assertEqual(node.type, base_models.RecipeElementType.ATOMIC)

    def test_required_fields(self):
        """fully_qualified_name is required; inputs/outputs have no default."""
        with self.assertRaises(pydantic.ValidationError):
            atomic_model.AtomicNode()


class TestAtomicNodeFQN(unittest.TestCase):
    """Tests for fully_qualified_name validation."""

    def test_valid_fqn_simple(self):
        node = atomic_model.AtomicNode(
            fully_qualified_name="module.func",
            inputs=[],
            outputs=[],
        )
        self.assertEqual(node.fully_qualified_name, "module.func")

    def test_valid_fqn_deep(self):
        node = atomic_model.AtomicNode(
            fully_qualified_name="a.b.c.d",
            inputs=[],
            outputs=[],
        )
        self.assertEqual(node.fully_qualified_name, "a.b.c.d")

    def test_fqn_no_period_rejected(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            atomic_model.AtomicNode(
                fully_qualified_name="noDot",
                inputs=[],
                outputs=[],
            )
        self.assertIn("at least one period", str(ctx.exception))

    def test_fqn_empty_string_rejected(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            atomic_model.AtomicNode(
                fully_qualified_name="",
                inputs=[],
                outputs=[],
            )
        self.assertIn("at least one period", str(ctx.exception))

    def test_fqn_empty_parts_rejected(self):
        """e.g., 'module.' or '.func' or 'a..b'"""
        invalid_fqns = [
            ("module.", "trailing dot"),
            (".func", "leading dot"),
            ("a..b", "consecutive dots"),
            (".", "single dot"),
            ("..", "only dots"),
        ]
        for fqn, reason in invalid_fqns:
            with self.subTest(fqn=fqn, reason=reason):
                with self.assertRaises(pydantic.ValidationError):
                    atomic_model.AtomicNode(
                        fully_qualified_name=fqn,
                        inputs=[],
                        outputs=[],
                    )


class TestAtomicNodeUnpackMode(unittest.TestCase):
    """Tests for unpack_mode validation."""

    def test_default_unpack_mode_is_tuple(self):
        node = atomic_model.AtomicNode(
            fully_qualified_name="mod.func",
            inputs=[],
            outputs=["a", "b"],
        )
        self.assertEqual(node.unpack_mode, atomic_model.UnpackMode.TUPLE)

    def test_tuple_mode_multiple_outputs(self):
        node = atomic_model.AtomicNode(
            fully_qualified_name="mod.func",
            inputs=[],
            outputs=["a", "b", "c"],
            unpack_mode=atomic_model.UnpackMode.TUPLE,
        )
        self.assertEqual(len(node.outputs), 3)

    def test_tuple_mode_zero_outputs(self):
        node = atomic_model.AtomicNode(
            fully_qualified_name="mod.func",
            inputs=[],
            outputs=[],
            unpack_mode=atomic_model.UnpackMode.TUPLE,
        )
        self.assertEqual(len(node.outputs), 0)

    def test_dataclass_mode_multiple_outputs(self):
        node = atomic_model.AtomicNode(
            fully_qualified_name="mod.func",
            inputs=[],
            outputs=["a", "b", "c"],
            unpack_mode=atomic_model.UnpackMode.DATACLASS,
        )
        self.assertEqual(len(node.outputs), 3)
        self.assertEqual(node.unpack_mode, atomic_model.UnpackMode.DATACLASS)

    def test_none_mode_single_output(self):
        node = atomic_model.AtomicNode(
            fully_qualified_name="mod.func",
            inputs=[],
            outputs=["result"],
            unpack_mode=atomic_model.UnpackMode.NONE,
        )
        self.assertEqual(node.outputs, ["result"])

    def test_none_mode_zero_outputs(self):
        node = atomic_model.AtomicNode(
            fully_qualified_name="mod.func",
            inputs=[],
            outputs=[],
            unpack_mode=atomic_model.UnpackMode.NONE,
        )
        self.assertEqual(len(node.outputs), 0)

    def test_none_mode_multiple_outputs_rejected(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            atomic_model.AtomicNode(
                fully_qualified_name="mod.func",
                inputs=[],
                outputs=["a", "b"],
                unpack_mode=atomic_model.UnpackMode.NONE,
            )
        self.assertIn("exactly one element", str(ctx.exception))
        self.assertIn(atomic_model.UnpackMode.NONE.value, str(ctx.exception))

    def test_all_unpack_modes_accepted_as_string(self):
        for mode in ["none", "tuple", "dataclass"]:
            with self.subTest(mode=mode):
                node = atomic_model.AtomicNode(
                    fully_qualified_name="mod.func",
                    inputs=[],
                    outputs=["out"],
                    unpack_mode=mode,
                )
                self.assertEqual(node.unpack_mode, mode)


class TestUnpackModeEnum(unittest.TestCase):
    """Tests for UnpackMode enum."""

    def test_all_expected_values_exist(self):
        expected = {"none", "tuple", "dataclass"}
        actual = {e.value for e in atomic_model.UnpackMode}
        self.assertEqual(expected, actual)

    def test_is_str_enum(self):
        for mode in atomic_model.UnpackMode:
            self.assertIsInstance(mode, str)
            self.assertEqual(mode, mode.value)


class TestAtomicNodeSerialization(unittest.TestCase):
    """Tests for serialization roundtrips."""

    def test_roundtrip(self):
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

    def test_roundtrip_with_unpack_mode(self):
        for mode in atomic_model.UnpackMode:
            with self.subTest(mode=mode):
                original = atomic_model.AtomicNode(
                    fully_qualified_name="mod.func",
                    inputs=[],
                    outputs=["out"],
                    unpack_mode=mode,
                )
                data = original.model_dump(mode="json")
                restored = atomic_model.AtomicNode.model_validate(data)
                self.assertEqual(original.unpack_mode, restored.unpack_mode)

    def test_json_structure(self):
        node = atomic_model.AtomicNode(
            fully_qualified_name="pkg.mod.func",
            inputs=["x", "y"],
            outputs=["z"],
            unpack_mode=atomic_model.UnpackMode.NONE,
        )
        data = node.model_dump(mode="json")
        self.assertEqual(data["type"], "atomic")
        self.assertEqual(data["fully_qualified_name"], "pkg.mod.func")
        self.assertEqual(data["inputs"], ["x", "y"])
        self.assertEqual(data["outputs"], ["z"])
        self.assertEqual(data["unpack_mode"], "none")


if __name__ == "__main__":
    unittest.main()
