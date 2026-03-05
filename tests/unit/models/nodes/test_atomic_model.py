"""Unit tests for flowrep.models.nodes.atomic_model"""

import unittest

import pydantic

from flowrep.models import base_models
from flowrep.models.nodes import atomic_model

from flowrep_static import makers


class TestAtomicNodeStructure(unittest.TestCase):
    """Tests for basic structure and schema."""

    def test_schema_generation(self):
        """model_json_schema() fails if forward refs aren't resolved."""
        atomic_model.AtomicNode.model_json_schema()

    def test_type_defaults_to_atomic(self):
        node = atomic_model.AtomicNode(
            reference=makers.make_reference(),
            inputs=[],
            outputs=[],
        )
        self.assertEqual(node.type, base_models.RecipeElementType.ATOMIC)

    def test_required_fields(self):
        """source is required; inputs/outputs have no default."""
        with self.assertRaises(pydantic.ValidationError):
            atomic_model.AtomicNode()


class TestAtomicNodeSource(unittest.TestCase):
    """Tests for the source field and fully_qualified_name property."""

    def test_fully_qualified_name_property(self):
        node = atomic_model.AtomicNode(
            reference=makers.make_reference("module", "func"),
            inputs=[],
            outputs=[],
        )
        self.assertEqual(node.fully_qualified_name, "module.func")

    def test_fully_qualified_name_deep(self):
        node = atomic_model.AtomicNode(
            reference=makers.make_reference("a.b.c", "d"),
            inputs=[],
            outputs=[],
        )
        self.assertEqual(node.fully_qualified_name, "a.b.c.d")

    def test_version_accessible_via_source(self):
        node = atomic_model.AtomicNode(
            reference=makers.make_reference(version="1.2.3"),
            inputs=[],
            outputs=[],
        )
        self.assertEqual(node.reference.info.version, "1.2.3")

    def test_version_defaults_to_none(self):
        node = atomic_model.AtomicNode(
            reference=makers.make_reference(),
            inputs=[],
            outputs=[],
        )
        self.assertIsNone(node.reference.info.version)


class TestAtomicNodeUnpackMode(unittest.TestCase):
    """Tests for unpack_mode validation."""

    def test_default_unpack_mode_is_tuple(self):
        node = atomic_model.AtomicNode(
            reference=makers.make_reference(),
            inputs=[],
            outputs=["a", "b"],
        )
        self.assertEqual(node.unpack_mode, atomic_model.UnpackMode.TUPLE)

    def test_tuple_mode_multiple_outputs(self):
        node = atomic_model.AtomicNode(
            reference=makers.make_reference(),
            inputs=[],
            outputs=["a", "b", "c"],
            unpack_mode=atomic_model.UnpackMode.TUPLE,
        )
        self.assertEqual(len(node.outputs), 3)

    def test_tuple_mode_zero_outputs(self):
        node = atomic_model.AtomicNode(
            reference=makers.make_reference(),
            inputs=[],
            outputs=[],
            unpack_mode=atomic_model.UnpackMode.TUPLE,
        )
        self.assertEqual(len(node.outputs), 0)

    def test_dataclass_mode_multiple_outputs(self):
        node = atomic_model.AtomicNode(
            reference=makers.make_reference(),
            inputs=[],
            outputs=["a", "b", "c"],
            unpack_mode=atomic_model.UnpackMode.DATACLASS,
        )
        self.assertEqual(len(node.outputs), 3)
        self.assertEqual(node.unpack_mode, atomic_model.UnpackMode.DATACLASS)

    def test_none_mode_single_output(self):
        node = atomic_model.AtomicNode(
            reference=makers.make_reference(),
            inputs=[],
            outputs=["result"],
            unpack_mode=atomic_model.UnpackMode.NONE,
        )
        self.assertEqual(node.outputs, ["result"])

    def test_none_mode_zero_outputs(self):
        node = atomic_model.AtomicNode(
            reference=makers.make_reference(),
            inputs=[],
            outputs=[],
            unpack_mode=atomic_model.UnpackMode.NONE,
        )
        self.assertEqual(len(node.outputs), 0)

    def test_none_mode_multiple_outputs_rejected(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            atomic_model.AtomicNode(
                reference=makers.make_reference(),
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
                    reference=makers.make_reference(),
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
            reference=makers.make_reference(),
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
                    reference=makers.make_reference(),
                    inputs=[],
                    outputs=["out"],
                    unpack_mode=mode,
                )
                data = original.model_dump(mode="json")
                restored = atomic_model.AtomicNode.model_validate(data)
                self.assertEqual(original.unpack_mode, restored.unpack_mode)

    def test_roundtrip_with_version(self):
        original = atomic_model.AtomicNode(
            reference=makers.make_reference(version="0.5.1"),
            inputs=["a"],
            outputs=["b"],
        )
        for mode in ["json", "python"]:
            with self.subTest(mode=mode):
                data = original.model_dump(mode=mode)
                restored = atomic_model.AtomicNode.model_validate(data)
                self.assertEqual(
                    original.reference.info.version, restored.reference.info.version
                )

    def test_json_structure(self):
        node = atomic_model.AtomicNode(
            reference=makers.make_reference("pkg.mod", "func", "2.0.0"),
            inputs=["x", "y"],
            outputs=["z"],
            unpack_mode=atomic_model.UnpackMode.NONE,
        )
        data = node.model_dump(mode="json")
        self.assertEqual(data["type"], "atomic")
        self.assertEqual(
            data["reference"],
            {
                "info": {"module": "pkg.mod", "qualname": "func", "version": "2.0.0"},
                "inputs_with_defaults": [],
            },
        )
        self.assertEqual(data["inputs"], ["x", "y"])
        self.assertEqual(data["outputs"], ["z"])
        self.assertEqual(data["unpack_mode"], "none")

    def test_json_structure_version_null_when_absent(self):
        node = atomic_model.AtomicNode(
            reference=makers.make_reference(),
            inputs=[],
            outputs=[],
        )
        data = node.model_dump(mode="json")
        self.assertIsNone(data["reference"]["info"]["version"])

    def test_json_schema_includes_reference(self):
        schema = atomic_model.AtomicNode.model_json_schema()
        self.assertIn("reference", schema.get("properties", {}))


class TestAtomicNodeHasDefault(unittest.TestCase):
    """Tests for reference.inputs_with_defaults ⊆ inputs validation."""

    def test_empty_inputs_with_defaults(self):
        """Empty inputs_with_defaults is trivially valid."""
        node = atomic_model.AtomicNode(
            reference=makers.make_reference(),
            inputs=["a", "b"],
            outputs=[],
        )
        self.assertEqual(node.reference.inputs_with_defaults, [])

    def test_valid_subset(self):
        ref = makers.make_reference(inputs_with_defaults=["b"])
        node = atomic_model.AtomicNode(
            reference=ref,
            inputs=["a", "b"],
            outputs=[],
        )
        self.assertEqual(node.reference.inputs_with_defaults, ["b"])

    def test_full_match(self):
        ref = makers.make_reference(inputs_with_defaults=["a", "b"])
        node = atomic_model.AtomicNode(
            reference=ref,
            inputs=["a", "b"],
            outputs=[],
        )
        self.assertEqual(node.reference.inputs_with_defaults, ["a", "b"])

    def test_not_subset_rejected(self):
        ref = makers.make_reference(inputs_with_defaults=["a", "c"])
        with self.assertRaises(pydantic.ValidationError) as ctx:
            atomic_model.AtomicNode(
                reference=ref,
                inputs=["a", "b"],
                outputs=[],
            )
        self.assertIn("c", str(ctx.exception))

    def test_no_inputs_with_inputs_with_defaults_rejected(self):
        ref = makers.make_reference(inputs_with_defaults=["x"])
        with self.assertRaises(pydantic.ValidationError):
            atomic_model.AtomicNode(
                reference=ref,
                inputs=[],
                outputs=[],
            )


if __name__ == "__main__":
    unittest.main()
