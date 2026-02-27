"""Unit tests for flowrep.models.nodes.atomic_model"""

import unittest

import pydantic

from flowrep.models import base_models
from flowrep.models.nodes import atomic_model


def _source(module: str = "mod", qualname: str = "func", version: str | None = None):
    """Shorthand for building a source dict in tests."""
    return {"module": module, "qualname": qualname, "version": version}


class TestAtomicNodeStructure(unittest.TestCase):
    """Tests for basic structure and schema."""

    def test_schema_generation(self):
        """model_json_schema() fails if forward refs aren't resolved."""
        atomic_model.AtomicNode.model_json_schema()

    def test_type_defaults_to_atomic(self):
        node = atomic_model.AtomicNode(
            source=_source(),
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
            source=_source("module", "func"),
            inputs=[],
            outputs=[],
        )
        self.assertEqual(node.fully_qualified_name, "module.func")

    def test_fully_qualified_name_deep(self):
        node = atomic_model.AtomicNode(
            source=_source("a.b.c", "d"),
            inputs=[],
            outputs=[],
        )
        self.assertEqual(node.fully_qualified_name, "a.b.c.d")

    def test_version_accessible_via_source(self):
        node = atomic_model.AtomicNode(
            source=_source(version="1.2.3"),
            inputs=[],
            outputs=[],
        )
        self.assertEqual(node.source.version, "1.2.3")

    def test_version_defaults_to_none(self):
        node = atomic_model.AtomicNode(
            source=_source(),
            inputs=[],
            outputs=[],
        )
        self.assertIsNone(node.source.version)


class TestAtomicNodeUnpackMode(unittest.TestCase):
    """Tests for unpack_mode validation."""

    def test_default_unpack_mode_is_tuple(self):
        node = atomic_model.AtomicNode(
            source=_source(),
            inputs=[],
            outputs=["a", "b"],
        )
        self.assertEqual(node.unpack_mode, atomic_model.UnpackMode.TUPLE)

    def test_tuple_mode_multiple_outputs(self):
        node = atomic_model.AtomicNode(
            source=_source(),
            inputs=[],
            outputs=["a", "b", "c"],
            unpack_mode=atomic_model.UnpackMode.TUPLE,
        )
        self.assertEqual(len(node.outputs), 3)

    def test_tuple_mode_zero_outputs(self):
        node = atomic_model.AtomicNode(
            source=_source(),
            inputs=[],
            outputs=[],
            unpack_mode=atomic_model.UnpackMode.TUPLE,
        )
        self.assertEqual(len(node.outputs), 0)

    def test_dataclass_mode_multiple_outputs(self):
        node = atomic_model.AtomicNode(
            source=_source(),
            inputs=[],
            outputs=["a", "b", "c"],
            unpack_mode=atomic_model.UnpackMode.DATACLASS,
        )
        self.assertEqual(len(node.outputs), 3)
        self.assertEqual(node.unpack_mode, atomic_model.UnpackMode.DATACLASS)

    def test_none_mode_single_output(self):
        node = atomic_model.AtomicNode(
            source=_source(),
            inputs=[],
            outputs=["result"],
            unpack_mode=atomic_model.UnpackMode.NONE,
        )
        self.assertEqual(node.outputs, ["result"])

    def test_none_mode_zero_outputs(self):
        node = atomic_model.AtomicNode(
            source=_source(),
            inputs=[],
            outputs=[],
            unpack_mode=atomic_model.UnpackMode.NONE,
        )
        self.assertEqual(len(node.outputs), 0)

    def test_none_mode_multiple_outputs_rejected(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            atomic_model.AtomicNode(
                source=_source(),
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
                    source=_source(),
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
            source=_source(),
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
                    source=_source(),
                    inputs=[],
                    outputs=["out"],
                    unpack_mode=mode,
                )
                data = original.model_dump(mode="json")
                restored = atomic_model.AtomicNode.model_validate(data)
                self.assertEqual(original.unpack_mode, restored.unpack_mode)

    def test_roundtrip_with_version(self):
        original = atomic_model.AtomicNode(
            source=_source(version="0.5.1"),
            inputs=["a"],
            outputs=["b"],
        )
        for mode in ["json", "python"]:
            with self.subTest(mode=mode):
                data = original.model_dump(mode=mode)
                restored = atomic_model.AtomicNode.model_validate(data)
                self.assertEqual(original.source.version, restored.source.version)

    def test_json_structure(self):
        node = atomic_model.AtomicNode(
            source=_source("pkg.mod", "func", "2.0.0"),
            inputs=["x", "y"],
            outputs=["z"],
            unpack_mode=atomic_model.UnpackMode.NONE,
        )
        data = node.model_dump(mode="json")
        self.assertEqual(data["type"], "atomic")
        self.assertEqual(
            data["source"],
            {"module": "pkg.mod", "qualname": "func", "version": "2.0.0"},
        )
        self.assertEqual(data["inputs"], ["x", "y"])
        self.assertEqual(data["outputs"], ["z"])
        self.assertEqual(data["unpack_mode"], "none")

    def test_json_structure_version_null_when_absent(self):
        node = atomic_model.AtomicNode(
            source=_source(),
            inputs=[],
            outputs=[],
        )
        data = node.model_dump(mode="json")
        self.assertIsNone(data["source"]["version"])

    def test_json_schema_includes_source(self):
        schema = atomic_model.AtomicNode.model_json_schema()
        self.assertIn("source", schema.get("properties", {}))


class TestAtomicNodeSourceCode(unittest.TestCase):
    """Tests for the optional source_code field."""

    def test_defaults_to_none(self):
        node = atomic_model.AtomicNode(source=_source(), inputs=[], outputs=[])
        self.assertIsNone(node.source_code)

    def test_accepts_string(self):
        node = atomic_model.AtomicNode(
            source=_source(),
            inputs=[],
            outputs=[],
            source_code="def func(): pass",
        )
        self.assertEqual(node.source_code, "def func(): pass")

    def test_roundtrip(self):
        original = atomic_model.AtomicNode(
            source=_source(),
            inputs=["a"],
            outputs=["b"],
            source_code="def func(a):\n    return a",
        )
        for mode in ["json", "python"]:
            with self.subTest(mode=mode):
                data = original.model_dump(mode=mode)
                restored = atomic_model.AtomicNode.model_validate(data)
                self.assertEqual(original.source_code, restored.source_code)

    def test_json_structure_null_when_absent(self):
        node = atomic_model.AtomicNode(source=_source(), inputs=[], outputs=[])
        self.assertIsNone(node.model_dump(mode="json")["source_code"])


if __name__ == "__main__":
    unittest.main()
