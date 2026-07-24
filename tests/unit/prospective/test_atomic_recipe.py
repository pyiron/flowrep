"""Unit tests for flowrep.models.nodes.atomic_recipe"""

import unittest

import pydantic

from flowrep import base_models
from flowrep.prospective import atomic_recipe

from flowrep_static import makers


class TestAtomicRecipeStructure(unittest.TestCase):
    """Tests for basic structure and schema."""

    def test_schema_generation(self):
        """model_json_schema() fails if forward refs aren't resolved."""
        atomic_recipe.AtomicRecipe.model_json_schema()

    def test_type_defaults_to_atomic(self):
        node = atomic_recipe.AtomicRecipe(
            reference=makers.make_reference(),
            inputs=[],
            outputs=[],
        )
        self.assertEqual(node.type, base_models.RecipeElementType.ATOMIC)

    def test_required_fields(self):
        """source is required; inputs/outputs have no default."""
        with self.assertRaises(pydantic.ValidationError):
            atomic_recipe.AtomicRecipe()


class TestAtomicRecipeSource(unittest.TestCase):
    """Tests for the source field and fully_qualified_name property."""

    def test_fully_qualified_name_property(self):
        node = atomic_recipe.AtomicRecipe(
            reference=makers.make_reference("module", "func"),
            inputs=[],
            outputs=[],
        )
        self.assertEqual(node.fully_qualified_name, "module.func")

    def test_fully_qualified_name_deep(self):
        node = atomic_recipe.AtomicRecipe(
            reference=makers.make_reference("a.b.c", "d"),
            inputs=[],
            outputs=[],
        )
        self.assertEqual(node.fully_qualified_name, "a.b.c.d")

    def test_version_accessible_via_source(self):
        node = atomic_recipe.AtomicRecipe(
            reference=makers.make_reference(version="1.2.3"),
            inputs=[],
            outputs=[],
        )
        self.assertEqual(node.reference.info.version, "1.2.3")

    def test_version_defaults_to_none(self):
        node = atomic_recipe.AtomicRecipe(
            reference=makers.make_reference(),
            inputs=[],
            outputs=[],
        )
        self.assertIsNone(node.reference.info.version)


class TestAtomicRecipeSerialization(unittest.TestCase):
    """Tests for serialization roundtrips."""

    def test_roundtrip(self):
        original = atomic_recipe.AtomicRecipe(
            reference=makers.make_reference(),
            inputs=["a"],
            outputs=["b"],
        )
        for mode in ["json", "python"]:
            with self.subTest(mode=mode):
                data = original.model_dump(mode=mode)
                restored = atomic_recipe.AtomicRecipe.model_validate(data)
                self.assertEqual(original, restored)

    def test_roundtrip_with_version(self):
        original = atomic_recipe.AtomicRecipe(
            reference=makers.make_reference(version="0.5.1"),
            inputs=["a"],
            outputs=["b"],
        )
        for mode in ["json", "python"]:
            with self.subTest(mode=mode):
                data = original.model_dump(mode=mode)
                restored = atomic_recipe.AtomicRecipe.model_validate(data)
                self.assertEqual(
                    original.reference.info.version, restored.reference.info.version
                )

    def test_json_structure(self):
        node = atomic_recipe.AtomicRecipe(
            reference=makers.make_reference("pkg.mod", "func", "2.0.0"),
            inputs=["x", "y"],
            outputs=["z"],
        )
        data = node.model_dump(mode="json")
        self.assertEqual(data["type"], "atomic")
        self.assertEqual(
            data["reference"],
            {
                "info": {"module": "pkg.mod", "qualname": "func", "version": "2.0.0"},
                "inputs_with_defaults": [],
                "restricted_input_kinds": {},
            },
        )
        self.assertEqual(data["inputs"], ["x", "y"])
        self.assertEqual(data["outputs"], ["z"])

    def test_json_structure_version_null_when_absent(self):
        node = atomic_recipe.AtomicRecipe(
            reference=makers.make_reference(),
            inputs=[],
            outputs=[],
        )
        data = node.model_dump(mode="json")
        self.assertIsNone(data["reference"]["info"]["version"])

    def test_json_schema_includes_reference(self):
        schema = atomic_recipe.AtomicRecipe.model_json_schema()
        self.assertIn("reference", schema.get("properties", {}))


class TestAtomicRecipeHasDefault(unittest.TestCase):
    """Tests for reference.inputs_with_defaults ⊆ inputs validation."""

    def test_empty_inputs_with_defaults(self):
        """Empty inputs_with_defaults is trivially valid."""
        node = atomic_recipe.AtomicRecipe(
            reference=makers.make_reference(),
            inputs=["a", "b"],
            outputs=[],
        )
        self.assertEqual(node.reference.inputs_with_defaults, [])

    def test_valid_subset(self):
        ref = makers.make_reference(inputs_with_defaults=["b"])
        node = atomic_recipe.AtomicRecipe(
            reference=ref,
            inputs=["a", "b"],
            outputs=[],
        )
        self.assertEqual(node.reference.inputs_with_defaults, ["b"])

    def test_full_match(self):
        ref = makers.make_reference(inputs_with_defaults=["a", "b"])
        node = atomic_recipe.AtomicRecipe(
            reference=ref,
            inputs=["a", "b"],
            outputs=[],
        )
        self.assertEqual(node.reference.inputs_with_defaults, ["a", "b"])

    def test_not_subset_rejected(self):
        ref = makers.make_reference(inputs_with_defaults=["a", "c"])
        with self.assertRaises(pydantic.ValidationError) as ctx:
            atomic_recipe.AtomicRecipe(
                reference=ref,
                inputs=["a", "b"],
                outputs=[],
            )
        self.assertIn("c", str(ctx.exception))

    def test_no_inputs_with_inputs_with_defaults_rejected(self):
        ref = makers.make_reference(inputs_with_defaults=["x"])
        with self.assertRaises(pydantic.ValidationError):
            atomic_recipe.AtomicRecipe(
                reference=ref,
                inputs=[],
                outputs=[],
            )


class TestAtomicRecipeCall(unittest.TestCase):
    def test_call_on_bad_reference(self):
        recipe = makers.make_atomic(inputs=["x", "y"])
        with self.assertRaises(
            ModuleNotFoundError,
            msg="Neither the recipe nor the call validate that the underlying python "
            "funcion reference is actually there -- so attempting to run it should "
            "only fail at the point we actually try to import our made-up reference",
        ):
            recipe(x=1, y=2)

    def test_call(self):
        """
        Calling atomic recipes should import and execute their underlying function
        """
        recipe = atomic_recipe.AtomicRecipe(
            reference=makers.make_reference("builtins", "int"),
            inputs=["x_str"],
            outputs=["x_int"],
        )
        result = recipe("1")
        self.assertIsInstance(result, int)
        self.assertEqual(result, 1)


if __name__ == "__main__":
    unittest.main()
