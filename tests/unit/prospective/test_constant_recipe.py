import unittest

import pydantic

from flowrep import base_models
from flowrep.prospective import constant_recipe, union_types


class TestIsJsonable(unittest.TestCase):
    def test_is_jsonable(self):
        self.assertTrue(constant_recipe.is_jsonable([1, 2]))
        self.assertFalse(constant_recipe.is_jsonable((1, 2)))


class TestConstantRecipe(unittest.TestCase):
    def test_defaults_and_call(self):
        c = constant_recipe.ConstantRecipe(constant=0.5)
        self.assertEqual(c.type, base_models.RecipeElementType.CONSTANT)
        self.assertEqual(c.inputs, [])
        self.assertEqual(c.outputs, ["constant"])
        self.assertEqual(c.constant, 0.5)
        self.assertEqual(c(), 0.5)  # __call__ returns the value

    def test_type_fidelity_preserved(self):
        for value, expected_type in [
            (1, int),
            (1.0, float),
            (True, bool),
            (None, type(None)),
        ]:
            with self.subTest(value=value):
                c = constant_recipe.ConstantRecipe(constant=value)
                self.assertIs(type(c.constant), expected_type)

    def test_compound_value(self):
        value = [1.0, 1, "1", {"key": [42]}]
        c = constant_recipe.ConstantRecipe(constant=value)
        self.assertEqual(c.constant, value)

    def test_rejects_non_json(self):
        for bad in [(1, 2), {1, 2}, [1, (2, 3)], {1: "int-key"}, {"k": (1,)}]:
            with self.subTest(bad=bad), self.assertRaises(pydantic.ValidationError):
                constant_recipe.ConstantRecipe(constant=bad)

    def test_rejects_non_finite_float(self):
        for bad in [float("nan"), float("inf"), float("-inf")]:
            with self.subTest(bad=bad), self.assertRaises(pydantic.ValidationError):
                constant_recipe.ConstantRecipe(constant=bad)

    def test_bad_ports_rejected(self):
        with self.assertRaises(pydantic.ValidationError):
            constant_recipe.ConstantRecipe(constant=1, inputs=["x"])
        with self.assertRaises(pydantic.ValidationError):
            constant_recipe.ConstantRecipe(constant=1, outputs=["wrong"])

    def test_discriminated_union_roundtrip(self):
        adapter = pydantic.TypeAdapter(union_types.RecipeDiscrimination)
        data = {
            "type": "constant",
            "inputs": [],
            "outputs": ["constant"],
            "constant": [1.0, 1, "1", {"key": [42]}],
        }
        node = adapter.validate_python(data)
        self.assertIsInstance(node, constant_recipe.ConstantRecipe)
        restored = adapter.validate_python(adapter.dump_python(node, mode="json"))
        self.assertEqual(node, restored)


if __name__ == "__main__":
    unittest.main()
