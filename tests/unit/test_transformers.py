"""Tests for the scatter/gather transformer recipes."""

from __future__ import annotations

import unittest

from flowrep import transformers, wfms


class TestTransform1toN(unittest.TestCase):
    def test_recipe_shape(self):
        recipe = transformers.Transform1toN(3).recipe
        self.assertEqual(recipe.inputs, ["items"])
        self.assertEqual(recipe.outputs, ["output_0", "output_1", "output_2"])

    def test_scatters(self):
        data = wfms.run_recipe(transformers.Transform1toN(3).recipe, items=[10, 20, 30])
        self.assertEqual(
            [port.value for port in data.output_ports.values()], [10, 20, 30]
        )

    def test_single_output_receives_the_element_not_a_tuple(self):
        """A recipe declaring exactly one output receives the whole return value, so a
        1-wide scatter must return the element itself."""
        data = wfms.run_recipe(transformers.Transform1toN(1).recipe, items=[42])
        self.assertEqual(data.output_ports["output_0"].value, 42)

    def test_zero_outputs_raises(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            transformers.Transform1toN(0)


class TestTransformNto1(unittest.TestCase):
    def test_recipe_shape(self):
        recipe = transformers.TransformNto1(2).recipe
        self.assertEqual(recipe.inputs, ["item_0", "item_1"])
        self.assertEqual(recipe.outputs, ["output_0"])

    def test_gathers(self):
        data = wfms.run_recipe(
            transformers.TransformNto1(3).recipe, item_0=1, item_1=2, item_2=3
        )
        self.assertEqual(data.output_ports["output_0"].value, [1, 2, 3])

    def test_zero_inputs_gathers_an_empty_list(self):
        """The empty-iterable for-loop builds aggregators with nothing to gather."""
        data = wfms.run_recipe(transformers.TransformNto1(0).recipe)
        self.assertEqual(data.output_ports["output_0"].value, [])

    def test_negative_raises(self):
        with self.assertRaisesRegex(ValueError, "at least 0"):
            transformers.TransformNto1(-1)
