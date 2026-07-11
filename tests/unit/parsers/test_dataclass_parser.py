import dataclasses
import inspect
import unittest

from flowrep.parsers import atomic_parser, dataclass_parser
from flowrep.prospective import atomic_recipe


class TestDataclassDecorator(unittest.TestCase):
    def test_bare_stacks_dataclass_and_atomic(self):
        @dataclass_parser.dataclass
        class Point:
            x: int = 1
            y: str = "a"

        self.assertTrue(
            dataclasses.is_dataclass(Point), "Should apply @dataclasses.dataclass"
        )
        self.assertIsInstance(Point.flowrep_recipe, atomic_recipe.AtomicRecipe)
        self.assertEqual(Point.flowrep_recipe.inputs, ["x", "y"])
        self.assertEqual(Point.flowrep_recipe.outputs, ["instance"])

    def test_empty_parens(self):
        @dataclass_parser.dataclass()
        class Point:
            x: int = 1

        self.assertTrue(dataclasses.is_dataclass(Point))
        self.assertEqual(Point.flowrep_recipe.outputs, ["instance"])

    def test_captures_field_defaults(self):
        @dataclass_parser.dataclass
        class Point:
            x: int = 42
            y: int = 7

        self.assertEqual(
            Point.flowrep_recipe.reference.inputs_with_defaults, ["x", "y"]
        )
        self.assertEqual(Point(), Point(x=42, y=7))


class TestDataclassOutputLabels(unittest.TestCase):
    def test_explicit_output_label(self):
        @dataclass_parser.dataclass("thing")
        class Point:
            x: int = 1

        self.assertEqual(Point.flowrep_recipe.outputs, ["thing"])

    def test_too_many_output_labels_raises(self):
        with self.assertRaises(ValueError):

            @dataclass_parser.dataclass("a", "b")
            class Point:
                x: int = 1


class TestDataclassKwargRouting(unittest.TestCase):
    def test_dataclass_kwarg_routed(self):
        @dataclass_parser.dataclass(frozen=True)
        class Point:
            x: int = 1

        self.assertTrue(Point.__dataclass_params__.frozen)
        self.assertTrue(hasattr(Point, "flowrep_recipe"))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            Point(x=1).x = 2

    def test_atomic_kwarg_routed(self):
        # forbid_locals belongs to @atomic; a class defined in this method has
        # "<locals>" in its qualname, so routing it through should raise.
        with self.assertRaisesRegex(ValueError, "<locals>"):

            @dataclass_parser.dataclass(forbid_locals=True)
            class Point:
                x: int = 1

    def test_both_kwargs_routed_together(self):
        @dataclass_parser.dataclass(frozen=True, forbid_main=False)
        class Point:
            x: int = 1

        self.assertTrue(Point.__dataclass_params__.frozen)
        self.assertEqual(Point.flowrep_recipe.inputs, ["x"])


class TestDataclassGuardsInherited(unittest.TestCase):
    def test_recipe_attribute_collision_still_guarded(self):
        # The collision guard lives in @atomic and should apply transitively.
        with self.assertRaisesRegex(TypeError, "flowrep_recipe"):

            @dataclass_parser.dataclass
            class Point:
                flowrep_recipe: int = 1


class TestNamespaceOverlap(unittest.TestCase):
    def test_dataclass_and_atomic_namespace_overlap(self):
        atomic_kwargs = frozenset(
            name
            for name, param in inspect.signature(
                atomic_parser.parse_atomic
            ).parameters.items()
            if param.kind is inspect.Parameter.KEYWORD_ONLY
        )
        self.assertFalse(
            set(dataclass_parser._DATACLASS_KWARGS).intersection(atomic_kwargs),
            msg="Ensure that dataclass and atomic kwargs don't overlap.",
        )


if __name__ == "__main__":
    unittest.main()
