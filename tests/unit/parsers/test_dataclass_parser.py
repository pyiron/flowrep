import dataclasses
import inspect
import typing
import unittest

from flowrep import base_models, wfms
from flowrep.parsers import atomic_parser, dataclass_parser
from flowrep.prospective import atomic_recipe
from flowrep.retrospective import datastructures

from flowrep_static import library


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


# `datastructures.recipe2data` resolves the bound unpacker via its fully qualified
# name, which requires it to be importable. Classes defined inline inside a test
# method have "<locals>" in their qualname and cannot be resolved this way, so the
# fixtures feeding recipe2data-based assertions must live at module scope.
@dataclass_parser.dataclass
class _ModuleWithClassVar:
    x: int
    tag: typing.ClassVar[str] = "t"


class TestInverseRecipeStructure(unittest.TestCase):
    def test_multi_field_recipe_shape(self):
        recipe = library.Pair.flowrep_recipe_unpacking
        self.assertIsInstance(recipe, atomic_recipe.AtomicRecipe)
        self.assertEqual(recipe.inputs, ["dataclass"])
        self.assertEqual(recipe.outputs, ["foo", "bar"])
        self.assertIs(recipe.unpack_mode, atomic_recipe.UnpackMode.TUPLE)
        self.assertEqual(
            recipe.reference.restricted_input_kinds,
            {"dataclass": base_models.RestrictedParamKind.POSITIONAL_ONLY},
        )

    def test_multi_field_output_annotations(self):
        ports = datastructures.recipe2data(
            library.Pair.flowrep_recipe_unpacking
        ).output_ports
        self.assertEqual(list(ports), ["foo", "bar"])
        self.assertIs(ports["foo"].annotation, int)
        self.assertIs(ports["bar"].annotation, str)

    def test_reference_and_unpacker_follow_the_class(self):
        unpacker = library.Pair._Pair_fields_to_outputs
        self.assertEqual(unpacker.__name__, "_Pair_fields_to_outputs")
        self.assertEqual(
            unpacker.__qualname__,
            f"{library.Pair.__qualname__}._Pair_fields_to_outputs",
        )
        self.assertEqual(unpacker.__module__, library.Pair.__module__)

        info = library.Pair.flowrep_recipe_unpacking.reference.info
        self.assertEqual(
            info.qualname, f"{library.Pair.__qualname__}._Pair_fields_to_outputs"
        )
        self.assertEqual(info.module, library.Pair.__module__)

    def test_single_field_recipe_shape(self):
        recipe = library.Single.flowrep_recipe_unpacking
        self.assertEqual(recipe.outputs, ["only"])
        ports = datastructures.recipe2data(recipe).output_ports
        self.assertEqual(list(ports), ["only"])
        self.assertIs(ports["only"].annotation, int)

    def test_single_field_return_annotation_is_bare_not_tuple(self):
        sig = inspect.signature(library.Single._Single_fields_to_outputs)
        self.assertIs(sig.return_annotation, int)

    def test_classvar_excluded_from_outputs(self):
        recipe = _ModuleWithClassVar.flowrep_recipe_unpacking
        self.assertEqual(recipe.outputs, ["x"])
        ports = datastructures.recipe2data(recipe).output_ports
        self.assertEqual(list(ports), ["x"])

    def test_unpacker_returns_tuple_for_multi_bare_for_single(self):
        self.assertEqual(
            library.Pair._Pair_fields_to_outputs(library.Pair(1, "a")), (1, "a")
        )
        self.assertEqual(library.Single._Single_fields_to_outputs(library.Single(5)), 5)

    def test_single_field_cannot_be_tuple_unpacked(self):
        result = library.Single._Single_fields_to_outputs(library.Single(5))
        self.assertEqual(result, 5)  # `o = ...` form works
        with self.assertRaises(TypeError):
            iter(result)  # `(o,) = ...` would raise: bare value, not a 1-tuple


class TestInverseRecipeFailures(unittest.TestCase):
    def test_forward_reference_annotation_raises_at_decoration(self):
        # We greedily resolve hints via get_type_hints; an unresolvable forward
        # reference fails at decoration time. This is an accepted limitation.
        with self.assertRaises(NameError):

            @dataclass_parser.dataclass
            class Fwd:
                val: "Undefined"  # noqa: F821

    def test_predefined_inverse_attr_is_guarded(self):
        with self.assertRaisesRegex(AttributeError, "refusing to overwrite"):

            @dataclass_parser.dataclass
            class Collide:
                foo: int
                flowrep_recipe_unpacking: int = 0


class TestRecipeMatchesPython(unittest.TestCase):
    """
    The inverse recipe is callable, so WfMS execution and raw python must agree.
    """

    def test_multi_field_inverse_recipe_call_matches_wfms_run(self):
        instance = library.Pair(1, "a")
        recipe = library.Pair.flowrep_recipe_unpacking

        raw = recipe(instance)
        node = wfms.run_recipe(recipe, dataclass=instance)
        via_wfms = tuple(node.output_ports[label].value for label in recipe.outputs)

        self.assertEqual(raw, (1, "a"), msg="Raw call unpacks the fields")
        self.assertEqual(
            raw,
            via_wfms,
            msg="Running the recipe through a WfMS must agree with calling it directly",
        )

    def test_single_field_inverse_recipe_call_matches_wfms_run(self):
        instance = library.Single(5)
        recipe = library.Single.flowrep_recipe_unpacking

        raw = recipe(instance)
        node = wfms.run_recipe(recipe, dataclass=instance)

        self.assertEqual(raw, 5, msg="A single field unpacks bare, not as a 1-tuple")
        self.assertEqual(raw, node.output_ports["only"].value)


if __name__ == "__main__":
    unittest.main()
