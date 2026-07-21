"""
End-to-end coverage of the ways ``@atomic`` can decorate class-ish targets.

Every target must be defined at module scope: :func:`flowrep.wfms.run_recipe`
resolves references by fully-qualified name at run time, so ``<locals>`` targets
cannot be executed.

Note that ``@atomic`` on a ``classmethod`` is *rejected* (its ``cls`` is bound
away at access time); that failure case lives in the unit tests, not here.
"""

import dataclasses
import numbers
import unittest
from typing import Generic, TypeVar

import flowrep as fr
from flowrep import base_models
from flowrep.retrospective.datastructures import NotData

from flowrep_static import library

_KEYWORD_ONLY = base_models.RestrictedParamKind.KEYWORD_ONLY


@fr.atomic
class Widget:
    def __init__(self, *, x: int = 100):
        self.x = x

    @fr.atomic
    def add(self, y):
        self.x += y
        return self.x

    @staticmethod
    @fr.atomic
    def triple(a: str):
        return 3 * a


@fr.atomic
@dataclasses.dataclass
class Point:
    x: int = 42


_TBoundInt = TypeVar("_TBoundInt", bound=int)


class GenericHolder(Generic[_TBoundInt]):
    def __init__(self, x: _TBoundInt):
        self.x = x


@fr.atomic
class ConcreteHolder(GenericHolder[int]): ...


_TReal = TypeVar("_TReal", bound=numbers.Real)
_UReal = TypeVar("_UReal", bound=numbers.Real)


class DeepBase(Generic[_TReal]):
    def __init__(self, x: _TReal):
        self.x = x


class DeepMiddle(DeepBase[_UReal], Generic[_UReal]):
    """Re-exports the type parameter under a new name."""


@fr.atomic
class DeepTwoLevel(DeepMiddle[int]):
    """Binds ``_UReal -> int``; resolving ``_TReal`` needs two-level composition."""


class TestAtomicMethod(unittest.TestCase):
    def test_method_runs_and_mutates_self(self):
        recipe = Widget.add.flowrep_recipe
        widget = Widget(x=42)
        out = fr.tools.run_recipe(recipe, self=widget, y=-2)
        self.assertEqual(out.output_ports["output_0"].value, 40)
        self.assertEqual(widget.x, 40, "The bound instance should be mutated in place")


class TestAtomicStaticmethod(unittest.TestCase):
    def test_staticmethod_preserved_and_runs(self):
        self.assertIsInstance(Widget.__dict__["triple"], staticmethod)
        recipe = Widget.triple.flowrep_recipe
        out = fr.tools.run_recipe(recipe, a="foo")
        self.assertEqual(out.output_ports["output_0"].value, "foofoofoo")


class TestAtomicClass(unittest.TestCase):
    def test_keyword_only_input_kind(self):
        recipe = Widget.flowrep_recipe
        self.assertEqual(recipe.reference.restricted_input_kinds, {"x": _KEYWORD_ONLY})

    def test_input_port_and_instance_output(self):
        out = fr.tools.run_recipe(Widget.flowrep_recipe, x=42)
        port = out.input_ports["x"]
        self.assertEqual(port.value, 42)
        self.assertIs(port.annotation, int)
        self.assertEqual(port.default, 100)

        instance = out.output_ports["instance"]
        self.assertIs(instance.annotation, Widget)
        self.assertEqual(instance.value.x, 42)


class TestAtomicDataclass(unittest.TestCase):
    def test_dataclass_input_uses_default(self):
        recipe = Point.flowrep_recipe
        self.assertEqual(recipe.reference.restricted_input_kinds, {})
        out = fr.tools.run_recipe(recipe)
        port = out.input_ports["x"]
        self.assertIsInstance(port.value, NotData)
        self.assertIs(port.annotation, int)
        self.assertEqual(port.default, 42)

        instance = out.output_ports["instance"]
        self.assertIs(instance.annotation, Point)
        self.assertEqual(instance.value.x, 42)


class TestAtomicConcreteGeneric(unittest.TestCase):
    def test_typevar_resolves_to_concrete_arg(self):
        out = fr.tools.run_recipe(ConcreteHolder.flowrep_recipe, x=100)
        port = out.input_ports["x"]
        self.assertEqual(port.value, 100)
        self.assertIs(
            port.annotation, int, "MyGeneric[int] should resolve _TBoundInt -> int"
        )

        instance = out.output_ports["instance"]
        self.assertIs(instance.annotation, ConcreteHolder)
        self.assertEqual(instance.value.x, 100)


class TestAtomicDeepGenericLimitation(unittest.TestCase):
    def test_two_level_typevar_stays_unresolved(self):
        # Documents a known limitation: single-level ``__orig_bases__`` mapping
        # cannot compose _TReal -> _UReal -> int, so the annotation stays a TypeVar.
        out = fr.tools.run_recipe(DeepTwoLevel.flowrep_recipe, x=100)
        port = out.input_ports["x"]
        self.assertEqual(port.value, 100)
        self.assertIsInstance(
            port.annotation,
            TypeVar,
            msg="This doesn't document a desired behaviour -- it would be lovely if we "
            "could resolve the int hint here -- this is just to document that "
            "resolving generics to port annotations has limits in the current "
            "implementation",
        )
        self.assertEqual(port.annotation.__name__, "_TReal")


class TestInverseRecipeExecution(unittest.TestCase):
    def test_autoencoder_round_trip_direct(self):
        self.assertEqual(library.autoencoder(42, "towel"), (42, "towel"))

    def test_autoencoder_round_trip_run_recipe(self):
        out = fr.tools.run_recipe(
            library.autoencoder.flowrep_recipe, foo=42, bar="towel"
        )
        self.assertEqual(out.output_ports["f"].value, 42)
        self.assertEqual(out.output_ports["b"].value, "towel")

    def test_single_autoencoder_round_trip_direct(self):
        self.assertEqual(library.single_autoencoder(7), 7)

    def test_single_autoencoder_round_trip_run_recipe(self):
        out = fr.tools.run_recipe(library.single_autoencoder.flowrep_recipe, only=7)
        self.assertEqual(out.output_ports["o"].value, 7)

    def test_inverse_recipe_run_directly(self):
        out = fr.tools.run_recipe(
            library.Pair.flowrep_recipe_unpacking, dataclass=library.Pair(1, "a")
        )
        self.assertEqual(out.output_ports["foo"].value, 1)
        self.assertEqual(out.output_ports["bar"].value, "a")

    def test_inverse_recipe_with_defaulted_field(self):
        # MyDataclass(a: ComplexData, x: int = 1) -- the default is irrelevant
        # when unpacking an existing instance.
        instance = library.MyDataclass(library.ComplexData(3), 7)
        out = fr.tools.run_recipe(
            library.MyDataclass.flowrep_recipe_unpacking, dataclass=instance
        )
        self.assertEqual(out.output_ports["a"].value.val, 3)
        self.assertEqual(out.output_ports["x"].value, 7)


if __name__ == "__main__":
    unittest.main()
