import unittest

import pydantic

from flowrep import edge_models, std, wfms
from flowrep.compiler import source
from flowrep.parsers import workflow_parser
from flowrep.prospective import union_types

from flowrep_static import library, makers


def kinetic_energy(mass, velocity):
    v_2 = std.mul(velocity, velocity)
    mv_2 = std.mul(mass, v_2)
    ke = std.mul(0.5, mv_2)
    return ke


class TestConstantEndToEnd(unittest.TestCase):
    def test_parse_serialize_run_compile(self):
        recipe = workflow_parser.parse_workflow(kinetic_energy)

        # 1. Serializes and round-trips through the discriminated union.
        adapter = pydantic.TypeAdapter(union_types.RecipeDiscrimination)
        restored = adapter.validate_python(adapter.dump_python(recipe, mode="json"))
        self.assertEqual(makers.dump_no_refs(restored), makers.dump_no_refs(recipe))

        # 2. Runs to the correct number.
        result = wfms.run_recipe(recipe, mass=2.0, velocity=3.0)
        self.assertAlmostEqual(result.output_ports["ke"].value, 9.0)

        # 3. Compiles back to source, re-parses to an equal recipe.
        free = makers.reference_free(kinetic_energy)
        fn = source._workflow2python(free).build()
        self.assertEqual(
            makers.dump_no_refs(fn.flowrep_recipe), makers.dump_no_refs(free)
        )


@workflow_parser.workflow
def shadowed_constant_symbol(x):
    """A user symbol named like a generated port must not be clobbered by one."""
    constant_0 = std.neg(x)
    if library.my_condition(x, 3):  # noqa: SIM108
        y = std.add(constant_0, x)
    else:
        y = std.identity(x)
    return y


class TestGeneratedPortDodgesUserSymbols(unittest.TestCase):
    """A flow-control node's inputs include its *body's* inputs, so a generated port
    must dodge every enclosing symbol -- not just the condition's arguments."""

    def test_generated_port_dodges_the_user_symbol(self):
        recipe = shadowed_constant_symbol.flowrep_recipe
        # The literal 3's port steps aside for the user's `constant_0`.
        self.assertEqual(recipe.nodes["if_0"].inputs, ["x", "constant_1", "constant_0"])

    def test_body_reads_the_user_symbol_not_the_literal(self):
        recipe = shadowed_constant_symbol.flowrep_recipe
        self.assertEqual(
            recipe.edges[edge_models.TargetHandle(node="if_0", port="constant_0")],
            edge_models.SourceHandle(node="neg_0", port="negative"),
        )

    def test_generated_port_reads_the_constant_peer(self):
        recipe = shadowed_constant_symbol.flowrep_recipe
        self.assertEqual(
            recipe.edges[edge_models.TargetHandle(node="if_0", port="constant_1")],
            edge_models.SourceHandle(node="constant_0", port="constant"),
        )

    def test_executes_via_run_recipe(self):
        result = wfms.run_recipe(shadowed_constant_symbol.flowrep_recipe, x=1)
        self.assertEqual(result.output_ports["y"].value, shadowed_constant_symbol(1))

    def test_round_trips_through_source(self):
        free = makers.reference_free(shadowed_constant_symbol)
        rendered = source._workflow2python(free)
        fn = rendered.build()
        self.assertEqual(
            makers.dump_no_refs(fn.flowrep_recipe), makers.dump_no_refs(free)
        )


if __name__ == "__main__":
    unittest.main()
