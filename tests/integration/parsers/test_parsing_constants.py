import unittest

import pydantic

from flowrep import wfms
from flowrep.compiler import source
from flowrep.parsers import workflow_parser
from flowrep.prospective import union_types

from flowrep_static import library, makers


def kinetic_energy(mass, velocity):
    v_2 = library.my_mul(velocity, velocity)
    mv_2 = library.my_mul(mass, v_2)
    ke = library.my_mul(0.5, mv_2)
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


if __name__ == "__main__":
    unittest.main()
