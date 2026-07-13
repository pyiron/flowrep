import unittest

from flowrep import edge_models, wfms
from flowrep.compiler import source
from flowrep.parsers import workflow_parser

from flowrep_static import library, makers


@workflow_parser.workflow
def wf(x0: int, comp: library.ComplexData):
    dc = library.MyDataclass(comp, x0)
    my_val = dc.a.val
    repeated = library.my_mul(dc.x, my_val)
    return repeated


class TestAttributeAccessEndToEnd(unittest.TestCase):
    def test_parses_expected_graph(self):
        recipe = wf.flowrep_recipe

        self.assertEqual(
            set(recipe.nodes),
            {
                "MyDataclass_0",
                "getattr_a_0",
                "getattr_val_0",
                "getattr_x_0",
                "my_mul_0",
                "constant_0",
                "constant_1",
                "constant_2",
            },
        )
        self.assertEqual(
            recipe.edges[edge_models.TargetHandle(node="getattr_a_0", port="obj")],
            edge_models.SourceHandle(node="MyDataclass_0", port="instance"),
        )
        self.assertEqual(
            recipe.edges[edge_models.TargetHandle(node="getattr_val_0", port="obj")],
            edge_models.SourceHandle(node="getattr_a_0", port="attr"),
        )
        self.assertEqual(
            recipe.edges[edge_models.TargetHandle(node="getattr_x_0", port="obj")],
            edge_models.SourceHandle(node="MyDataclass_0", port="instance"),
        )
        self.assertEqual(
            recipe.edges[edge_models.TargetHandle(node="my_mul_0", port="a")],
            edge_models.SourceHandle(node="getattr_x_0", port="attr"),
        )
        self.assertEqual(
            recipe.edges[edge_models.TargetHandle(node="my_mul_0", port="b")],
            edge_models.SourceHandle(node="getattr_val_0", port="attr"),
        )

    def test_executes_via_run_recipe(self):
        result = wfms.run_recipe(
            wf.flowrep_recipe, x0=3, comp=library.ComplexData(val=7)
        )
        expected = wf(3, library.ComplexData(val=7))
        self.assertEqual(result.output_ports["repeated"].value, expected)
        self.assertEqual(expected, 21)

    def test_round_trips_through_source(self):
        free = makers.reference_free(wf)
        rendered = source._workflow2python(free)
        fn = rendered.build()
        self.assertEqual(
            fn(3, library.ComplexData(val=7)), wf(3, library.ComplexData(val=7))
        )
        self.assertEqual(
            makers.dump_no_refs(fn.flowrep_recipe), makers.dump_no_refs(free)
        )

    def test_compiled_source_is_sugared(self):
        free = makers.reference_free(wf)
        rendered = source._workflow2python(free)
        # Attribute syntax, one statement per access -- never a call to the
        # underlying std wrapper, and never inlined into a consumer.
        self.assertRegex(rendered.source, r"\n\s*(\w+) = \w+\.a\n\s*\w+ = \1\.val\n")
        self.assertRegex(rendered.source, r"\n\s*\w+ = \w+\.x\n")
        self.assertNotIn("_getattr_wrapper", rendered.source)


if __name__ == "__main__":
    unittest.main()
