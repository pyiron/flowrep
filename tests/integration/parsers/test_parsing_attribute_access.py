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


@workflow_parser.workflow
def if_attribute_condition(comp: library.ComplexData, x: int):
    if library.my_condition(comp.val, 3):
        y = library.increment(x)
    else:
        y = library.decrement(x)
    return y


@workflow_parser.workflow
def if_bound_condition(comp: library.ComplexData, x: int):
    """The form the parser forced yesterday, and the form the compiler still emits."""
    val_0 = comp.val
    if library.my_condition(val_0, 3):  # noqa: SIM108
        y = library.increment(x)
    else:
        y = library.decrement(x)
    return y


@workflow_parser.workflow
def while_unlooped_attribute_condition(comp: library.ComplexData, seed: int):
    """`comp` is never reassigned, so hoisting the getattr is faithful to Python."""
    x = library.identity(seed)
    while library.my_condition(x, comp.val):
        x = library.loop_inc(x)
    return x


class TestAttributeInCondition(unittest.TestCase):
    def test_generated_port_is_named_for_the_attribute(self):
        recipe = if_attribute_condition.flowrep_recipe
        self.assertEqual(recipe.nodes["if_0"].inputs, ["val_0", "constant_0", "x"])

    def test_getattr_peer_feeds_the_generated_port(self):
        recipe = if_attribute_condition.flowrep_recipe
        self.assertEqual(
            recipe.edges[edge_models.TargetHandle(node="if_0", port="val_0")],
            edge_models.SourceHandle(node="getattr_val_0", port="attr"),
        )

    def test_identical_to_the_bound_form(self):
        """The headline invariant: the sugar is not merely equivalent, it is equal."""
        self.assertEqual(
            makers.dump_no_refs(makers.reference_free(if_attribute_condition)),
            makers.dump_no_refs(makers.reference_free(if_bound_condition)),
        )

    def test_executes_via_run_recipe(self):
        comp = library.ComplexData(val=7)
        result = wfms.run_recipe(if_attribute_condition.flowrep_recipe, comp=comp, x=1)
        self.assertEqual(
            result.output_ports["y"].value, if_attribute_condition(comp, 1)
        )

    def test_round_trips_through_source(self):
        free = makers.reference_free(if_attribute_condition)
        rendered = source._workflow2python(free)
        fn = rendered.build()
        self.assertEqual(
            makers.dump_no_refs(fn.flowrep_recipe), makers.dump_no_refs(free)
        )

    def test_compiles_back_to_the_bound_form(self):
        rendered = source._workflow2python(
            makers.reference_free(if_attribute_condition)
        )
        self.assertRegex(rendered.source, r"\n\s*val_0 = comp\.val\n")


class TestAttributeInWhileCondition(unittest.TestCase):
    def test_unlooped_root_is_allowed(self):
        recipe = while_unlooped_attribute_condition.flowrep_recipe
        self.assertEqual(recipe.nodes["while_0"].inputs, ["x", "val_0"])

    def test_executes_via_run_recipe(self):
        comp = library.ComplexData(val=7)
        result = wfms.run_recipe(
            while_unlooped_attribute_condition.flowrep_recipe, comp=comp, seed=1
        )
        self.assertEqual(
            result.output_ports["x"].value,
            while_unlooped_attribute_condition(comp, 1),
        )

    def test_round_trips_through_source(self):
        free = makers.reference_free(while_unlooped_attribute_condition)
        rendered = source._workflow2python(free)
        fn = rendered.build()
        self.assertEqual(
            makers.dump_no_refs(fn.flowrep_recipe), makers.dump_no_refs(free)
        )

    def test_looped_root_raises(self):
        """Python would re-read `x.val` each iteration; a hoisted getattr cannot."""

        def wf(comp: library.ComplexData):
            x = library.identity(comp)
            while library.my_condition(x, x.val):
                x = library.identity(x)
            return x

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(wf)
        message = str(ctx.exception)
        self.assertIn("x.val", message)
        self.assertIn("re-read", message)
        # Both honest fixes must be named -- only the author knows which they meant.
        self.assertIn("bind it outside the loop", message.lower())
        self.assertIn("condition function", message.lower())


@workflow_parser.workflow
def elif_two_attribute_conditions(a: library.ComplexData, b: library.ComplexData, x):
    if library.my_condition(a.val, 3):
        y = library.increment(x)
    elif library.my_condition(b.val, 4):
        y = library.decrement(x)
    else:
        y = library.negate(x)
    return y


class TestAttributeInElifChain(unittest.TestCase):
    def test_generated_ports_are_distinct_across_the_chain(self):
        recipe = elif_two_attribute_conditions.flowrep_recipe
        inputs = recipe.nodes["if_0"].inputs
        self.assertEqual(inputs[:4], ["val_0", "constant_0", "val_1", "constant_1"])

    def test_executes_via_run_recipe(self):
        a, b = library.ComplexData(val=7), library.ComplexData(val=2)
        result = wfms.run_recipe(
            elif_two_attribute_conditions.flowrep_recipe, a=a, b=b, x=1
        )
        self.assertEqual(
            result.output_ports["y"].value, elif_two_attribute_conditions(a, b, 1)
        )


if __name__ == "__main__":
    unittest.main()
