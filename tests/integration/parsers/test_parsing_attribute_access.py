import unittest

from flowrep import edge_models, wfms
from flowrep.compiler import source
from flowrep.parsers import for_parser, workflow_parser

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


@workflow_parser.workflow
def for_attribute_iterable(holder: library.Payload):
    ys = for_parser.accumulator()
    for n in holder.xs:
        y = library.increment(n)
        ys.append(y)
    return ys


@workflow_parser.workflow
def for_bound_iterable(holder: library.Payload):
    xs_0 = holder.xs
    ys = for_parser.accumulator()
    for n in xs_0:
        y = library.increment(n)
        ys.append(y)
    return ys


@workflow_parser.workflow
def for_zipped_attribute_iterables(a: library.Payload, b: library.Payload):
    zs = for_parser.accumulator()
    for m, n in zip(a.xs, b.xs, strict=True):
        z = library.my_add(m, n)
        zs.append(z)
    return zs


@workflow_parser.workflow
def for_nested_attribute_iterables(a: library.Payload, b: library.Payload):
    zs = for_parser.accumulator()
    for m in a.xs:
        for n in b.xs:
            z = library.my_add(m, n)
            zs.append(z)
    return zs


class TestAttributeAsForIterable(unittest.TestCase):
    def test_generated_port_is_named_for_the_attribute(self):
        recipe = for_attribute_iterable.flowrep_recipe
        self.assertEqual(recipe.nodes["for_each_0"].inputs, ["xs_0"])

    def test_getattr_peer_feeds_the_generated_port(self):
        recipe = for_attribute_iterable.flowrep_recipe
        self.assertEqual(
            recipe.edges[edge_models.TargetHandle(node="for_each_0", port="xs_0")],
            edge_models.SourceHandle(node="getattr_xs_0", port="attr"),
        )

    def test_identical_to_the_bound_form(self):
        self.assertEqual(
            makers.dump_no_refs(makers.reference_free(for_attribute_iterable)),
            makers.dump_no_refs(makers.reference_free(for_bound_iterable)),
        )

    def test_executes_via_run_recipe(self):
        holder = library.Payload(xs=[1, 2, 3])
        result = wfms.run_recipe(for_attribute_iterable.flowrep_recipe, holder=holder)
        self.assertEqual(
            result.output_ports["ys"].value, for_attribute_iterable(holder)
        )

    def test_round_trips_through_source(self):
        free = makers.reference_free(for_attribute_iterable)
        rendered = source._workflow2python(free)
        fn = rendered.build()
        self.assertEqual(
            makers.dump_no_refs(fn.flowrep_recipe), makers.dump_no_refs(free)
        )


class TestAttributesInZippedForIterables(unittest.TestCase):
    def test_generated_ports_are_distinct(self):
        recipe = for_zipped_attribute_iterables.flowrep_recipe
        self.assertEqual(recipe.nodes["for_each_0"].inputs, ["xs_0", "xs_1"])

    def test_executes_via_run_recipe(self):
        a, b = library.Payload(xs=[1, 2]), library.Payload(xs=[10, 20])
        result = wfms.run_recipe(
            for_zipped_attribute_iterables.flowrep_recipe, a=a, b=b
        )
        self.assertEqual(
            result.output_ports["zs"].value, for_zipped_attribute_iterables(a, b)
        )

    def test_round_trips_through_source(self):
        free = makers.reference_free(for_zipped_attribute_iterables)
        rendered = source._workflow2python(free)
        fn = rendered.build()
        self.assertEqual(
            makers.dump_no_refs(fn.flowrep_recipe), makers.dump_no_refs(free)
        )


class TestAttributesInNestedForIterables(unittest.TestCase):
    def test_generated_ports_are_distinct(self):
        recipe = for_nested_attribute_iterables.flowrep_recipe
        self.assertEqual(recipe.nodes["for_each_0"].inputs, ["xs_0", "xs_1"])

    def test_executes_via_run_recipe(self):
        a, b = library.Payload(xs=[1, 2]), library.Payload(xs=[10, 20])
        result = wfms.run_recipe(
            for_nested_attribute_iterables.flowrep_recipe, a=a, b=b
        )
        self.assertEqual(
            result.output_ports["zs"].value, for_nested_attribute_iterables(a, b)
        )

    def test_round_trips_through_source(self):
        free = makers.reference_free(for_nested_attribute_iterables)
        rendered = source._workflow2python(free)
        fn = rendered.build()
        self.assertEqual(
            makers.dump_no_refs(fn.flowrep_recipe), makers.dump_no_refs(free)
        )


@workflow_parser.workflow
def for_appending_attribute(comp: library.ComplexData, ns: list):
    xs = for_parser.accumulator()
    for n in ns:
        d = library.MyDataclass(comp, n)
        xs.append(d.x)
    return xs


@workflow_parser.workflow
def for_appending_bound_attribute(comp: library.ComplexData, ns: list):
    xs = for_parser.accumulator()
    for n in ns:
        d = library.MyDataclass(comp, n)
        x_0 = d.x
        xs.append(x_0)
    return xs


class TestAppendingAnAttribute(unittest.TestCase):
    def test_body_output_port_is_named_for_the_attribute(self):
        body = for_appending_attribute.flowrep_recipe.nodes["for_each_0"].body_node
        self.assertEqual(body.recipe.outputs, ["x_0"])

    def test_for_output_port_still_comes_from_the_accumulator(self):
        recipe = for_appending_attribute.flowrep_recipe
        self.assertEqual(recipe.nodes["for_each_0"].outputs, ["xs"])

    def test_getattr_lives_inside_the_body(self):
        """It must re-execute every iteration, exactly as the attribute access would."""
        body = for_appending_attribute.flowrep_recipe.nodes["for_each_0"].body_node
        self.assertIn("getattr_x_0", body.recipe.nodes)

    def test_identical_to_the_bound_form(self):
        self.assertEqual(
            makers.dump_no_refs(makers.reference_free(for_appending_attribute)),
            makers.dump_no_refs(makers.reference_free(for_appending_bound_attribute)),
        )

    def test_executes_via_run_recipe(self):
        comp = library.ComplexData(val=7)
        result = wfms.run_recipe(
            for_appending_attribute.flowrep_recipe, comp=comp, ns=[1, 2, 3]
        )
        self.assertEqual(
            result.output_ports["xs"].value, for_appending_attribute(comp, [1, 2, 3])
        )

    def test_round_trips_through_source(self):
        free = makers.reference_free(for_appending_attribute)
        rendered = source._workflow2python(free)
        fn = rendered.build()
        self.assertEqual(
            makers.dump_no_refs(fn.flowrep_recipe), makers.dump_no_refs(free)
        )


class TestPreservedRejections(unittest.TestCase):
    """The relaxation is bounded. These must keep raising."""

    def test_returned_attribute_raises(self):
        """A workflow's outputs are public IO; their names may not be generated."""

        def wf(comp: library.ComplexData):
            dc = library.MyDataclass(comp, 1)
            return dc.x

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("returned from a workflow", str(ctx.exception))

    def test_method_call_on_data_raises(self):
        def wf(comp: library.ComplexData):
            dc = library.MyDataclass(comp, 1)
            y = dc.method(comp)
            return y

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("method", str(ctx.exception).lower())

    def test_literal_iterable_raises(self):
        def wf(x):
            ys = for_parser.accumulator()
            for n in [1, 2, 3]:
                y = library.my_add(n, x)
                ys.append(y)
            return ys

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("symbol", str(ctx.exception).lower())

    def test_literal_append_raises(self):
        def wf(ns: list):
            ys = for_parser.accumulator()
            for n in ns:
                y = library.identity(n)  # noqa: F841
                ys.append(3)
            return ys

        with self.assertRaises(TypeError):
            workflow_parser.parse_workflow(wf)

    def test_attribute_of_accumulator_raises(self):
        """Accumulators are name-tracked, not graph data -- taking an attribute of
        one directly (while it is still active) must raise. A plain post-loop
        access such as ``ys.count`` is deliberately *not* this case: by then ``ys``
        is the finalised list output, an ordinary symbol like any other, and taking
        its attribute is exactly what the error message below tells the reader to
        do instead. To exercise the guard we shadow the accumulator's own name with
        the loop variable, so the loop body's ``ys.count`` refers to the still-live
        accumulator, not the eventual output."""

        def wf(items: list):
            ys = for_parser.accumulator()
            for ys in items:
                z = ys.count
                w = library.identity(z)
                ys.append(w)
            return ys

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("accumulator", str(ctx.exception).lower())


@workflow_parser.workflow
def try_attribute_in_branches(holder: library.Payload):
    try:
        y = library.divide(holder.num, holder.den)
    except ZeroDivisionError:
        y = library.identity(holder.num)
    return y


class TestAttributeInsideTryBranches(unittest.TestCase):
    def test_try_node_takes_the_root_symbol_not_a_generated_port(self):
        recipe = try_attribute_in_branches.flowrep_recipe
        self.assertEqual(recipe.nodes["try_0"].inputs, ["holder"])

    def test_getattr_lives_inside_the_branch(self):
        try_node = try_attribute_in_branches.flowrep_recipe.nodes["try_0"]
        self.assertIn("getattr_num_0", try_node.try_node.recipe.nodes)

    def test_executes_both_branches(self):
        ok = library.Payload(num=6, den=2)
        boom = library.Payload(num=6, den=0)
        for holder in (ok, boom):
            result = wfms.run_recipe(
                try_attribute_in_branches.flowrep_recipe, holder=holder
            )
            self.assertEqual(
                result.output_ports["y"].value, try_attribute_in_branches(holder)
            )

    @unittest.expectedFailure
    def test_round_trips_through_source(self):
        """Known gap: a getattr *inside a branch body* leaks into the branch outputs.

        The compiler expands every getattr into its own ``name = obj.attr`` statement,
        and a branch body's outputs are every locally-registered symbol -- not just a
        returned one, which is what keeps this invisible at the top level of a
        ``@workflow``. So re-parsing promotes the compiler's own ``getattr_num`` to a
        branch output that the original recipe never had.

        Predates this work and is independent of it: attribute access as an ordinary
        call argument has always been legal, and a plain ``if`` branch with no relaxed
        condition, for-header, or append reproduces it identically. Left executable
        rather than deleted, so it flips to an unexpected success when fixed.
        """
        free = makers.reference_free(try_attribute_in_branches)
        rendered = source._workflow2python(free)
        fn = rendered.build()
        self.assertEqual(
            makers.dump_no_refs(fn.flowrep_recipe), makers.dump_no_refs(free)
        )


if __name__ == "__main__":
    unittest.main()
