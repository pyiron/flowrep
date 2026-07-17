import unittest

from flowrep import edge_models, std, wfms
from flowrep.compiler import source
from flowrep.parsers import workflow_parser

from flowrep_static import library, makers


@workflow_parser.workflow
def item_symbol_key(
    input_dict: dict[library.CustomKey, int], input_obj: library.CustomKey
):
    d = std.identity(input_dict)
    i = std.identity(input_obj)
    accessed = d[i]
    return accessed


@workflow_parser.workflow
def item_symbol_key_std_form(
    input_dict: dict[library.CustomKey, int], input_obj: library.CustomKey
):
    """The explicit call the sugar must be indistinguishable from."""
    d = std.identity(input_dict)
    i = std.identity(input_obj)
    accessed = std.getitem(d, i)
    return accessed


class TestComplexKeyFromAnotherStep(unittest.TestCase):
    def test_parses_expected_graph(self):
        recipe = item_symbol_key.flowrep_recipe
        self.assertEqual(set(recipe.nodes), {"identity_0", "identity_1", "getitem_0"})
        self.assertEqual(
            recipe.edges[edge_models.TargetHandle(node="getitem_0", port="a")],
            edge_models.SourceHandle(node="identity_0", port="x"),
        )
        self.assertEqual(
            recipe.edges[edge_models.TargetHandle(node="getitem_0", port="b")],
            edge_models.SourceHandle(node="identity_1", port="x"),
        )

    def test_identical_to_the_std_call_form(self):
        """The headline invariant: `d[i]` is not merely equivalent to
        `std.getitem(d, i)` -- it is the same recipe. That identity is what lets the
        compiler round-trip item access without knowing item access exists."""
        self.assertEqual(
            makers.dump_no_refs(makers.reference_free(item_symbol_key)),
            makers.dump_no_refs(makers.reference_free(item_symbol_key_std_form)),
        )

    def test_executes_via_run_recipe(self):
        key = library.CustomKey("a")
        data = {key: 7}
        result = wfms.run_recipe(
            item_symbol_key.flowrep_recipe, input_dict=data, input_obj=key
        )
        self.assertEqual(result.output_ports["accessed"].value, 7)
        self.assertEqual(item_symbol_key(data, key), 7)

    def test_round_trips_through_source(self):
        free = makers.reference_free(item_symbol_key)
        rendered = source._workflow2python(free)
        fn = rendered.build()
        key = library.CustomKey("a")
        self.assertEqual(fn({key: 7}, key), item_symbol_key({key: 7}, key))
        self.assertEqual(
            makers.dump_no_refs(fn.flowrep_recipe), makers.dump_no_refs(free)
        )


@workflow_parser.workflow
def item_from_inputs(
    input_dict: dict[library.CustomKey, int], input_obj: library.CustomKey
):
    accessed = input_dict[input_obj]
    return accessed


class TestComplexKeyFromParentInput(unittest.TestCase):
    def test_both_ports_are_input_edges(self):
        recipe = item_from_inputs.flowrep_recipe
        self.assertEqual(set(recipe.nodes), {"getitem_0"})
        self.assertEqual(
            recipe.input_edges[edge_models.TargetHandle(node="getitem_0", port="a")],
            edge_models.InputSource(port="input_dict"),
        )
        self.assertEqual(
            recipe.input_edges[edge_models.TargetHandle(node="getitem_0", port="b")],
            edge_models.InputSource(port="input_obj"),
        )

    def test_executes_via_run_recipe(self):
        key = library.CustomKey("a")
        result = wfms.run_recipe(
            item_from_inputs.flowrep_recipe, input_dict={key: 7}, input_obj=key
        )
        self.assertEqual(result.output_ports["accessed"].value, 7)

    def test_round_trips_through_source(self):
        free = makers.reference_free(item_from_inputs)
        rendered = source._workflow2python(free)
        fn = rendered.build()
        self.assertEqual(
            makers.dump_no_refs(fn.flowrep_recipe), makers.dump_no_refs(free)
        )


@workflow_parser.workflow
def item_constant_keys(xs: list, d: dict):
    first = xs[0]
    named = d["mass"]
    total = std.add(first, named)
    return total


class TestConstantKeys(unittest.TestCase):
    def test_each_key_gets_a_constant_peer(self):
        recipe = item_constant_keys.flowrep_recipe
        self.assertEqual(
            list(recipe.nodes),
            ["getitem_0", "constant_0", "getitem_1", "constant_1", "add_0"],
        )
        self.assertEqual(recipe.nodes["constant_0"].constant, 0)
        self.assertEqual(recipe.nodes["constant_1"].constant, "mass")

    def test_executes_via_run_recipe(self):
        result = wfms.run_recipe(
            item_constant_keys.flowrep_recipe, xs=[1, 2], d={"mass": 10}
        )
        self.assertEqual(result.output_ports["total"].value, 11)
        self.assertEqual(item_constant_keys([1, 2], {"mass": 10}), 11)

    def test_round_trips_through_source(self):
        free = makers.reference_free(item_constant_keys)
        rendered = source._workflow2python(free)
        fn = rendered.build()
        self.assertEqual(
            makers.dump_no_refs(fn.flowrep_recipe), makers.dump_no_refs(free)
        )


@workflow_parser.workflow
def mixed_chain(holder: library.Nested):
    value = holder.sub["item"]["subitem"].val
    return value


class TestMixedChain(unittest.TestCase):
    def test_one_node_per_link_root_outward(self):
        recipe = mixed_chain.flowrep_recipe
        self.assertEqual(
            list(recipe.nodes),
            [
                "getattr_sub_0",
                "constant_0",
                "getitem_0",
                "constant_1",
                "getitem_1",
                "constant_2",
                "getattr_val_0",
                "constant_3",
            ],
        )

    def test_links_are_chained_in_order(self):
        recipe = mixed_chain.flowrep_recipe
        self.assertEqual(
            recipe.edges[edge_models.TargetHandle(node="getitem_0", port="a")],
            edge_models.SourceHandle(node="getattr_sub_0", port="attr"),
        )
        self.assertEqual(
            recipe.edges[edge_models.TargetHandle(node="getitem_1", port="a")],
            edge_models.SourceHandle(node="getitem_0", port="item"),
        )
        self.assertEqual(
            recipe.edges[edge_models.TargetHandle(node="getattr_val_0", port="obj")],
            edge_models.SourceHandle(node="getitem_1", port="item"),
        )

    def test_executes_via_run_recipe(self):
        holder = library.Nested()
        result = wfms.run_recipe(mixed_chain.flowrep_recipe, holder=holder)
        self.assertEqual(result.output_ports["value"].value, 7)
        self.assertEqual(mixed_chain(holder), 7)

    def test_round_trips_through_source(self):
        """Mixed sugar: the compiler emits attribute syntax for the getattr links and
        the plain call form for the getitem links, and both re-parse to the originals.
        """
        free = makers.reference_free(mixed_chain)
        rendered = source._workflow2python(free)
        fn = rendered.build()
        self.assertEqual(fn(library.Nested()), mixed_chain(library.Nested()))
        self.assertEqual(
            makers.dump_no_refs(fn.flowrep_recipe), makers.dump_no_refs(free)
        )


class TestRejections(unittest.TestCase):
    def test_literal_slice_raises(self):
        def wf(xs: list):
            sliced = xs[1:2]
            return sliced

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(wf)
        message = str(ctx.exception)
        self.assertIn("slice", message.lower())
        self.assertIn("std.slice", message)

    def test_chain_in_key_position_raises_and_names_the_bind(self):
        """Out of scope by design; the message must hand the reader the workaround."""

        def wf(d: dict, holder: library.Nested):
            v = d[holder.sub]
            return v

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(wf)
        message = str(ctx.exception)
        self.assertIn("k = holder.sub", message)
        self.assertIn("d[k]", message)

    def test_returned_item_raises(self):
        """A workflow's outputs are public IO; their names may not be generated."""

        def wf(d: dict, k):
            return d[k]

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("returned from a workflow", str(ctx.exception))

    def test_call_on_item_raises(self):
        def wf(d: dict, k, x):
            y = d[k](x)
            return y

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("workflow data", str(ctx.exception))

    def test_unknown_key_symbol_raises(self):
        def wf(d: dict):
            v = d[nope]  # noqa: F821
            return v

        with self.assertRaises(KeyError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("not in scope", str(ctx.exception))


class TestOnlyLoadContextIsParsed(unittest.TestCase):
    """Item access is parsed only where a value is read. Every other subscript context
    is refused upstream, before chain parsing ever sees the node -- which is why the
    chain code carries no `ctx` guard of its own."""

    def test_store_raises(self):
        def wf(dd: dict, v):
            d = std.identity(dd)
            d["k"] = v
            return d

        with self.assertRaises(TypeError):
            workflow_parser.parse_workflow(wf)

    def test_augmented_store_raises(self):
        def wf(dd: dict):
            d = std.identity(dd)
            d["k"] += 1
            return d

        with self.assertRaises(TypeError):
            workflow_parser.parse_workflow(wf)

    def test_delete_raises(self):
        def wf(dd: dict):
            d = std.identity(dd)
            del d["k"]
            return d

        with self.assertRaises(TypeError):
            workflow_parser.parse_workflow(wf)


@workflow_parser.workflow
def if_item_condition(d: dict, x: int):
    if library.my_condition(d["threshold"], 3):  # noqa: SIM108
        y = library.increment(x)
    else:
        y = library.decrement(x)
    return y


@workflow_parser.workflow
def if_bound_item_condition(d: dict, x: int):
    """The form the compiler emits: the generated port name, spelled out."""
    item_0 = d["threshold"]
    if library.my_condition(item_0, 3):  # noqa: SIM108
        y = library.increment(x)
    else:
        y = library.decrement(x)
    return y


class TestItemInCondition(unittest.TestCase):
    def test_generated_port_uses_the_item_base(self):
        recipe = if_item_condition.flowrep_recipe
        self.assertEqual(recipe.nodes["if_0"].inputs, ["item_0", "constant_0", "x"])

    def test_getitem_peer_feeds_the_generated_port(self):
        recipe = if_item_condition.flowrep_recipe
        self.assertEqual(
            recipe.edges[edge_models.TargetHandle(node="if_0", port="item_0")],
            edge_models.SourceHandle(node="getitem_0", port="item"),
        )

    def test_identical_to_the_bound_form(self):
        self.assertEqual(
            makers.dump_no_refs(makers.reference_free(if_item_condition)),
            makers.dump_no_refs(makers.reference_free(if_bound_item_condition)),
        )

    def test_executes_via_run_recipe(self):
        d = {"threshold": 1}
        result = wfms.run_recipe(if_item_condition.flowrep_recipe, d=d, x=1)
        self.assertEqual(result.output_ports["y"].value, if_item_condition(d, 1))

    def test_round_trips_through_source(self):
        free = makers.reference_free(if_item_condition)
        rendered = source._workflow2python(free)
        fn = rendered.build()
        self.assertEqual(
            makers.dump_no_refs(fn.flowrep_recipe), makers.dump_no_refs(free)
        )


@workflow_parser.workflow
def for_item_iterable(data: dict):
    ys = []
    for n in data["xs"]:
        y = library.increment(n)
        ys.append(y)
    return ys


class TestItemAsForIterable(unittest.TestCase):
    def test_generated_port_uses_the_item_base(self):
        recipe = for_item_iterable.flowrep_recipe
        self.assertEqual(recipe.nodes["for_each_0"].inputs, ["item_0"])

    def test_getitem_peer_feeds_the_generated_port(self):
        recipe = for_item_iterable.flowrep_recipe
        self.assertEqual(
            recipe.edges[edge_models.TargetHandle(node="for_each_0", port="item_0")],
            edge_models.SourceHandle(node="getitem_0", port="item"),
        )

    def test_executes_via_run_recipe(self):
        data = {"xs": [1, 2, 3]}
        result = wfms.run_recipe(for_item_iterable.flowrep_recipe, data=data)
        self.assertEqual(result.output_ports["ys"].value, for_item_iterable(data))

    def test_round_trips_through_source(self):
        free = makers.reference_free(for_item_iterable)
        rendered = source._workflow2python(free)
        fn = rendered.build()
        self.assertEqual(
            makers.dump_no_refs(fn.flowrep_recipe), makers.dump_no_refs(free)
        )


@workflow_parser.workflow
def for_appending_item(lookup: dict, ns: list):
    xs = []
    for n in ns:
        row = std.identity(lookup)
        xs.append(row[n])
    return xs


class TestAppendingAnItem(unittest.TestCase):
    def test_body_output_port_uses_the_item_base(self):
        body = for_appending_item.flowrep_recipe.nodes["for_each_0"].body_node
        self.assertEqual(body.recipe.outputs, ["item_0"])

    def test_getitem_lives_inside_the_body(self):
        """It must re-execute every iteration, exactly as the item access would."""
        body = for_appending_item.flowrep_recipe.nodes["for_each_0"].body_node
        self.assertIn("getitem_0", body.recipe.nodes)

    def test_executes_via_run_recipe(self):
        lookup = {1: "a", 2: "b"}
        result = wfms.run_recipe(
            for_appending_item.flowrep_recipe, lookup=lookup, ns=[1, 2]
        )
        self.assertEqual(
            result.output_ports["xs"].value, for_appending_item(lookup, [1, 2])
        )

    def test_round_trips_through_source(self):
        free = makers.reference_free(for_appending_item)
        rendered = source._workflow2python(free)
        fn = rendered.build()
        self.assertEqual(
            makers.dump_no_refs(fn.flowrep_recipe), makers.dump_no_refs(free)
        )


@workflow_parser.workflow
def while_unlooped_item_condition(d: dict, k: int, seed: int):
    """Neither `d` nor `k` is reassigned, so hoisting the getitem is faithful."""
    x = std.identity(seed)
    while library.my_condition(x, d[k]):
        x = library.loop_inc(x)
    return x


class TestItemInWhileCondition(unittest.TestCase):
    def test_unlooped_dependencies_are_allowed(self):
        recipe = while_unlooped_item_condition.flowrep_recipe
        self.assertEqual(recipe.nodes["while_0"].inputs, ["x", "item_0"])

    def test_executes_via_run_recipe(self):
        d = {1: 3}
        result = wfms.run_recipe(
            while_unlooped_item_condition.flowrep_recipe, d=d, k=1, seed=1
        )
        self.assertEqual(
            result.output_ports["x"].value, while_unlooped_item_condition(d, 1, 1)
        )

    def test_looped_key_symbol_raises(self):
        """The item-only half of the guard: `d` is untouched, but `x` -- the *key* --
        is reassigned every iteration, so Python would re-read `d[x]` and a hoisted
        getitem would not."""

        def wf(d: dict, seed: int):
            x = std.identity(seed)
            while library.my_condition(x, d[x]):
                x = library.loop_inc(x)
            return x

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(wf)
        message = str(ctx.exception)
        self.assertIn("d[x]", message)
        self.assertIn("'x'", message)
        self.assertIn("re-read", message)

    def test_looped_root_raises(self):
        def wf(dd: dict, k):
            d = std.identity(dd)
            while library.my_condition(d, d[k]):
                d = std.identity(d)
            return d

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("'d'", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
