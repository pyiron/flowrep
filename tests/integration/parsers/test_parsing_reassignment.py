import unittest

from flowrep import edge_models, std
from flowrep.parsers import workflow_parser

from flowrep_static import library


def _parse(func):
    return workflow_parser.parse_workflow(func)


class TestAliasHappyPath(unittest.TestCase):
    def test_alias_of_input_matches_direct_return_but_relabels(self):
        def direct(x):
            return x

        def aliased(x):
            y = x
            return y

        direct_recipe = _parse(direct)
        aliased_recipe = _parse(aliased)

        self.assertEqual(direct_recipe.outputs, ["x"])
        self.assertEqual(aliased_recipe.outputs, ["y"])
        # The only difference is the output port label: both pass input `x`
        # straight through to their single output.
        self.assertEqual(
            list(direct_recipe.output_edges.values()),
            list(aliased_recipe.output_edges.values()),
        )
        self.assertEqual(aliased_recipe.nodes, {})

    def test_chained_alias_resolves_to_original_input(self):
        def macro(x):
            y = x
            z = y
            return z

        recipe = _parse(macro)
        self.assertEqual(recipe.outputs, ["z"])
        self.assertEqual(recipe.nodes, {})
        # Sole output edge sources from input `x`.
        (source,) = recipe.output_edges.values()
        self.assertEqual(source.port, "x")

    def test_alias_of_node_output(self):
        def macro(a, b):
            y = std.add(a, b)
            z = y
            return z

        recipe = _parse(macro)
        self.assertEqual(recipe.outputs, ["z"])
        (source,) = recipe.output_edges.values()
        self.assertEqual((source.node, source.port), ("add_0", "added"))

    def test_local_alias_then_consumed(self):
        def macro(a, b):
            y = a
            z = std.add(y, b)
            return z

        recipe = _parse(macro)
        self.assertEqual(recipe.outputs, ["z"])
        # my_add's `a` input is fed by workflow input `a` (through the alias).
        add_a = recipe.input_edges[edge_models.TargetHandle(node="add_0", port="a")]
        self.assertEqual(add_a.port, "a")

    def test_self_alias_is_noop(self):
        def macro(x):
            x = x
            return x

        recipe = _parse(macro)
        self.assertEqual(recipe.outputs, ["x"])
        self.assertEqual(recipe.nodes, {})

    def test_alias_with_explicit_output_label(self):
        def macro(x):
            y = x
            return y

        recipe = workflow_parser.parse_workflow(macro, "out")
        self.assertEqual(recipe.outputs, ["out"])


class TestAliasNonRegression(unittest.TestCase):
    def test_unpacking_rhs_raises(self):
        def macro(x):
            a, b = x
            return a, b

        with self.assertRaisesRegex(ValueError, "no unpacking raw symbols"):
            _parse(macro)

    def test_binop_rhs_still_raises(self):
        def macro(x):
            y = x + 1
            return y

        with self.assertRaises(ValueError):
            _parse(macro)

    def test_attribute_rhs_on_known_symbol_now_injects_getattr(self):
        """
        Attribute access rooted at a known workflow symbol is parsed as an
        injected ``std.getattr_`` node (see ``flowrep.parsers.attribute_parser``).
        """

        def macro(x):
            y = x.real
            return y

        recipe = _parse(macro)
        self.assertIn("get_attr_0", recipe.nodes)
        self.assertEqual(recipe.outputs, ["y"])

    def test_alias_to_undefined_symbol_raises(self):
        def macro(x):
            y = w  # noqa: F821
            return y

        with self.assertRaises(ValueError):
            _parse(macro)


class TestAliasAccumulatorInteractions(unittest.TestCase):
    def test_reassigning_accumulator_breaks_later_append(self):
        def macro(xs, shift=1):
            ys = []
            ys = shift  # de-registers the accumulator
            for x in xs:
                y = std.add(x, shift)  # noqa: F841
                ys.append(x)
            return ys

        with self.assertRaises(ValueError) as ctx:
            _parse(macro)
        self.assertIn("accumulator", str(ctx.exception).lower())

    def test_deregister_allows_independent_accumulator(self):
        def macro(xs, shift=1):
            ys = []
            ys = shift  # ys becomes a plain alias of `shift`
            zs = []
            for x in xs:
                z = std.add(x, shift)
                zs.append(z)
            return zs, ys

        recipe = _parse(macro)
        self.assertEqual(recipe.outputs, ["zs", "ys"])

    def test_alias_from_accumulator_is_rejected(self):
        def macro(xs):
            ys = []
            zs = ys  # aliasing an accumulator -> error
            for x in xs:
                zs.append(x)
            return zs

        with self.assertRaises(ValueError) as ctx:
            _parse(macro)
        self.assertIn("accumulator", str(ctx.exception).lower())

    def test_appending_to_input_alias_is_rejected(self):
        def macro(xs):
            ys = xs  # plain input alias, NOT an accumulator
            for x in xs:
                ys.append(x)
            return ys

        with self.assertRaises(ValueError):
            _parse(macro)


class TestAliasForLoopInteractions(unittest.TestCase):
    def test_alias_the_iterable(self):
        def macro(xs):
            zs = xs
            acc = []
            for x in zs:
                y = std.identity(x)
                acc.append(y)
            return acc

        recipe = _parse(macro)
        self.assertEqual(recipe.outputs, ["acc"])
        # The for-node's iterated input resolves back to workflow input `xs`.
        (for_label,) = [k for k, v in recipe.nodes.items() if k.startswith("for_each")]
        edge = recipe.edges | recipe.input_edges  # noqa: F841
        iterated = [
            src for tgt, src in recipe.input_edges.items() if tgt.node == for_label
        ]
        self.assertTrue(any(getattr(s, "port", None) == "xs" for s in iterated))

    def test_new_symbol_alias_inside_body(self):
        def macro(xs):
            acc = []
            for x in xs:
                y = x
                z = std.identity(y)
                acc.append(z)
            return acc

        recipe = _parse(macro)
        self.assertEqual(recipe.outputs, ["acc"])

    def test_alias_rebinding_enclosing_symbol_is_leaked_reassignment(self):
        def macro(xs, w):
            acc = []
            for x in xs:
                w = x  # noqa: F841
                # ^^ reassigns an enclosing symbol via alias -> leak
                z = std.identity(x)
                acc.append(z)
            return acc

        with self.assertRaises(ValueError) as ctx:
            _parse(macro)
        self.assertIn("reassign", str(ctx.exception).lower())


class TestAliasWhileLoopInteractions(unittest.TestCase):
    def test_alias_rebind_of_loop_carried_variable(self):
        def macro(y):
            while library.is_positive(y):
                z = library.decrement(y)
                y = z  # loop-carried via alias to a node output
            return y

        recipe = _parse(macro)
        (while_label,) = [k for k in recipe.nodes if k.startswith("while")]
        self.assertEqual(recipe.nodes[while_label].outputs, ["y"])

    def test_new_symbol_alias_not_a_loop_output(self):
        def macro(x):
            while library.is_positive(x):
                x = library.decrement(x)
                y = x  # noqa: F841
                # ^^ new alias, not reassigned -> not a loop output
            return x

        recipe = _parse(macro)
        (while_label,) = [k for k in recipe.nodes if k.startswith("while")]
        self.assertEqual(recipe.nodes[while_label].outputs, ["x"])

    def test_input_alias_as_loop_output_raises(self):
        def macro(x, seed):
            while library.is_positive(x):
                x = library.decrement(x)
                x = seed  # rebinds loop-carried x to a broadcast input
            return x

        with self.assertRaises(ValueError) as ctx:
            _parse(macro)
        self.assertIn("input", str(ctx.exception).lower())


class TestAliasBranchInteractions(unittest.TestCase):
    def test_local_alias_in_branch_consumed_downstream(self):
        def macro(a, x):
            if library.is_positive(a):
                w = x  # local alias, consumed -> never a branch output
                y = std.identity(w)
            else:
                y = library.decrement(x)
            return y

        recipe = _parse(macro)
        self.assertEqual(recipe.outputs, ["y"])

    def test_cross_branch_input_alias_output_raises(self):
        def macro(a, x, z):
            if library.is_positive(a):  # noqa: SIM108
                y = std.identity(x)  # node output
            else:
                y = z  # input alias -> would-be output disagrees across branches
            return y

        with self.assertRaises(ValueError) as ctx:
            _parse(macro)
        self.assertIn("input", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
