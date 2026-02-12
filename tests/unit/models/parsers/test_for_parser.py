"""
Unit tests for flowrep.models.parsers.for_parser.

Covers the pure-AST helpers (_is_zip_call, _parse_single_for_header,
parse_for_iterations) directly, and the stateful ForParser / walk_ast_for
logic indirectly through workflow_parser.parse_workflow.
"""

import ast
import textwrap
import unittest

from flowrep.models import edge_models
from flowrep.models.nodes import atomic_model, for_model, workflow_model
from flowrep.models.parsers import atomic_parser, for_parser, workflow_parser

# ---------------------------------------------------------------------------
# Helper callables reachable from the test-module scope so that
# object_scope.get_scope(wf) can resolve them during parsing.
# ---------------------------------------------------------------------------


def identity(x):
    return x


@atomic_parser.atomic(unpack_mode=atomic_model.UnpackMode.NONE)
def pair(a, b):
    return a, b


@atomic_parser.atomic(unpack_mode=atomic_model.UnpackMode.TUPLE)
def split(a, b):
    return a, b


def my_range(n):
    return list(range(n))


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _parse_for_stmt(code: str) -> ast.For:
    """Return the first ast.For found in *code*."""
    tree = ast.parse(textwrap.dedent(code))
    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            return node
    raise ValueError("No ast.For found")


# ===================================================================
# _is_zip_call
# ===================================================================


class TestIsZipCall(unittest.TestCase):
    def test_plain_zip(self):
        call = ast.parse("zip(a, b)", mode="eval").body
        self.assertTrue(for_parser._is_zip_call(call))

    def test_other_builtin(self):
        call = ast.parse("range(10)", mode="eval").body
        self.assertFalse(for_parser._is_zip_call(call))

    def test_attribute_call_not_zip(self):
        call = ast.parse("itertools.zip(a)", mode="eval").body
        self.assertFalse(for_parser._is_zip_call(call))


# ===================================================================
# _parse_single_for_header
# ===================================================================


class TestParseSingleForHeader(unittest.TestCase):
    """Direct tests for every branch in _parse_single_for_header."""

    # --- happy paths ---

    def test_simple_iteration(self):
        stmt = _parse_for_stmt("for x in xs: pass")
        is_zip, pairs = for_parser._parse_single_for_header(stmt)
        self.assertFalse(is_zip)
        self.assertEqual(pairs, [("x", "xs")])

    def test_zip_two_elements(self):
        stmt = _parse_for_stmt("for a, b in zip(xs, ys): pass")
        is_zip, pairs = for_parser._parse_single_for_header(stmt)
        self.assertTrue(is_zip)
        self.assertEqual(pairs, [("a", "xs"), ("b", "ys")])

    def test_zip_three_elements(self):
        stmt = _parse_for_stmt("for a, b, c in zip(xs, ys, zs): pass")
        is_zip, pairs = for_parser._parse_single_for_header(stmt)
        self.assertTrue(is_zip)
        self.assertEqual(pairs, [("a", "xs"), ("b", "ys"), ("c", "zs")])

    # --- error paths ---

    def test_zip_without_tuple_unpacking_raises(self):
        stmt = _parse_for_stmt("for x in zip(xs, ys): pass")
        with self.assertRaises(ValueError) as ctx:
            for_parser._parse_single_for_header(stmt)
        self.assertIn("tuple unpacking", str(ctx.exception))

    def test_zip_non_name_target_raises(self):
        stmt = _parse_for_stmt("for a, b in zip(xs, ys): pass")
        # Patch one target element to an Attribute so it isn't an ast.Name
        stmt.target.elts[1] = ast.Attribute(value=ast.Name(id="obj"), attr="field")
        with self.assertRaises(ValueError) as ctx:
            for_parser._parse_single_for_header(stmt)
        self.assertIn("simple names", str(ctx.exception))

    def test_zip_non_symbol_arg_raises(self):
        stmt = _parse_for_stmt("for a, b in zip(xs, foo()): pass")
        with self.assertRaises(ValueError) as ctx:
            for_parser._parse_single_for_header(stmt)
        self.assertIn("simple symbols", str(ctx.exception))

    def test_zip_var_arg_count_mismatch_raises(self):
        stmt = _parse_for_stmt("for a, b, c in zip(xs, ys): pass")
        with self.assertRaises(ValueError) as ctx:
            for_parser._parse_single_for_header(stmt)
        self.assertIn("variable count", str(ctx.exception))

    def test_iteration_over_expression_raises(self):
        stmt = _parse_for_stmt("for x in range(10): pass")
        with self.assertRaises(ValueError) as ctx:
            for_parser._parse_single_for_header(stmt)
        self.assertIn("symbol", str(ctx.exception).lower())

    def test_tuple_unpacking_without_zip_raises(self):
        stmt = _parse_for_stmt("for a, b in items: pass")
        with self.assertRaises(ValueError) as ctx:
            for_parser._parse_single_for_header(stmt)
        self.assertIn("requires zip()", str(ctx.exception))

    def test_unsupported_target_type_raises(self):
        stmt = _parse_for_stmt("for x in items: pass")
        stmt.target = ast.Subscript(
            value=ast.Name(id="arr"),
            slice=ast.Constant(value=0),
        )
        with self.assertRaises(ValueError) as ctx:
            for_parser._parse_single_for_header(stmt)
        self.assertIn("Unsupported", str(ctx.exception))


# ===================================================================
# parse_for_iterations
# ===================================================================


class TestParseForIterations(unittest.TestCase):
    """Tests for the iteration-header unwrapping logic."""

    def test_single_simple_iteration(self):
        stmt = _parse_for_stmt("for x in xs:\n  pass")
        nested, zipped, body = for_parser.parse_for_iterations(stmt)
        self.assertEqual(nested, [("x", "xs")])
        self.assertEqual(zipped, [])

    def test_single_zip_iteration(self):
        stmt = _parse_for_stmt("for a, b in zip(xs, ys):\n  pass")
        nested, zipped, body = for_parser.parse_for_iterations(stmt)
        self.assertEqual(nested, [])
        self.assertEqual(zipped, [("a", "xs"), ("b", "ys")])

    def test_two_nested_simple_iterations(self):
        stmt = _parse_for_stmt("for x in xs:\n  for y in ys:\n    pass")
        nested, zipped, body = for_parser.parse_for_iterations(stmt)
        self.assertEqual(nested, [("x", "xs"), ("y", "ys")])
        self.assertEqual(zipped, [])

    def test_nested_then_zip(self):
        code = "for x in xs:\n  for a, b in zip(as_, bs):\n    pass"
        stmt = _parse_for_stmt(code)
        nested, zipped, body = for_parser.parse_for_iterations(stmt)
        self.assertEqual(nested, [("x", "xs")])
        self.assertEqual(zipped, [("a", "as_"), ("b", "bs")])

    def test_zipped_then_nested(self):
        code = "for a, b in zip(as_, bs):\n    for x in xs:\n      pass"
        stmt = _parse_for_stmt(code)
        nested, zipped, body = for_parser.parse_for_iterations(stmt)
        self.assertEqual(nested, [("x", "xs")])
        self.assertEqual(zipped, [("a", "as_"), ("b", "bs")])

    def test_body_tree_is_innermost_for(self):
        code = "for x in xs:\n  for y in ys:\n    z = 1"
        stmt = _parse_for_stmt(code)
        _, _, body = for_parser.parse_for_iterations(stmt)
        self.assertIsInstance(body, ast.For)
        # The innermost for's body should contain the assignment
        self.assertIsInstance(body.body[0], ast.Assign)

    def test_stops_nesting_when_body_starts_with_non_for(self):
        """If body[0] is not a For, don't descend even if a For follows."""
        code = "for x in xs:\n  y = 1\n  for z in zs:\n    pass"
        stmt = _parse_for_stmt(code)
        nested, zipped, body = for_parser.parse_for_iterations(stmt)
        self.assertEqual(nested, [("x", "xs")])
        self.assertEqual(zipped, [])
        # body is the outer for – its body contains the Assign then another For
        self.assertIsInstance(body.body[0], ast.Assign)


# ===================================================================
# walk_ast_for – error paths (tested via parse_workflow)
# ===================================================================


class TestWalkAstForErrors(unittest.TestCase):
    """
    walk_ast_for is always reached through WorkflowParser.handle_for,
    so we test error paths by defining small invalid workflow functions.
    """

    def test_if_in_for_body_raises(self):
        def wf(xs):
            results = []
            for x in xs:
                if True:
                    pass
                results.append(x)
            return results

        with self.assertRaises(NotImplementedError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("If", str(ctx.exception))

    def test_try_in_for_body_raises(self):
        def wf(xs):
            results = []
            for x in xs:
                try:  # noqa: SIM105
                    pass
                except Exception:
                    pass
                results.append(x)
            return results

        with self.assertRaises(NotImplementedError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("Try", str(ctx.exception))

    def test_unrecognised_body_stmt_raises_type_error(self):
        """ast.Return inside a for body is not handled → TypeError."""

        def wf(xs):
            results = []
            for x in xs:
                y = identity(x)
                results.append(y)
                return results
            return results

        with self.assertRaises(TypeError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("ast found", str(ctx.exception))

    def test_no_accumulator_used_raises(self):
        def wf(xs):
            results = []
            for x in xs:
                y = identity(x)  # noqa: F841
            return results

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("at least one accumulator", str(ctx.exception))

    def test_duplicate_accumulator_appends_raises(self):
        def wf(xs):
            results = []
            for x in xs:
                y = identity(x)
                results.append(y)
                results.append(y)
            return results

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("at most once", str(ctx.exception))

    def test_assigning_non_empty_list_raises(self):
        def wf(xs):
            results = [0]
            for x in xs:
                y = identity(x)
                results.append(y)
            return results

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("or empty list", str(ctx.exception))


class TestWalkAstForHeaderErrorsViaParseWorkflow(unittest.TestCase):
    """Header-level errors that surface through parse_workflow."""

    def test_iteration_over_call_raises(self):
        def wf():
            results = []
            for x in range(10):
                y = identity(x)
                results.append(y)
            return results

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("symbol", str(ctx.exception).lower())

    def test_tuple_unpacking_without_zip_raises(self):
        def wf(items):
            results_a = []
            results_b = []
            for a, b in items:
                y = identity(a)
                z = identity(b)
                results_a.append(y)
                results_b.append(z)
            return results_a, results_b

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("requires zip()", str(ctx.exception))

    def test_zip_with_expression_arg_raises(self):
        def wf(xs):
            results_a = []
            results_b = []
            for a, b in zip(xs, list(), strict=True):
                y = identity(a)
                z = identity(b)
                results_a.append(y)
                results_b.append(z)
            return results_a, results_b

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("simple symbols", str(ctx.exception))


# ===================================================================
# ForParser – edge wiring (tested via parse_workflow)
# ===================================================================


class TestForParserEdgeWiring(unittest.TestCase):
    """Verify the input/output/transfer edge construction."""

    # --- helpers ---

    @staticmethod
    def _get_for_node(func) -> for_model.ForNode:
        wf = workflow_parser.parse_workflow(func)
        for_nodes = [n for n in wf.nodes.values() if isinstance(n, for_model.ForNode)]
        assert len(for_nodes) == 1, f"Expected 1 ForNode, got {len(for_nodes)}"
        return for_nodes[0]

    # --- output edges from body computation ---

    def test_output_edge_from_body(self):
        def wf(xs):
            results = []
            for x in xs:
                y = identity(x)
                results.append(y)
            return results

        fn = self._get_for_node(wf)
        target = edge_models.OutputTarget(port="results")
        self.assertIn(target, fn.output_edges)
        self.assertEqual(fn.output_edges[target].node, "body")
        self.assertEqual(fn.output_edges[target].port, "y")

    # --- transfer edges (forwarding an iteration variable) ---

    def test_forwarded_iteration_var_is_input_source_in_output_edges(self):
        def wf(xs):
            collected_xs = []
            results = []
            for x in xs:
                y = identity(x)
                collected_xs.append(x)
                results.append(y)
            return collected_xs, results

        fn = self._get_for_node(wf)

        # Forwarded iteration variable → InputSource in output_edges
        fwd_target = edge_models.OutputTarget(port="collected_xs")
        self.assertIn(fwd_target, fn.output_edges)
        self.assertIsInstance(fn.output_edges[fwd_target], edge_models.InputSource)

        # Computed body result → SourceHandle in output_edges
        body_target = edge_models.OutputTarget(port="results")
        self.assertIn(body_target, fn.output_edges)
        self.assertIsInstance(fn.output_edges[body_target], edge_models.SourceHandle)

    def test_forwarded_iteration_var_classified_as_transferred(self):
        """Forwarded iterated input should appear in transferred_outputs."""

        def wf(xs):
            collected_xs = []
            results = []
            for x in xs:
                y = identity(x)
                collected_xs.append(x)
                results.append(y)
            return collected_xs, results

        fn = self._get_for_node(wf)
        self.assertEqual(len(fn.transferred_outputs), 1)
        self.assertIn(
            edge_models.OutputTarget(port="collected_xs"), fn.transferred_outputs
        )

    # --- broadcast vs scattered inputs ---

    def test_broadcast_symbol_appears_as_input(self):
        """A symbol used in the body but not iterated is broadcast."""

        def wf(xs, c):
            results = []
            for x in xs:
                y = pair(x, c)
                results.append(y)
            return results

        fn = self._get_for_node(wf)
        self.assertIn("c", fn.inputs)
        # c is broadcast → input edge maps directly
        target = edge_models.TargetHandle(node="body", port="c")
        self.assertIn(target, fn.input_edges)
        self.assertEqual(fn.input_edges[target].port, "c")

    def test_scattered_nested_input(self):
        def wf(xs):
            results = []
            for x in xs:
                y = identity(x)
                results.append(y)
            return results

        fn = self._get_for_node(wf)
        self.assertIn("xs", fn.inputs)
        self.assertEqual(fn.nested_ports, ["x"])
        target = edge_models.TargetHandle(node="body", port="x")
        self.assertIn(target, fn.input_edges)
        self.assertEqual(fn.input_edges[target].port, "xs")

    def test_scattered_zipped_inputs(self):
        def wf(xs, ys):
            results = []
            for x, y in zip(xs, ys, strict=True):
                z = pair(x, y)
                results.append(z)
            return results

        fn = self._get_for_node(wf)
        self.assertEqual(sorted(fn.zipped_ports), ["x", "y"])
        self.assertIn("xs", fn.inputs)
        self.assertIn("ys", fn.inputs)

    def test_mixed_nested_and_zipped(self):
        """Both nested and zipped iteration, all variables consumed."""

        def wf(xs, ys, zs):
            results = []
            for x in xs:
                for y, z in zip(ys, zs, strict=True):
                    t = pair(y, z)
                    # x must be consumed or we get an unused-iterator error
                    u = identity(x)  # noqa: F841
                    # But there is no necessity for everything in the body to be
                    # captured and passed back out to the for-node output
                    results.append(t)
            return results

        fn = self._get_for_node(wf)
        self.assertEqual(fn.nested_ports, ["x"])
        self.assertEqual(sorted(fn.zipped_ports), ["y", "z"])

    def test_unused_iterator_raises(self):
        """An iteration variable never consumed in the body is an error."""

        def wf(xs, ys, zs):
            results = []
            for x in xs:  # noqa: B007
                for y, z in zip(ys, zs, strict=True):
                    t = pair(y, z)
                    results.append(t)
            return results

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("never used", str(ctx.exception))
        self.assertIn("x", str(ctx.exception))


# ===================================================================
# ForParser – structural properties
# ===================================================================


class TestForParserStructure(unittest.TestCase):
    def _parse(self, func) -> workflow_model.WorkflowNode:
        return workflow_parser.parse_workflow(func)

    def test_for_node_registered_in_parent(self):
        def wf(xs):
            results = []
            for x in xs:
                y = identity(x)
                results.append(y)
            return results

        node = self._parse(wf)
        self.assertIn("for_0", node.nodes)
        self.assertIsInstance(node.nodes["for_0"], for_model.ForNode)

    def test_body_node_label_is_body(self):
        def wf(xs):
            results = []
            for x in xs:
                y = identity(x)
                results.append(y)
            return results

        fn = self._parse(wf).nodes["for_0"]
        self.assertEqual(fn.body_node.label, "body")

    def test_body_node_is_workflow_node(self):
        def wf(xs):
            results = []
            for x in xs:
                y = identity(x)
                results.append(y)
            return results

        fn = self._parse(wf).nodes["for_0"]
        self.assertIsInstance(fn.body_node.node, workflow_model.WorkflowNode)

    def test_multiple_accumulators(self):
        def wf(xs):
            as_ = []
            bs = []
            for x in xs:
                a, b = split(x, x)
                as_.append(a)
                bs.append(b)
            return as_, bs

        fn = self._parse(wf).nodes["for_0"]
        self.assertEqual(sorted(fn.outputs), ["as_", "bs"])

    def test_for_output_consumed_by_downstream_node(self):
        """Symbols produced by a for-node should be usable downstream."""

        def wf(xs):
            results = []
            for x in xs:
                y = identity(x)
                results.append(y)
            z = identity(results)
            return z

        node = self._parse(wf)
        target = edge_models.TargetHandle(node="identity_0", port="x")
        self.assertIn(target, node.edges)
        self.assertEqual(node.edges[target].node, "for_0")
        self.assertEqual(node.edges[target].port, "results")

    def test_for_consumes_upstream_node_output(self):
        def wf(n):
            xs = my_range(n)
            results = []
            for x in xs:
                y = identity(x)
                results.append(y)
            return results

        node = self._parse(wf)
        target = edge_models.TargetHandle(node="for_0", port="xs")
        self.assertIn(target, node.edges)
        self.assertEqual(node.edges[target].node, "my_range_0")
        self.assertEqual(node.edges[target].port, "output_0")

    def test_nested_for_inside_body(self):
        def wf(xs):
            results = []
            for x in xs:
                ys = my_range(x)
                inner = []
                for y in ys:
                    z = identity(y)
                    inner.append(z)
                results.append(inner)
            return results

        fn = self._parse(wf).nodes["for_0"]
        body = fn.body_node.node
        self.assertIsInstance(body, workflow_model.WorkflowNode)
        self.assertIn("for_0", body.nodes)
        self.assertIsInstance(body.nodes["for_0"], for_model.ForNode)

    def test_accumulator_cleanup_allows_second_for(self):
        """After a for-node consumes accumulators, new ones can be defined."""

        def wf(xs, ys):
            first = []
            for x in xs:
                a = identity(x)
                first.append(a)
            second = []
            for y in ys:
                b = identity(y)
                second.append(b)
            return first, second

        node = self._parse(wf)
        self.assertIn("for_0", node.nodes)
        self.assertIn("for_1", node.nodes)
        self.assertEqual(sorted(node.outputs), ["first", "second"])


# ===================================================================
# Round-trip serialisation
# ===================================================================


class TestForNodeRoundTrip(unittest.TestCase):
    def test_for_node_round_trip(self):
        def wf(xs):
            results = []
            for x in xs:
                y = identity(x)
                results.append(y)
            return results

        fn = workflow_parser.parse_workflow(wf).nodes["for_0"]
        for mode in ["json", "python"]:
            with self.subTest(mode=mode):
                dumped = fn.model_dump(mode=mode)
                restored = for_model.ForNode.model_validate(dumped)
                self.assertEqual(fn, restored)

    def test_workflow_round_trip(self):
        """The whole workflow containing a for-node survives round-trip."""

        def wf(xs, c):
            collected = []
            results = []
            for x in xs:
                y = pair(x, c)
                collected.append(x)
                results.append(y)
            return collected, results

        node = workflow_parser.parse_workflow(wf)
        for mode in ["json", "python"]:
            with self.subTest(mode=mode):
                dumped = node.model_dump(mode=mode)
                restored = workflow_model.WorkflowNode.model_validate(dumped)
                self.assertEqual(node, restored)


# ===================================================================
# Append-to-accumulator edge cases
# ===================================================================


class TestAppendAccumulator(unittest.TestCase):
    """Edge cases around handle_appending_to_accumulator inside for bodies."""

    def test_non_append_expr_raises(self):
        """A bare expression that isn't an .append() on an accumulator."""

        def wf(xs):
            results = []
            for x in xs:
                identity(x)
                results.append(x)
            return results

        # The bare `identity(x)` is an ast.Expr but not an append to an
        # accumulator, so is_append_call returns False → TypeError
        with self.assertRaises(TypeError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("accumulator", str(ctx.exception).lower())

    def test_append_to_unknown_symbol_raises(self):
        """Appending to a list that was never initialised as []."""

        def wf(xs):
            results = []
            for x in xs:
                y = identity(x)
                other.append(y)  # noqa: F821
                results.append(y)
            return results

        with self.assertRaises(TypeError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("accumulator", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
