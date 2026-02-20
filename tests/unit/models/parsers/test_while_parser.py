import unittest

from flowrep.models import edge_models
from flowrep.models.nodes import (
    for_model,
    if_model,
    try_model,
    while_model,
    workflow_model,
)
from flowrep.models.parsers import atomic_parser, while_parser, workflow_parser


@atomic_parser.atomic
def identity(x):
    return x


@atomic_parser.atomic
def my_add(a, b):
    return a + b


@atomic_parser.atomic
def my_condition(m, n):
    return m < n


def multi_result(x):
    """Undecorated; parsed on-the-fly with two outputs."""
    a = x + 1
    b = x - 1
    return a, b


class TestParseWhileConditionErrors(unittest.TestCase):
    """Error paths in parse_while_condition, surfaced through parse_workflow."""

    def test_bare_symbol_condition_raises(self):
        """Condition must be a function call, not a bare name."""

        def wf(x, flag):
            while flag:
                x = identity(x)
            return x

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("function call", str(ctx.exception))

    def test_comparison_condition_raises(self):
        """Inline comparison is not a function call."""

        def wf(x, bound):
            while x < bound:
                x = my_add(x, bound)
            return x

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("function call", str(ctx.exception))

    def test_multi_output_condition_raises(self):
        """Condition function must return exactly one value."""

        def wf(x):
            while multi_result(x):
                x = identity(x)
            return x

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("exactly one", str(ctx.exception))


class TestWhileParserErrors(unittest.TestCase):
    """
    The while parser is reached through WorkflowParser.handle_while,
    so we test error paths by defining small invalid workflow functions.
    """

    def test_unrecognized_body_stmt_raises(self):
        """ast.Return inside a while body is not handled → TypeError."""

        def wf(x, bound):
            while my_condition(x, bound):
                x = identity(x)
                return x
            return x

        with self.assertRaises(TypeError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("ast found", str(ctx.exception))

    def test_no_reassignment_raises(self):
        """While body must reassign at least one symbol from the enclosing scope."""

        def wf(x, bound):
            while my_condition(x, bound):
                y = identity(x)  # noqa: F841
            return x

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("reassign", str(ctx.exception).lower())

    def test_while_else_raises(self):
        """while...else is not supported."""

        def wf(x, bound):
            while my_condition(x, bound):
                x = identity(x)
            else:
                x = identity(x)
            return x

        with self.assertRaises((ValueError, NotImplementedError)):
            workflow_parser.parse_workflow(wf)


class TestWhileParserEdgeWiring(unittest.TestCase):
    """Verify the input/output edge construction for parsed while nodes."""

    @staticmethod
    def _get_while_node(func) -> while_model.WhileNode:
        wf = workflow_parser.parse_workflow(func)
        while_nodes = [
            n for n in wf.nodes.values() if isinstance(n, while_model.WhileNode)
        ]
        assert len(while_nodes) == 1, f"Expected 1 WhileNode, got {len(while_nodes)}"
        return while_nodes[0]

    # --- condition wiring ---

    def test_condition_input_edges(self):
        """Condition node receives its inputs via input_edges."""

        def wf(x, bound):
            while my_condition(x, bound):
                x = identity(x)
            return x

        wn = self._get_while_node(wf)
        cond_m = edge_models.TargetHandle(node="condition", port="m")
        cond_n = edge_models.TargetHandle(node="condition", port="n")
        self.assertIn(cond_m, wn.input_edges)
        self.assertIn(cond_n, wn.input_edges)
        self.assertEqual(wn.input_edges[cond_m].port, "x")
        self.assertEqual(wn.input_edges[cond_n].port, "bound")

    # --- body wiring ---

    def test_body_input_edges(self):
        """Body node receives its inputs via input_edges."""

        def wf(x, bound):
            while my_condition(x, bound):
                x = identity(x)
            return x

        wn = self._get_while_node(wf)
        body_x = edge_models.TargetHandle(node="body", port="x")
        self.assertIn(body_x, wn.input_edges)
        self.assertEqual(wn.input_edges[body_x].port, "x")

    def test_output_edges_from_body(self):
        """Output edges source from the body node."""

        def wf(x, bound):
            while my_condition(x, bound):
                x = identity(x)
            return x

        wn = self._get_while_node(wf)
        target = edge_models.OutputTarget(port="x")
        self.assertIn(target, wn.output_edges)
        source = wn.output_edges[target]
        self.assertIsInstance(source, edge_models.SourceHandle)
        self.assertEqual(source.node, "body")

    # --- shared, broadcast, and condition-only inputs ---

    def test_shared_input_feeds_condition_and_body(self):
        """Same while-node input can feed both condition and body."""

        def wf(x, bound):
            while my_condition(x, bound):
                x = identity(x)
            return x

        wn = self._get_while_node(wf)
        cond_target = edge_models.TargetHandle(node="condition", port="m")
        body_target = edge_models.TargetHandle(node="body", port="x")
        self.assertEqual(wn.input_edges[cond_target].port, "x")
        self.assertEqual(wn.input_edges[body_target].port, "x")

    def test_broadcast_symbol_is_input_not_output(self):
        """A symbol consumed in the body but not reassigned is broadcast."""

        def wf(x, step, bound):
            while my_condition(x, bound):
                x = my_add(x, step)
            return x

        wn = self._get_while_node(wf)
        self.assertIn("step", wn.inputs)
        self.assertNotIn("step", wn.outputs)
        body_step = edge_models.TargetHandle(node="body", port="step")
        self.assertIn(body_step, wn.input_edges)

    def test_condition_only_input_not_in_outputs(self):
        """An input consumed only by the condition is not an output."""

        def wf(x, bound):
            while my_condition(x, bound):
                x = identity(x)
            return x

        wn = self._get_while_node(wf)
        self.assertIn("bound", wn.inputs)
        self.assertNotIn("bound", wn.outputs)

    # --- reassignment → output ---

    def test_single_reassignment_becomes_output(self):
        def wf(x, bound):
            while my_condition(x, bound):
                x = identity(x)
            return x

        wn = self._get_while_node(wf)
        self.assertEqual(wn.outputs, ["x"])

    def test_multiple_reassignments_become_outputs(self):
        def wf(x, y, bound):
            while my_condition(x, bound):
                x = my_add(x, y)
                y = identity(x)
            return x, y

        wn = self._get_while_node(wf)
        self.assertEqual(sorted(wn.outputs), ["x", "y"])
        self.assertTrue(set(wn.outputs).issubset(set(wn.inputs)))

    def test_reassignment_from_outer_scope_becomes_output(self):
        def wf(y, bound):
            x = identity(y)
            while my_condition(x, bound):
                x = identity(x)
            return x

        wn = self._get_while_node(wf)
        self.assertEqual(wn.outputs, ["x"])


class TestWhileParserStructure(unittest.TestCase):

    def _parse(self, func) -> workflow_model.WorkflowNode:
        return workflow_parser.parse_workflow(func)

    def test_while_node_registered_in_parent(self):
        def wf(x, bound):
            while my_condition(x, bound):
                x = identity(x)
            return x

        node = self._parse(wf)
        self.assertIn("while_0", node.nodes)
        self.assertIsInstance(node.nodes["while_0"], while_model.WhileNode)

    def test_condition_label(self):
        def wf(x, bound):
            while my_condition(x, bound):
                x = identity(x)
            return x

        wn = self._parse(wf).nodes["while_0"]
        self.assertEqual(wn.case.condition.label, while_parser.WHILE_CONDITION_LABEL)

    def test_body_label(self):
        def wf(x, bound):
            while my_condition(x, bound):
                x = identity(x)
            return x

        wn = self._parse(wf).nodes["while_0"]
        self.assertEqual(wn.case.body.label, while_parser.WHILE_BODY_LABEL)

    def test_body_node_is_workflow(self):
        """
        In principle, very simple bodies could be atomic nodes directly, but for
        uniformity and simplicity of parsing, we always parse control flow bodies as a
        workflow
        """

        def wf(x, bound):
            while my_condition(x, bound):
                x = identity(x)
            return x

        wn = self._parse(wf).nodes["while_0"]
        self.assertIsInstance(wn.case.body.node, workflow_model.WorkflowNode)

    def test_outputs_subset_of_inputs(self):
        def wf(x, step, bound):
            while my_condition(x, bound):
                x = my_add(x, step)
            return x

        wn = self._parse(wf).nodes["while_0"]
        self.assertTrue(set(wn.outputs).issubset(set(wn.inputs)))

    def test_while_consumes_upstream_node_output(self):
        """While node can consume sibling output from a preceding node."""

        def wf(a, bound):
            x = identity(a)
            while my_condition(x, bound):
                x = identity(x)
            return x

        node = self._parse(wf)
        target = edge_models.TargetHandle(node="while_0", port="x")
        self.assertIn(target, node.edges)
        self.assertEqual(node.edges[target].node, "identity_0")
        self.assertEqual(node.edges[target].port, "x")

    def test_while_output_consumed_by_downstream_node(self):
        """Output of while node feeds a downstream sibling."""

        def wf(x, bound):
            while my_condition(x, bound):
                x = identity(x)
            y = identity(x)
            return y

        node = self._parse(wf)
        target = edge_models.TargetHandle(node="identity_0", port="x")
        self.assertIn(target, node.edges)
        self.assertEqual(node.edges[target].node, "while_0")
        self.assertEqual(node.edges[target].port, "x")

    def test_multiple_while_nodes_get_unique_labels(self):
        def wf(x, m, n):
            while my_condition(x, m):
                x = identity(x)
            while my_condition(x, n):
                x = identity(x)
            return x

        node = self._parse(wf)
        self.assertIn("while_0", node.nodes)
        self.assertIn("while_1", node.nodes)

    def test_while_catches_reassigned_symbols_as_input(self):
        """When a symbol is reassigned, but not explicitly body or condition input."""

        def wf(x, bound, y):
            while my_condition(x, bound):
                x = my_add(x, x)
                y = identity(x)
            return y

        node = self._parse(wf)
        self.assertIn("y", node.nodes["while_0"].inputs)
        self.assertIn("y", node.nodes["while_0"].outputs)

    def test_for_nested_inside_while_body(self):
        """A for-loop inside a while-body produces a ForNode in the body workflow."""

        def wf(x, bound):
            while my_condition(x, bound):
                xs = identity(x)
                acc = []
                for i in xs:
                    v = identity(i)
                    acc.append(v)
                x = identity(acc)
            return x

        wn = self._parse(wf).nodes["while_0"]
        body = wn.case.body.node
        self.assertIsInstance(body, workflow_model.WorkflowNode)
        for_nodes = [n for n in body.nodes.values() if isinstance(n, for_model.ForNode)]
        self.assertEqual(len(for_nodes), 1)
        # The for-node's output is consumed downstream in the while body
        self.assertIn("acc", for_nodes[0].outputs)

    def test_if_nested_inside_while_body(self):
        """An if-node inside a while-body produces an IfNode in the body workflow."""

        def wf(x, y, bound):
            while my_condition(x, bound):
                if my_condition(x, y):  # noqa: SIM108
                    x = my_add(x, y)
                else:
                    x = identity(x)
            return x

        wn = self._parse(wf).nodes["while_0"]
        body = wn.case.body.node
        self.assertIsInstance(body, workflow_model.WorkflowNode)
        if_nodes = [n for n in body.nodes.values() if isinstance(n, if_model.IfNode)]
        self.assertEqual(len(if_nodes), 1)
        # The if-node's output feeds the while reassignment
        self.assertIn("x", if_nodes[0].outputs)

    def test_try_nested_inside_while_body(self):
        """A try/except inside a while-body produces a TryNode in the body workflow."""

        def wf(x, y, bound):
            while my_condition(x, bound):
                try:
                    x = my_add(x, y)
                except ValueError:
                    x = identity(x)
            return x

        wn = self._parse(wf).nodes["while_0"]
        body = wn.case.body.node
        self.assertIsInstance(body, workflow_model.WorkflowNode)
        try_nodes = [n for n in body.nodes.values() if isinstance(n, try_model.TryNode)]
        self.assertEqual(len(try_nodes), 1)
        self.assertIn("x", try_nodes[0].outputs)


class TestWhileParserRoundTrip(unittest.TestCase):
    def test_while_node_round_trip(self):
        def wf(x, bound):
            while my_condition(x, bound):
                x = identity(x)
            return x

        wn = workflow_parser.parse_workflow(wf).nodes["while_0"]
        for mode in ["json", "python"]:
            with self.subTest(mode=mode):
                dumped = wn.model_dump(mode=mode)
                restored = while_model.WhileNode.model_validate(dumped)
                self.assertEqual(wn, restored)

    def test_workflow_with_while_round_trip(self):
        """The whole workflow containing a while-node survives round-trip."""

        def wf(x, step, bound):
            while my_condition(x, bound):
                x = my_add(x, step)
            y = identity(x)
            return y

        node = workflow_parser.parse_workflow(wf)
        for mode in ["json", "python"]:
            with self.subTest(mode=mode):
                dumped = node.model_dump(mode=mode)
                restored = workflow_model.WorkflowNode.model_validate(dumped)
                self.assertEqual(node, restored)


if __name__ == "__main__":
    unittest.main()
