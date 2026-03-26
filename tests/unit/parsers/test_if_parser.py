# ruff: noqa: SIM108
# We have a lot of if-clauses that could be one-liners and ruff complains
# Even if/when we have the syntactic sugar to parse one-liners, we'll still _want_
# to test the multi-line equivalent, so get ruff to be quiet about this
import ast
import textwrap
import unittest

from flowrep import edge_models
from flowrep.nodes import (
    for_model,
    if_model,
    try_model,
    while_model,
    workflow_model,
)
from flowrep.parsers import if_parser, workflow_parser

from flowrep_static import library

# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _parse_if_stmt(code: str) -> ast.If:
    """Return the first ast.If found in *code*."""
    tree = ast.parse(textwrap.dedent(code))
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            return node
    raise ValueError("No ast.If found")


# ===================================================================
# parse_if_elif_chain
# ===================================================================


class TestParseIfElifChain(unittest.TestCase):
    """Direct tests for the if/elif/else chain flattener."""

    def test_simple_if(self):
        stmt = _parse_if_stmt("if cond():\n  x = 1")
        cases, else_body = if_parser._parse_if_elif_chain(stmt)
        self.assertEqual(len(cases), 1)
        self.assertIsNone(else_body)

    def test_if_else(self):
        stmt = _parse_if_stmt("if cond():\n  x = 1\nelse:\n  x = 2")
        cases, else_body = if_parser._parse_if_elif_chain(stmt)
        self.assertEqual(len(cases), 1)
        self.assertIsNotNone(else_body)
        self.assertEqual(len(else_body), 1)

    def test_if_elif(self):
        code = "if a():\n  x = 1\nelif b():\n  x = 2"
        stmt = _parse_if_stmt(code)
        cases, else_body = if_parser._parse_if_elif_chain(stmt)
        self.assertEqual(len(cases), 2)
        self.assertIsNone(else_body)

    def test_if_elif_else(self):
        code = "if a():\n  x = 1\nelif b():\n  x = 2\nelse:\n  x = 3"
        stmt = _parse_if_stmt(code)
        cases, else_body = if_parser._parse_if_elif_chain(stmt)
        self.assertEqual(len(cases), 2)
        self.assertIsNotNone(else_body)

    def test_if_elif_elif_else(self):
        code = (
            "if a():\n  x = 1\n"
            "elif b():\n  x = 2\n"
            "elif c():\n  x = 3\n"
            "else:\n  x = 4"
        )
        stmt = _parse_if_stmt(code)
        cases, else_body = if_parser._parse_if_elif_chain(stmt)
        self.assertEqual(len(cases), 3)
        self.assertIsNotNone(else_body)

    def test_body_statements_preserved(self):
        code = "if a():\n  x = 1\n  y = 2\nelse:\n  z = 3"
        stmt = _parse_if_stmt(code)
        cases, else_body = if_parser._parse_if_elif_chain(stmt)
        self.assertEqual(len(cases[0][1]), 2)
        self.assertEqual(len(else_body), 1)


# ===================================================================
# parse_if_condition error paths (tested via parse_workflow)
# ===================================================================


class TestParseIfConditionErrors(unittest.TestCase):
    """Error paths in parse_if_condition, surfaced through parse_workflow."""

    def test_bare_symbol_condition_raises(self):
        """Condition must be a function call, not a bare name."""

        def wf(x, flag):
            if flag:
                x = library.identity(x)
            else:
                x = library.my_add(x, x)
            return x

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("function call", str(ctx.exception))

    def test_comparison_condition_raises(self):
        """Inline comparison is not a function call."""

        def wf(x, bound):
            if x < bound:
                x = library.identity(x)
            else:
                x = library.my_add(x, bound)
            return x

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("function call", str(ctx.exception))

    def test_multi_output_condition_raises(self):
        """Condition function must return exactly one value."""

        def wf(x):
            if library.multi_result(x):
                x = library.identity(x)
            else:
                x = library.my_add(x, x)
            return x

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("exactly one", str(ctx.exception))

    def test_elif_bare_symbol_raises(self):
        """elif branch conditions are also validated."""

        def wf(x, flag, other):
            if library.my_condition(x, flag):
                x = library.identity(x)
            elif other:
                x = library.my_add(x, x)
            return x

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("function call", str(ctx.exception))


# ===================================================================
# Body-level errors (tested via parse_workflow)
# ===================================================================


class TestIfBodyErrors(unittest.TestCase):
    """Errors inside if/else bodies surfaced through parse_workflow."""

    def test_unrecognized_body_stmt_raises(self):
        """ast.Return inside an if body is not handled TypeError."""

        def wf(x, y):
            if library.my_condition(x, y):
                x = library.identity(x)
                return x
            else:
                x = library.identity(x)
            return x

        with self.assertRaises(TypeError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("ast found", str(ctx.exception))


# ===================================================================
# IfParser edge wiring (tested via parse_workflow)
# ===================================================================


class TestIfParserEdgeWiring(unittest.TestCase):
    """Verify the input/output/prospective-output edge construction."""

    @staticmethod
    def _get_if_node(func) -> if_model.IfNode:
        wf = workflow_parser.parse_workflow(func)
        if_nodes = [n for n in wf.nodes.values() if isinstance(n, if_model.IfNode)]
        assert len(if_nodes) == 1, f"Expected 1 IfNode, got {len(if_nodes)}"
        return if_nodes[0]

    # --- condition wiring ---

    def test_condition_input_edges(self):
        """Condition node receives its inputs via input_edges."""

        def wf(x, y):
            if library.my_condition(x, y):
                z = library.my_add(x, y)
            else:
                z = library.my_mul(x, y)
            return z

        ifn = self._get_if_node(wf)
        cond_m = edge_models.TargetHandle(node="condition_0", port="m")
        cond_n = edge_models.TargetHandle(node="condition_0", port="n")
        self.assertIn(cond_m, ifn.input_edges)
        self.assertIn(cond_n, ifn.input_edges)
        self.assertEqual(ifn.input_edges[cond_m].port, "x")
        self.assertEqual(ifn.input_edges[cond_n].port, "y")

    def test_elif_condition_input_edges(self):
        """Each elif condition gets its own input edges."""

        def wf(x, y, flag):
            if library.my_condition(x, flag):
                z = library.my_add(x, y)
            elif library.my_condition(y, flag):
                z = library.my_mul(x, y)
            else:
                z = library.identity(x)
            return z

        ifn = self._get_if_node(wf)
        # First condition
        self.assertIn(
            edge_models.TargetHandle(node="condition_0", port="m"), ifn.input_edges
        )
        # Second condition
        self.assertIn(
            edge_models.TargetHandle(node="condition_1", port="m"), ifn.input_edges
        )

    # --- body wiring ---

    def test_body_input_edges(self):
        """Body nodes receive their inputs via input_edges."""

        def wf(x, y):
            if library.my_condition(x, y):
                z = library.my_add(x, y)
            else:
                z = library.my_mul(x, y)
            return z

        ifn = self._get_if_node(wf)
        # body_0 should have input edges for x and y
        body_x = edge_models.TargetHandle(node="body_0", port="x")
        body_y = edge_models.TargetHandle(node="body_0", port="y")
        self.assertIn(body_x, ifn.input_edges)
        self.assertIn(body_y, ifn.input_edges)
        self.assertEqual(ifn.input_edges[body_x].port, "x")
        self.assertEqual(ifn.input_edges[body_y].port, "y")

    def test_else_body_input_edges(self):
        """Else body receives its inputs via input_edges."""

        def wf(x, y):
            if library.my_condition(x, y):
                z = library.my_add(x, y)
            else:
                z = library.my_mul(x, y)
            return z

        ifn = self._get_if_node(wf)
        else_x = edge_models.TargetHandle(node="else_body", port="x")
        else_y = edge_models.TargetHandle(node="else_body", port="y")
        self.assertIn(else_x, ifn.input_edges)
        self.assertIn(else_y, ifn.input_edges)

    # --- prospective output edges ---

    def test_prospective_output_edges_if_else(self):
        """Both branches contribute prospective sources for each output."""

        def wf(x, y):
            if library.my_condition(x, y):
                z = library.my_add(x, y)
            else:
                z = library.my_mul(x, y)
            return z

        ifn = self._get_if_node(wf)
        target = edge_models.OutputTarget(port="z")
        self.assertIn(target, ifn.prospective_output_edges)
        sources = ifn.prospective_output_edges[target]
        self.assertEqual(len(sources), 2)
        source_nodes = {s.node for s in sources}
        self.assertEqual(source_nodes, {"body_0", "else_body"})

    def test_prospective_output_edges_if_elif_else(self):
        """All three branches contribute prospective sources."""

        def wf(x, y, flag):
            if library.my_condition(x, flag):
                z = library.my_add(x, y)
            elif library.my_condition(y, flag):
                z = library.my_mul(x, y)
            else:
                z = library.identity(x)
            return z

        ifn = self._get_if_node(wf)
        target = edge_models.OutputTarget(port="z")
        sources = ifn.prospective_output_edges[target]
        self.assertEqual(len(sources), 3)
        source_nodes = {s.node for s in sources}
        self.assertEqual(source_nodes, {"body_0", "body_1", "else_body"})

    def test_prospective_output_edges_no_else(self):
        """Without else, only the if-branch contributes a prospective source."""

        def wf(x, y):
            z = library.identity(x)
            if library.my_condition(x, y):
                z = library.my_add(x, y)
            return z

        ifn = self._get_if_node(wf)
        target = edge_models.OutputTarget(port="z")
        sources = ifn.prospective_output_edges[target]
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].node, "body_0")

    # --- partial branch coverage of outputs ---

    def test_output_only_in_one_branch(self):
        """An output assigned only in one branch still appears in the outputs,
        but with a single prospective source."""

        def wf(x, y):
            if library.my_condition(x, y):
                a = library.my_add(x, y)
                b = library.my_mul(x, y)  # noqa: F841
                # b is part of the if-node output intentionally, since not all branches
                # need to return the same things. We don't care here whether (or in this
                # case not) it appears in the final workflow output
            else:
                a = library.my_mul(x, y)
            return a

        ifn = self._get_if_node(wf)
        # 'a' has sources from both branches
        a_target = edge_models.OutputTarget(port="a")
        self.assertEqual(len(ifn.prospective_output_edges[a_target]), 2)
        # 'b' has a source from only body_0
        b_target = edge_models.OutputTarget(port="b")
        self.assertIn(b_target, ifn.prospective_output_edges)
        self.assertEqual(len(ifn.prospective_output_edges[b_target]), 1)
        self.assertEqual(ifn.prospective_output_edges[b_target][0].node, "body_0")

    # --- input collection ---

    def test_condition_only_input_appears(self):
        """A symbol consumed only by the condition still appears in inputs."""

        def wf(x, y, flag):
            if library.my_condition(x, flag):
                z = library.identity(y)
            else:
                z = library.identity(y)
            return z

        ifn = self._get_if_node(wf)
        self.assertIn("flag", ifn.inputs)

    def test_all_branches_contribute_inputs(self):
        """Inputs are collected from conditions, bodies, and else."""

        def wf(x, y, flag):
            if library.my_condition(x, flag):
                z = library.my_add(x, y)
            else:
                z = library.identity(y)
            return z

        ifn = self._get_if_node(wf)
        self.assertIn("x", ifn.inputs)
        self.assertIn("y", ifn.inputs)
        self.assertIn("flag", ifn.inputs)


# ===================================================================
# IfParser structural properties
# ===================================================================


class TestIfParserStructure(unittest.TestCase):
    def _parse(self, func) -> workflow_model.WorkflowNode:
        return workflow_parser.parse_workflow(func)

    def test_if_node_registered_in_parent(self):
        def wf(x, y):
            if library.my_condition(x, y):
                z = library.my_add(x, y)
            else:
                z = library.my_mul(x, y)
            return z

        node = self._parse(wf)
        self.assertIn("if_0", node.nodes)
        self.assertIsInstance(node.nodes["if_0"], if_model.IfNode)

    def test_condition_labels(self):
        def wf(x, y, flag):
            if library.my_condition(x, flag):
                z = library.my_add(x, y)
            elif library.my_condition(y, flag):
                z = library.my_mul(x, y)
            else:
                z = library.identity(x)
            return z

        ifn = self._parse(wf).nodes["if_0"]
        self.assertEqual(ifn.cases[0].condition.label, "condition_0")
        self.assertEqual(ifn.cases[1].condition.label, "condition_1")

    def test_body_labels(self):
        def wf(x, y, flag):
            if library.my_condition(x, flag):
                z = library.my_add(x, y)
            elif library.my_condition(y, flag):
                z = library.my_mul(x, y)
            else:
                z = library.identity(x)
            return z

        ifn = self._parse(wf).nodes["if_0"]
        self.assertEqual(ifn.cases[0].body.label, "body_0")
        self.assertEqual(ifn.cases[1].body.label, "body_1")

    def test_else_label(self):
        def wf(x, y):
            if library.my_condition(x, y):
                z = library.my_add(x, y)
            else:
                z = library.my_mul(x, y)
            return z

        ifn = self._parse(wf).nodes["if_0"]
        self.assertIsNotNone(ifn.else_case)
        self.assertEqual(ifn.else_case.label, if_parser.IF_ELSE_LABEL)

    def test_no_else_case(self):
        def wf(x, y):
            z = library.identity(x)
            if library.my_condition(x, y):
                z = library.my_add(x, y)
            return z

        ifn = self._parse(wf).nodes["if_0"]
        self.assertIsNone(ifn.else_case)

    def test_body_nodes_are_workflows(self):
        """All body nodes are WorkflowNodes for uniformity."""

        def wf(x, y):
            if library.my_condition(x, y):
                z = library.my_add(x, y)
            else:
                z = library.my_mul(x, y)
            return z

        ifn = self._parse(wf).nodes["if_0"]
        self.assertIsInstance(ifn.cases[0].body.node, workflow_model.WorkflowNode)
        self.assertIsInstance(ifn.else_case.node, workflow_model.WorkflowNode)

    def test_multiple_outputs(self):
        def wf(x, y):
            if library.my_condition(x, y):
                a = library.my_add(x, y)
                b = library.my_mul(x, y)
            else:
                a = library.my_mul(x, y)
                b = library.my_add(x, y)
            return a, b

        ifn = self._parse(wf).nodes["if_0"]
        self.assertEqual(sorted(ifn.outputs), ["a", "b"])

    def test_if_consumes_upstream_node_output(self):
        """If node can consume sibling output from a preceding node."""

        def wf(a, b):
            x = library.my_add(a, b)
            if library.my_condition(x, b):
                y = library.my_mul(x, b)
            else:
                y = library.my_add(x, b)
            return y

        node = self._parse(wf)
        target = edge_models.TargetHandle(node="if_0", port="x")
        self.assertIn(target, node.edges)
        self.assertEqual(node.edges[target].node, "my_add_0")
        self.assertEqual(node.edges[target].port, "output_0")

    def test_if_output_consumed_by_downstream_node(self):
        """Output of if node feeds a downstream sibling."""

        def wf(x, y):
            if library.my_condition(x, y):
                z = library.my_add(x, y)
            else:
                z = library.my_mul(x, y)
            result = library.identity(z)
            return result

        node = self._parse(wf)
        target = edge_models.TargetHandle(node="identity_0", port="x")
        self.assertIn(target, node.edges)
        self.assertEqual(node.edges[target].node, "if_0")
        self.assertEqual(node.edges[target].port, "z")

    def test_multiple_if_nodes_get_unique_labels(self):
        def wf(x, y, m, n):
            if library.my_condition(x, m):
                a = library.my_add(x, y)
            else:
                a = library.my_mul(x, y)
            if library.my_condition(a, n):
                b = library.identity(a)
            else:
                b = library.my_add(a, y)
            return b

        node = self._parse(wf)
        self.assertIn("if_0", node.nodes)
        self.assertIn("if_1", node.nodes)

    def test_sequential_if_nodes_edge_wiring(self):
        """Output of first if feeds input of second if via sibling edge."""

        def wf(x, y, m, n):
            if library.my_condition(x, m):
                a = library.my_add(x, y)
            else:
                a = library.my_mul(x, y)
            if library.my_condition(a, n):
                b = library.identity(a)
            else:
                b = library.my_add(a, y)
            return b

        node = self._parse(wf)
        target = edge_models.TargetHandle(node="if_1", port="a")
        self.assertIn(target, node.edges)
        self.assertEqual(node.edges[target].node, "if_0")
        self.assertEqual(node.edges[target].port, "a")

    # --- nesting other control flows inside if bodies ---

    def test_for_nested_inside_if_body(self):
        """A for-loop inside an if-body produces a ForNode in the body workflow."""

        def wf(xs, y):
            if library.my_condition(y, y):
                results = []
                for x in xs:
                    v = library.identity(x)
                    results.append(v)
                z = library.identity(results)
            else:
                z = library.identity(y)
            return z

        ifn = self._parse(wf).nodes["if_0"]
        body = ifn.cases[0].body.node
        self.assertIsInstance(body, workflow_model.WorkflowNode)
        for_nodes = [n for n in body.nodes.values() if isinstance(n, for_model.ForNode)]
        self.assertEqual(len(for_nodes), 1)

    def test_while_nested_inside_if_body(self):
        """A while-loop inside an if-body produces a WhileNode in the body workflow."""

        def wf(x, y, bound):
            if library.my_condition(x, y):
                while library.my_condition(x, bound):
                    x = library.my_add(x, y)
                z = library.identity(x)
            else:
                z = library.identity(y)
            return z

        ifn = self._parse(wf).nodes["if_0"]
        body = ifn.cases[0].body.node
        self.assertIsInstance(body, workflow_model.WorkflowNode)
        while_nodes = [
            n for n in body.nodes.values() if isinstance(n, while_model.WhileNode)
        ]
        self.assertEqual(len(while_nodes), 1)

    def test_if_nested_inside_if_body(self):
        """An if-node inside an if-body produces an IfNode in the body workflow."""

        def wf(x, y, m):
            if library.my_condition(x, y):
                if library.my_condition(x, m):
                    z = library.my_add(x, y)
                else:
                    z = library.my_mul(x, y)
            else:
                z = library.identity(x)
            return z

        ifn = self._parse(wf).nodes["if_0"]
        body = ifn.cases[0].body.node
        self.assertIsInstance(body, workflow_model.WorkflowNode)
        inner_if_nodes = [
            n for n in body.nodes.values() if isinstance(n, if_model.IfNode)
        ]
        self.assertEqual(len(inner_if_nodes), 1)

    def test_for_nested_inside_else_body(self):
        """A for-loop inside an else-body produces a ForNode in the else workflow."""

        def wf(xs, y):
            if library.my_condition(y, y):
                z = library.identity(y)
            else:
                results = []
                for x in xs:
                    v = library.identity(x)
                    results.append(v)
                z = library.identity(results)
            return z

        ifn = self._parse(wf).nodes["if_0"]
        else_body = ifn.else_case.node
        self.assertIsInstance(else_body, workflow_model.WorkflowNode)
        for_nodes = [
            n for n in else_body.nodes.values() if isinstance(n, for_model.ForNode)
        ]
        self.assertEqual(len(for_nodes), 1)

    def test_try_nested_inside_if_body(self):
        """A try/except inside an if-body produces a TryNode in the body workflow."""

        def wf(x, y):
            if library.my_condition(x, y):
                try:
                    z = library.my_add(x, y)
                except ValueError:
                    z = library.my_mul(x, y)
            else:
                z = library.identity(x)
            return z

        ifn = self._parse(wf).nodes["if_0"]
        body = ifn.cases[0].body.node
        self.assertIsInstance(body, workflow_model.WorkflowNode)
        try_nodes = [n for n in body.nodes.values() if isinstance(n, try_model.TryNode)]
        self.assertEqual(len(try_nodes), 1)

    def test_try_nested_inside_else_body(self):
        """A try/except inside an else-body produces a TryNode in the else workflow."""

        def wf(x, y):
            if library.my_condition(x, y):
                z = library.identity(x)
            else:
                try:
                    z = library.my_add(x, y)
                except ValueError:
                    z = library.my_mul(x, y)
            return z

        ifn = self._parse(wf).nodes["if_0"]
        else_body = ifn.else_case.node
        self.assertIsInstance(else_body, workflow_model.WorkflowNode)
        try_nodes = [
            n for n in else_body.nodes.values() if isinstance(n, try_model.TryNode)
        ]
        self.assertEqual(len(try_nodes), 1)


# ===================================================================
# Round-trip serialisation
# ===================================================================


class TestIfNodeRoundTrip(unittest.TestCase):
    def test_if_else_round_trip(self):
        def wf(x, y):
            if library.my_condition(x, y):
                z = library.my_add(x, y)
            else:
                z = library.my_mul(x, y)
            return z

        ifn = workflow_parser.parse_workflow(wf).nodes["if_0"]
        for mode in ["json", "python"]:
            with self.subTest(mode=mode):
                dumped = ifn.model_dump(mode=mode)
                restored = if_model.IfNode.model_validate(dumped)
                self.assertEqual(ifn, restored)

    def test_if_elif_else_round_trip(self):
        def wf(x, y, flag):
            if library.my_condition(x, flag):
                z = library.my_add(x, y)
            elif library.my_condition(y, flag):
                z = library.my_mul(x, y)
            else:
                z = library.identity(x)
            return z

        ifn = workflow_parser.parse_workflow(wf).nodes["if_0"]
        for mode in ["json", "python"]:
            with self.subTest(mode=mode):
                dumped = ifn.model_dump(mode=mode)
                restored = if_model.IfNode.model_validate(dumped)
                self.assertEqual(ifn, restored)

    def test_if_no_else_round_trip(self):
        def wf(x, y):
            z = library.identity(x)
            if library.my_condition(x, y):
                z = library.my_add(x, y)
            return z

        ifn = workflow_parser.parse_workflow(wf).nodes["if_0"]
        for mode in ["json", "python"]:
            with self.subTest(mode=mode):
                dumped = ifn.model_dump(mode=mode)
                restored = if_model.IfNode.model_validate(dumped)
                self.assertEqual(ifn, restored)

    def test_workflow_with_if_round_trip(self):
        """The whole workflow containing an if-node survives round-trip."""

        def wf(x, y):
            if library.my_condition(x, y):
                z = library.my_add(x, y)
            else:
                z = library.my_mul(x, y)
            result = library.identity(z)
            return result

        node = workflow_parser.parse_workflow(wf)
        for mode in ["json", "python"]:
            with self.subTest(mode=mode):
                dumped = node.model_dump(mode=mode)
                restored = workflow_model.WorkflowNode.model_validate(dumped)
                self.assertEqual(node, restored)

    def test_multi_output_if_round_trip(self):
        def wf(x, y):
            if library.my_condition(x, y):
                a = library.my_add(x, y)
                b = library.my_mul(x, y)
            else:
                a = library.my_mul(x, y)
                b = library.my_add(x, y)
            return a, b

        node = workflow_parser.parse_workflow(wf)
        for mode in ["json", "python"]:
            with self.subTest(mode=mode):
                dumped = node.model_dump(mode=mode)
                restored = workflow_model.WorkflowNode.model_validate(dumped)
                self.assertEqual(node, restored)


# ===================================================================
# Version propagation
# ===================================================================


class TestIfParserVersionPropagation(unittest.TestCase):
    """Version scraping/constraints propagate into if/else body child nodes."""

    def _pkg(self) -> str:
        return library.undecorated_identity.__module__.split(".")[0]

    def test_version_scraping_propagates_into_if_body(self):
        """Undecorated child inside an if body receives the scraping map."""
        custom = "10.20.30"

        def wf(x, y):
            if library.my_condition(x, y):
                z = library.undecorated_identity(x)
            else:
                z = library.undecorated_identity(y)
            return z

        node = workflow_parser.parse_workflow(
            wf, version_scraping={self._pkg(): lambda _: custom}
        )
        if_node = node.nodes["if_0"]
        body = if_node.cases[0].body.node
        child = body.nodes["undecorated_identity_0"]
        self.assertEqual(child.reference.info.version, custom)

    def test_version_scraping_propagates_into_else_body(self):
        """Undecorated child inside an else body receives the scraping map."""
        custom = "20.30.40"

        def wf(x, y):
            if library.my_condition(x, y):
                z = library.identity(x)
            else:
                z = library.undecorated_identity(y)
            return z

        node = workflow_parser.parse_workflow(
            wf, version_scraping={self._pkg(): lambda _: custom}
        )
        if_node = node.nodes["if_0"]
        else_body = if_node.else_case.node
        child = else_body.nodes["undecorated_identity_0"]
        self.assertEqual(child.reference.info.version, custom)

    def test_version_scraping_propagates_to_condition(self):
        """
        The condition node is pre-decorated (@atomic), so scraping should
        not override it.  Verify the body child *does* get the custom version
        while the condition does not — confirming selective propagation.
        """
        custom = "99.0.0"

        def wf(x, y):
            if library.my_condition(x, y):
                z = library.undecorated_identity(x)
            else:
                z = library.undecorated_identity(y)
            return z

        node = workflow_parser.parse_workflow(
            wf, version_scraping={self._pkg(): lambda _: custom}
        )
        if_node = node.nodes["if_0"]
        # condition is pre-decorated, and so keeps its own version
        condition_node = if_node.cases[0].condition.node
        self.assertNotEqual(condition_node.reference.info.version, custom)
        # body child is undecorated, and so picks up custom version
        body_child = if_node.cases[0].body.node.nodes["undecorated_identity_0"]
        self.assertEqual(body_child.reference.info.version, custom)

    def test_version_constraints_propagate_to_condition(self):
        def wf(x, y):
            if library.my_condition(x, y):
                z = library.undecorated_identity(x)
            else:
                z = library.undecorated_identity(y)
            return z

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(wf, require_version=True)
        self.assertIn("Could not find a version", str(ctx.exception))


class TestIfParserLocalImport(unittest.TestCase):
    def test_local_import_resolves(self):

        def my_wf(x, y):
            if library.my_condition(x, y):
                from flowrep_static.local_library import my_add as ma

                sum_ = ma(x, y)
            else:
                sum_ = library.my_add(x, y)
            return sum_

        node = workflow_parser.parse_workflow(my_wf)
        self.assertIsInstance(node, workflow_model.WorkflowNode)

    def test_local_imports_do_not_leak_to_sibling(self):

        def my_wf(x, y):
            if library.my_condition(x, y):
                from flowrep_static.local_library import my_add as ma

                sum_ = ma(x, y)
            else:
                sum_ = ma(x, y)
            return sum_

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(my_wf)
        self.assertIn("Could not find attribute 'ma'", str(ctx.exception))

    def test_local_imports_do_not_leak_to_parent(self):

        def my_wf(x, y):
            if library.my_condition(x, y):
                from flowrep_static.local_library import my_add as ma

                sum_ = ma(x, y)
            else:
                sum_ = library.my_add(x, y)

            bigger = ma(sum_, y)
            return bigger

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(my_wf)
        self.assertIn("Could not find attribute 'ma'", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
