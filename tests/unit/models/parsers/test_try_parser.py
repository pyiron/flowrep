import unittest

from pyiron_snippets import versions

from flowrep.models import edge_models
from flowrep.models.nodes import (
    for_model,
    if_model,
    try_model,
    while_model,
    workflow_model,
)
from flowrep.models.parsers import atomic_parser, try_parser, workflow_parser

# ---------------------------------------------------------------------------
# Helper callables reachable from the test-module scope so that
# object_scope.get_scope(wf) can resolve them during parsing.
# ---------------------------------------------------------------------------


@atomic_parser.atomic
def identity(x):
    return x


def undecorated_identity(x):
    """For checking parser propagation"""
    return x


@atomic_parser.atomic
def my_add(a, b):
    return a + b


@atomic_parser.atomic
def my_mul(a, b):
    return a * b


@atomic_parser.atomic
def my_condition(m, n):
    return m < n


def multi_result(x):
    """Undecorated; parsed on-the-fly with two outputs."""
    a = x + 1
    b = x - 1
    return a, b


def my_range(n):
    return list(range(n))


# ===================================================================
# Exception type parsing errors (tested via parse_workflow)
# ===================================================================


class TestParseExceptionTypeErrors(unittest.TestCase):
    """Error paths in _parse_exception_types, surfaced through parse_workflow."""

    def test_bare_except_raises(self):
        """Bare except (no type) is not supported."""

        def wf(x, y):
            try:
                z = my_add(x, y)
            except:  # noqa: E722
                z = identity(x)
            return z

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("Bare except", str(ctx.exception))

    def test_named_handler_raises(self):
        """'except ... as e:' is not supported."""

        def wf(x, y):
            try:
                z = my_add(x, y)
            except ValueError as e:  # noqa: F841
                z = identity(x)
            return z

        with self.assertRaises(NotImplementedError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("as e", str(ctx.exception))

    def test_non_exception_type_raises(self):
        """Catching something that isn't an exception type is rejected."""

        def wf(x, y):
            try:
                z = my_add(x, y)
            except identity:
                z = identity(x)
            return z

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("exception types", str(ctx.exception))


# ===================================================================
# Unsupported syntax errors
# ===================================================================


class TestTryUnsupportedSyntax(unittest.TestCase):

    def test_try_else_raises(self):
        """try/else is not supported."""

        def wf(x, y):
            try:
                z = my_add(x, y)
            except ValueError:
                z = identity(x)
            else:
                z = my_mul(x, y)
            return z

        with self.assertRaises(NotImplementedError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("else", str(ctx.exception))

    def test_try_finally_raises(self):
        """try/finally is not supported."""

        def wf(x, y):
            try:
                z = my_add(x, y)
            except ValueError:
                z = identity(x)
            finally:
                z = my_mul(x, y)
            return z

        with self.assertRaises(NotImplementedError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("finally", str(ctx.exception))


# ===================================================================
# Body-level errors (tested via parse_workflow)
# ===================================================================


class TestTryBodyErrors(unittest.TestCase):
    """Errors inside try/except bodies surfaced through parse_workflow."""

    def test_unrecognized_body_stmt_in_try_raises(self):
        """ast.Return inside a try body is not handled → TypeError."""

        def wf(x, y):
            try:
                z = identity(x)
                return z
            except ValueError:
                z = my_add(x, y)
            return z

        with self.assertRaises(TypeError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("ast found", str(ctx.exception))

    def test_unrecognized_body_stmt_in_except_raises(self):
        """ast.Return inside an except body is not handled → TypeError."""

        def wf(x, y):
            try:
                z = my_add(x, y)
            except ValueError:
                z = identity(x)
                return z
            return z

        with self.assertRaises(TypeError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("ast found", str(ctx.exception))


# ===================================================================
# TryParser edge wiring (tested via parse_workflow)
# ===================================================================


class TestTryParserEdgeWiring(unittest.TestCase):
    """Verify the input/output/prospective-output edge construction."""

    @staticmethod
    def _get_try_node(func) -> try_model.TryNode:
        wf = workflow_parser.parse_workflow(func)
        try_nodes = [n for n in wf.nodes.values() if isinstance(n, try_model.TryNode)]
        assert len(try_nodes) == 1, f"Expected 1 TryNode, got {len(try_nodes)}"
        return try_nodes[0]

    # --- try body wiring ---

    def test_try_body_input_edges(self):
        """Try body receives its inputs via input_edges."""

        def wf(x, y):
            try:
                z = my_add(x, y)
            except ValueError:
                z = my_mul(x, y)
            return z

        tn = self._get_try_node(wf)
        try_x = edge_models.TargetHandle(node="try_body", port="x")
        try_y = edge_models.TargetHandle(node="try_body", port="y")
        self.assertIn(try_x, tn.input_edges)
        self.assertIn(try_y, tn.input_edges)
        self.assertEqual(tn.input_edges[try_x].port, "x")
        self.assertEqual(tn.input_edges[try_y].port, "y")

    # --- except body wiring ---

    def test_except_body_input_edges(self):
        """Except body receives its inputs via input_edges."""

        def wf(x, y):
            try:
                z = my_add(x, y)
            except ValueError:
                z = my_mul(x, y)
            return z

        tn = self._get_try_node(wf)
        exc_x = edge_models.TargetHandle(node="except_body_0", port="x")
        exc_y = edge_models.TargetHandle(node="except_body_0", port="y")
        self.assertIn(exc_x, tn.input_edges)
        self.assertIn(exc_y, tn.input_edges)
        self.assertEqual(tn.input_edges[exc_x].port, "x")
        self.assertEqual(tn.input_edges[exc_y].port, "y")

    def test_multiple_except_bodies_input_edges(self):
        """Each except handler gets its own input edges."""

        def wf(x, y):
            try:
                z = my_add(x, y)
            except ValueError:
                z = my_mul(x, y)
            except TypeError:
                z = identity(x)
            return z

        tn = self._get_try_node(wf)
        self.assertIn(
            edge_models.TargetHandle(node="except_body_0", port="x"), tn.input_edges
        )
        self.assertIn(
            edge_models.TargetHandle(node="except_body_1", port="x"), tn.input_edges
        )

    # --- prospective output edges ---

    def test_prospective_output_edges_single_except(self):
        """Both try and except branches contribute prospective sources."""

        def wf(x, y):
            try:
                z = my_add(x, y)
            except ValueError:
                z = my_mul(x, y)
            return z

        tn = self._get_try_node(wf)
        target = edge_models.OutputTarget(port="z")
        self.assertIn(target, tn.prospective_output_edges)
        sources = tn.prospective_output_edges[target]
        self.assertEqual(len(sources), 2)
        source_nodes = {s.node for s in sources}
        self.assertEqual(source_nodes, {"try_body", "except_body_0"})

    def test_prospective_output_edges_multiple_except(self):
        """All branches contribute prospective sources."""

        def wf(x, y):
            try:
                z = my_add(x, y)
            except ValueError:
                z = my_mul(x, y)
            except TypeError:
                z = identity(x)
            return z

        tn = self._get_try_node(wf)
        target = edge_models.OutputTarget(port="z")
        sources = tn.prospective_output_edges[target]
        self.assertEqual(len(sources), 3)
        source_nodes = {s.node for s in sources}
        self.assertEqual(source_nodes, {"try_body", "except_body_0", "except_body_1"})

    # --- partial branch coverage of outputs ---

    def test_output_only_in_one_branch(self):
        """An output assigned only in the try body still appears, with a single
        prospective source."""

        def wf(x, y):
            try:
                a = my_add(x, y)
                b = my_mul(x, y)  # noqa: F841
            except ValueError:
                a = my_mul(x, y)
            return a

        tn = self._get_try_node(wf)
        a_target = edge_models.OutputTarget(port="a")
        self.assertEqual(len(tn.prospective_output_edges[a_target]), 2)
        b_target = edge_models.OutputTarget(port="b")
        self.assertIn(b_target, tn.prospective_output_edges)
        self.assertEqual(len(tn.prospective_output_edges[b_target]), 1)
        self.assertEqual(tn.prospective_output_edges[b_target][0].node, "try_body")

    # --- input collection ---

    def test_all_branches_contribute_inputs(self):
        """Inputs are collected from try body and all except bodies."""

        def wf(x, y):
            try:
                z = my_add(x, y)
            except ValueError:
                z = identity(x)
            return z

        tn = self._get_try_node(wf)
        self.assertIn("x", tn.inputs)
        self.assertIn("y", tn.inputs)

    def test_except_only_input_appears(self):
        """A symbol consumed only in an except body still appears in inputs."""

        def wf(x, y):
            try:
                z = identity(x)
            except ValueError:
                z = my_add(x, y)
            return z

        tn = self._get_try_node(wf)
        self.assertIn("y", tn.inputs)


# ===================================================================
# TryParser structural properties
# ===================================================================


class TestTryParserStructure(unittest.TestCase):
    def _parse(self, func) -> workflow_model.WorkflowNode:
        return workflow_parser.parse_workflow(func)

    def test_try_node_registered_in_parent(self):
        def wf(x, y):
            try:
                z = my_add(x, y)
            except ValueError:
                z = my_mul(x, y)
            return z

        node = self._parse(wf)
        self.assertIn("try_0", node.nodes)
        self.assertIsInstance(node.nodes["try_0"], try_model.TryNode)

    def test_try_body_label(self):
        def wf(x, y):
            try:
                z = my_add(x, y)
            except ValueError:
                z = my_mul(x, y)
            return z

        tn = self._parse(wf).nodes["try_0"]
        self.assertEqual(tn.try_node.label, try_parser.TRY_BODY_LABEL)

    def test_except_body_labels(self):
        def wf(x, y):
            try:
                z = my_add(x, y)
            except ValueError:
                z = my_mul(x, y)
            except TypeError:
                z = identity(x)
            return z

        tn = self._parse(wf).nodes["try_0"]
        self.assertEqual(tn.exception_cases[0].body.label, "except_body_0")
        self.assertEqual(tn.exception_cases[1].body.label, "except_body_1")

    def test_body_nodes_are_workflows(self):
        """All body nodes are WorkflowNodes for uniformity."""

        def wf(x, y):
            try:
                z = my_add(x, y)
            except ValueError:
                z = my_mul(x, y)
            return z

        tn = self._parse(wf).nodes["try_0"]
        self.assertIsInstance(tn.try_node.node, workflow_model.WorkflowNode)
        self.assertIsInstance(
            tn.exception_cases[0].body.node, workflow_model.WorkflowNode
        )

    def test_exception_types_resolved(self):
        """Exception types are resolved to fully qualified names."""

        def wf(x, y):
            try:
                z = my_add(x, y)
            except ValueError:
                z = my_mul(x, y)
            return z

        tn = self._parse(wf).nodes["try_0"]
        self.assertEqual(
            tn.exception_cases[0].exceptions, [versions.VersionInfo.of(ValueError)]
        )

    def test_tuple_exception_types_resolved(self):
        """Tuple exception types are all resolved."""

        def wf(x, y):
            try:
                z = my_add(x, y)
            except (ValueError, TypeError):
                z = my_mul(x, y)
            return z

        tn = self._parse(wf).nodes["try_0"]
        self.assertEqual(
            tn.exception_cases[0].exceptions,
            [
                versions.VersionInfo.of(ValueError),
                versions.VersionInfo.of(TypeError),
            ],
        )

    def test_multiple_outputs(self):
        def wf(x, y):
            try:
                a = my_add(x, y)
                b = my_mul(x, y)
            except ValueError:
                a = my_mul(x, y)
                b = my_add(x, y)
            return a, b

        tn = self._parse(wf).nodes["try_0"]
        self.assertEqual(sorted(tn.outputs), ["a", "b"])

    def test_try_consumes_upstream_node_output(self):
        """Try node can consume sibling output from a preceding node."""

        def wf(a, b):
            x = my_add(a, b)
            try:
                y = my_mul(x, b)
            except ValueError:
                y = my_add(x, b)
            return y

        node = self._parse(wf)
        target = edge_models.TargetHandle(node="try_0", port="x")
        self.assertIn(target, node.edges)
        self.assertEqual(node.edges[target].node, "my_add_0")
        self.assertEqual(node.edges[target].port, "output_0")

    def test_try_output_consumed_by_downstream_node(self):
        """Output of try node feeds a downstream sibling."""

        def wf(x, y):
            try:
                z = my_add(x, y)
            except ValueError:
                z = my_mul(x, y)
            result = identity(z)
            return result

        node = self._parse(wf)
        target = edge_models.TargetHandle(node="identity_0", port="x")
        self.assertIn(target, node.edges)
        self.assertEqual(node.edges[target].node, "try_0")
        self.assertEqual(node.edges[target].port, "z")

    def test_multiple_try_nodes_get_unique_labels(self):
        def wf(x, y):
            try:
                a = my_add(x, y)
            except ValueError:
                a = my_mul(x, y)
            try:
                b = identity(a)
            except TypeError:
                b = my_add(a, y)
            return b

        node = self._parse(wf)
        self.assertIn("try_0", node.nodes)
        self.assertIn("try_1", node.nodes)

    def test_sequential_try_nodes_edge_wiring(self):
        """Output of first try feeds input of second try via sibling edge."""

        def wf(x, y):
            try:
                a = my_add(x, y)
            except ValueError:
                a = my_mul(x, y)
            try:
                b = identity(a)
            except TypeError:
                b = my_add(a, y)
            return b

        node = self._parse(wf)
        target = edge_models.TargetHandle(node="try_1", port="a")
        self.assertIn(target, node.edges)
        self.assertEqual(node.edges[target].node, "try_0")
        self.assertEqual(node.edges[target].port, "a")

    # --- nesting other control flows inside try/except bodies ---

    def test_for_nested_inside_try_body(self):
        """A for-loop inside a try-body produces a ForNode in the body workflow."""

        def wf(xs, y):
            try:
                results = []
                for x in xs:
                    v = identity(x)
                    results.append(v)
                z = identity(results)
            except ValueError:
                z = identity(y)
            return z

        tn = self._parse(wf).nodes["try_0"]
        body = tn.try_node.node
        self.assertIsInstance(body, workflow_model.WorkflowNode)
        for_nodes = [n for n in body.nodes.values() if isinstance(n, for_model.ForNode)]
        self.assertEqual(len(for_nodes), 1)

    def test_while_nested_inside_try_body(self):
        """A while-loop inside a try-body produces a WhileNode in the body workflow."""

        def wf(x, y, bound):
            try:
                while my_condition(x, bound):
                    x = my_add(x, y)
                z = identity(x)
            except ValueError:
                z = identity(y)
            return z

        tn = self._parse(wf).nodes["try_0"]
        body = tn.try_node.node
        self.assertIsInstance(body, workflow_model.WorkflowNode)
        while_nodes = [
            n for n in body.nodes.values() if isinstance(n, while_model.WhileNode)
        ]
        self.assertEqual(len(while_nodes), 1)

    def test_if_nested_inside_try_body(self):
        """An if-node inside a try-body produces an IfNode in the body workflow."""

        def wf(x, y):
            try:
                if my_condition(x, y):  # noqa: SIM108
                    z = my_add(x, y)
                else:
                    z = my_mul(x, y)
            except ValueError:
                z = identity(x)
            return z

        tn = self._parse(wf).nodes["try_0"]
        body = tn.try_node.node
        self.assertIsInstance(body, workflow_model.WorkflowNode)
        if_nodes = [n for n in body.nodes.values() if isinstance(n, if_model.IfNode)]
        self.assertEqual(len(if_nodes), 1)

    def test_try_nested_inside_try_body(self):
        """A try/except inside a try-body produces a TryNode in the body workflow."""

        def wf(x, y):
            try:
                try:
                    z = my_add(x, y)
                except TypeError:
                    z = identity(x)
            except ValueError:
                z = my_mul(x, y)
            return z

        tn = self._parse(wf).nodes["try_0"]
        body = tn.try_node.node
        self.assertIsInstance(body, workflow_model.WorkflowNode)
        inner_try_nodes = [
            n for n in body.nodes.values() if isinstance(n, try_model.TryNode)
        ]
        self.assertEqual(len(inner_try_nodes), 1)

    def test_for_nested_inside_except_body(self):
        """A for-loop inside an except-body produces a ForNode in the except
        workflow."""

        def wf(xs, y):
            try:
                z = identity(y)
            except ValueError:
                results = []
                for x in xs:
                    v = identity(x)
                    results.append(v)
                z = identity(results)
            return z

        tn = self._parse(wf).nodes["try_0"]
        except_body = tn.exception_cases[0].body.node
        self.assertIsInstance(except_body, workflow_model.WorkflowNode)
        for_nodes = [
            n for n in except_body.nodes.values() if isinstance(n, for_model.ForNode)
        ]
        self.assertEqual(len(for_nodes), 1)

    def test_while_nested_inside_except_body(self):
        """A while-loop inside an except-body produces a WhileNode in the except
        workflow."""

        def wf(x, y, bound):
            try:
                z = identity(y)
            except ValueError:
                while my_condition(x, bound):
                    x = my_add(x, y)
                z = identity(x)
            return z

        tn = self._parse(wf).nodes["try_0"]
        except_body = tn.exception_cases[0].body.node
        self.assertIsInstance(except_body, workflow_model.WorkflowNode)
        while_nodes = [
            n
            for n in except_body.nodes.values()
            if isinstance(n, while_model.WhileNode)
        ]
        self.assertEqual(len(while_nodes), 1)

    def test_if_nested_inside_except_body(self):
        """An if-node inside an except-body produces an IfNode in the except
        workflow."""

        def wf(x, y):
            try:
                z = identity(x)
            except ValueError:
                if my_condition(x, y):  # noqa: SIM108
                    z = my_add(x, y)
                else:
                    z = my_mul(x, y)
            return z

        tn = self._parse(wf).nodes["try_0"]
        except_body = tn.exception_cases[0].body.node
        self.assertIsInstance(except_body, workflow_model.WorkflowNode)
        if_nodes = [
            n for n in except_body.nodes.values() if isinstance(n, if_model.IfNode)
        ]
        self.assertEqual(len(if_nodes), 1)


# ===================================================================
# Round-trip serialisation
# ===================================================================


class TestTryNodeRoundTrip(unittest.TestCase):
    def test_simple_try_except_round_trip(self):
        def wf(x, y):
            try:
                z = my_add(x, y)
            except ValueError:
                z = my_mul(x, y)
            return z

        tn = workflow_parser.parse_workflow(wf).nodes["try_0"]
        for mode in ["json", "python"]:
            with self.subTest(mode=mode):
                dumped = tn.model_dump(mode=mode)
                restored = try_model.TryNode.model_validate(dumped)
                self.assertEqual(tn, restored)

    def test_multi_except_round_trip(self):
        def wf(x, y):
            try:
                z = my_add(x, y)
            except ValueError:
                z = my_mul(x, y)
            except TypeError:
                z = identity(x)
            return z

        tn = workflow_parser.parse_workflow(wf).nodes["try_0"]
        for mode in ["json", "python"]:
            with self.subTest(mode=mode):
                dumped = tn.model_dump(mode=mode)
                restored = try_model.TryNode.model_validate(dumped)
                self.assertEqual(tn, restored)

    def test_tuple_exceptions_round_trip(self):
        def wf(x, y):
            try:
                z = my_add(x, y)
            except (ValueError, TypeError):
                z = my_mul(x, y)
            return z

        tn = workflow_parser.parse_workflow(wf).nodes["try_0"]
        for mode in ["json", "python"]:
            with self.subTest(mode=mode):
                dumped = tn.model_dump(mode=mode)
                restored = try_model.TryNode.model_validate(dumped)
                self.assertEqual(tn, restored)

    def test_workflow_with_try_round_trip(self):
        """The whole workflow containing a try-node survives round-trip."""

        def wf(x, y):
            try:
                z = my_add(x, y)
            except ValueError:
                z = my_mul(x, y)
            result = identity(z)
            return result

        node = workflow_parser.parse_workflow(wf)
        for mode in ["json", "python"]:
            with self.subTest(mode=mode):
                dumped = node.model_dump(mode=mode)
                restored = workflow_model.WorkflowNode.model_validate(dumped)
                self.assertEqual(node, restored)

    def test_multi_output_try_round_trip(self):
        def wf(x, y):
            try:
                a = my_add(x, y)
                b = my_mul(x, y)
            except ValueError:
                a = my_mul(x, y)
                b = my_add(x, y)
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


class TestTryParserVersionPropagation(unittest.TestCase):
    """Version scraping/constraints propagate into try/except body child nodes."""

    def _pkg(self) -> str:
        return undecorated_identity.__module__.split(".")[0]

    def test_version_scraping_propagates_into_try_body(self):
        """Undecorated child inside a try body receives the scraping map."""
        custom = "10.20.30"

        def wf(x, y):
            try:
                z = undecorated_identity(x)
            except ValueError:
                z = identity(y)
            return z

        node = workflow_parser.parse_workflow(
            wf, version_scraping={self._pkg(): lambda _: custom}
        )
        try_node = node.nodes["try_0"]
        body = try_node.try_node.node
        child = body.nodes["undecorated_identity_0"]
        self.assertEqual(child.source.version, custom)

    def test_version_scraping_propagates_into_except_body(self):
        """Undecorated child inside an except body receives the scraping map."""
        custom = "20.30.40"

        def wf(x, y):
            try:
                z = identity(x)
            except ValueError:
                z = undecorated_identity(y)
            return z

        node = workflow_parser.parse_workflow(
            wf, version_scraping={self._pkg(): lambda _: custom}
        )
        try_node = node.nodes["try_0"]
        except_body = try_node.exception_cases[0].body.node
        child = except_body.nodes["undecorated_identity_0"]
        self.assertEqual(child.source.version, custom)

    def test_version_scraping_does_not_override_prebuilt_recipe(self):
        """A function already decorated with @atomic keeps its own recipe."""
        custom = "99.0.0"

        def wf(x, y):
            try:
                z = identity(x)
            except ValueError:
                z = undecorated_identity(y)
            return z

        node = workflow_parser.parse_workflow(
            wf, version_scraping={self._pkg(): lambda _: custom}
        )
        try_node = node.nodes["try_0"]
        # Pre-decorated child in try body keeps its own version
        try_child = try_node.try_node.node.nodes["identity_0"]
        self.assertNotEqual(try_child.source.version, custom)
        # Undecorated child in except body picks up custom version
        except_child = try_node.exception_cases[0].body.node.nodes[
            "undecorated_identity_0"
        ]
        self.assertEqual(except_child.source.version, custom)

    def test_version_constraints_propagate_to_children(self):
        def wf(x, y):
            try:
                z = undecorated_identity(x)
            except ValueError:
                z = undecorated_identity(y)
            return z

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(wf, require_version=True)
        self.assertIn("Could not find a version", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
