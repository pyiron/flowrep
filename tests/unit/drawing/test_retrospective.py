import unittest

from flowrep import std, wfms
from flowrep.drawing import retrospective
from flowrep.retrospective import datastructures

from flowrep_static import library


def _by_path(graph, path):
    for node in graph.walk():
        if node.path == path:
            return node
    raise AssertionError(f"no node at {path!r}; have {[n.path for n in graph.walk()]}")


class TestDepthValidation(unittest.TestCase):
    def test_negative_depth_rejected(self):
        data = datastructures.recipe2data(std.neg.flowrep_recipe)
        with self.assertRaises(ValueError):
            retrospective.build(data, depth=-1)


class TestAtomicData(unittest.TestCase):
    def setUp(self):
        self.graph = retrospective.build(
            datastructures.recipe2data(library.decrement.flowrep_recipe)
        )

    def test_is_a_leaf(self):
        self.assertTrue(self.graph.is_leaf)

    def test_annotation_hint_present(self):
        """Unlike prospective, retrospective ports carry annotations."""
        self.assertEqual([p.hint for p in self.graph.inputs], ["int"])

    def test_subtitle_reads_through_the_recipe(self):
        self.assertTrue(self.graph.subtitle.endswith("decrement"))


class TestDefaults(unittest.TestCase):
    def test_default_flagged_from_the_data_port(self):
        graph = retrospective.build(
            datastructures.recipe2data(library.increment.flowrep_recipe)
        )
        flags = {p.label: p.has_default for p in graph.inputs}
        self.assertFalse(flags["x"])
        self.assertTrue(flags["step"])


class TestDagData(unittest.TestCase):
    def setUp(self):
        self.data = datastructures.recipe2data(library.simple_workflow.flowrep_recipe)
        self.graph = retrospective.build(self.data, depth=0)

    def test_children_match_the_data(self):
        self.assertEqual(
            sorted(c.path for c in self.graph.children), sorted(self.data.nodes)
        )

    def test_edges_match_the_data(self):
        expected = (
            len(self.data.input_edges)
            + len(self.data.edges)
            + len(self.data.output_edges)
        )
        self.assertEqual(len(self.graph.edges), expected)

    def test_no_note_when_edges_exist(self):
        self.assertIsNone(self.graph.note)

    def test_output_ports_have_no_default(self):
        self.assertTrue(all(not p.has_default for p in self.graph.outputs))

    def test_depth_zero_children_are_leaves(self):
        self.assertTrue(all(c.is_leaf for c in self.graph.children))

    def test_depth_zero_children_have_no_note(self):
        """A leaf never expanded, so it must never receive the missing-edges note."""
        self.assertTrue(all(c.note is None for c in self.graph.children))

    def test_input_edge_sources_the_parent(self):
        """A parent-input source is encoded as the empty node path."""
        target, source = next(iter(self.data.input_edges.items()))
        match = [
            e
            for e in self.graph.edges
            if e.target.node_path == target.node and e.target.port == target.port
        ]
        self.assertEqual(len(match), 1)
        self.assertEqual(match[0].source.node_path, "")
        self.assertEqual(match[0].source.port, source.port)

    def test_output_edge_targets_the_parent(self):
        """An output edge targets the enclosing node's own output (empty node path)."""
        target, source = next(iter(self.data.output_edges.items()))
        match = [
            e
            for e in self.graph.edges
            if e.target.node_path == "" and e.target.port == target.port
        ]
        self.assertEqual(len(match), 1)
        if source.node is None:
            self.assertEqual(match[0].source.node_path, "")
        else:
            self.assertEqual(match[0].source.node_path, source.node)

    def test_nothing_is_conditional(self):
        self.assertTrue(all(not e.conditional for e in self.graph.edges))


class TestDagNestedDepth(unittest.TestCase):
    """A workflow whose child is itself a workflow, with both children and its own edges."""

    def setUp(self):
        from flowrep import edge_models
        from flowrep.prospective import workflow_recipe

        inner = library.simple_workflow.flowrep_recipe
        self.recipe = workflow_recipe.WorkflowRecipe(
            inputs=["p", "q"],
            outputs=["r"],
            nodes={"inner": inner},
            input_edges={
                edge_models.TargetHandle(
                    node="inner", port="a"
                ): edge_models.InputSource(port="p"),
                edge_models.TargetHandle(
                    node="inner", port="b"
                ): edge_models.InputSource(port="q"),
            },
            edges={},
            output_edges={
                edge_models.OutputTarget(port="r"): edge_models.SourceHandle(
                    node="inner", port=inner.outputs[0]
                )
            },
        )
        self.data = datastructures.recipe2data(self.recipe)

    def test_depth_zero_leaves_the_child_closed(self):
        graph = retrospective.build(self.data, depth=0)
        self.assertTrue(_by_path(graph, "inner").is_leaf)

    def test_depth_one_opens_the_child(self):
        graph = retrospective.build(self.data, depth=1)
        self.assertFalse(_by_path(graph, "inner").is_leaf)

    def test_grandchild_paths_are_lexical(self):
        graph = retrospective.build(self.data, depth=1)
        inner = _by_path(graph, "inner")
        for child in inner.children:
            with self.subTest(path=child.path):
                self.assertTrue(child.path.startswith("inner."))

    def test_inner_has_edges_and_no_note(self):
        """The inner workflow has both children and edges: no missing-edges note."""
        graph = retrospective.build(self.data, depth=1)
        inner = _by_path(graph, "inner")
        self.assertTrue(inner.edges)
        self.assertTrue(inner.children)
        self.assertIsNone(inner.note)


class TestOutputEdgePassthrough(unittest.TestCase):
    """An output edge whose source is a parent-input passthrough, not a child output."""

    def setUp(self):
        from flowrep import edge_models
        from flowrep.prospective import workflow_recipe

        recipe = workflow_recipe.WorkflowRecipe(
            inputs=["a"],
            outputs=["passthrough"],
            nodes={"inc": library.increment.flowrep_recipe},
            input_edges={
                edge_models.TargetHandle(node="inc", port="x"): edge_models.InputSource(
                    port="a"
                ),
            },
            edges={},
            output_edges={
                edge_models.OutputTarget(port="passthrough"): edge_models.InputSource(
                    port="a"
                ),
            },
        )
        self.data = datastructures.recipe2data(recipe)
        self.graph = retrospective.build(self.data, depth=0)

    def test_passthrough_source_is_the_parent_input(self):
        match = [
            e
            for e in self.graph.edges
            if e.target.node_path == "" and e.target.port == "passthrough"
        ]
        self.assertEqual(len(match), 1)
        self.assertEqual(match[0].source.node_path, "")
        self.assertEqual(match[0].source.io_type.value, "inputs")
        self.assertEqual(match[0].source.port, "a")


class TestExecutedFlowControl(unittest.TestCase):
    """A run for-node has instance children but, per the known limitation, no edges."""

    def setUp(self):
        self.data = wfms.run_recipe(_for_recipe(), xs=[1, 2, 3])
        self.graph = retrospective.build(self.data, depth=0)

    def test_instance_children_present(self):
        self.assertEqual(len(self.graph.children), 3)

    def test_instance_paths_are_lexical(self):
        self.assertEqual(
            sorted(c.path for c in self.graph.children), ["body_0", "body_1", "body_2"]
        )

    def test_note_explains_the_missing_edges(self):
        self.assertEqual(self.graph.note, "(no recorded edges)")

    def test_no_edges_drawn(self):
        self.assertEqual(self.graph.edges, ())


class TestUnexpandedCompositeHasNoNote(unittest.TestCase):
    """A composite that never expands (depth < 0 branch skipped) must not get the note."""

    def test_depth_zero_but_no_children_no_note(self):
        data = wfms.run_recipe(_for_recipe(), xs=[])
        graph = retrospective.build(data, depth=0)
        self.assertEqual(graph.children, ())
        self.assertIsNone(graph.note)


def _for_recipe():
    from flowrep import edge_models
    from flowrep.prospective import for_recipe, helper_models

    return for_recipe.ForEachRecipe(
        inputs=["xs"],
        outputs=["ys"],
        body_node=helper_models.LabeledRecipe(
            label="body", recipe=std.neg.flowrep_recipe
        ),
        input_edges={
            edge_models.TargetHandle(node="body", port="a"): edge_models.InputSource(
                port="xs"
            )
        },
        output_edges={
            edge_models.OutputTarget(port="ys"): edge_models.SourceHandle(
                node="body", port="negative"
            )
        },
        nested_ports=["a"],
    )
