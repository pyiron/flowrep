import unittest
from typing import Literal

import pydantic

from flowrep import base_models, edge_models, std
from flowrep.drawing import prospective
from flowrep.prospective import (
    constant_recipe,
    workflow_recipe,
)

from flowrep_static import library


def _by_path(graph, path):
    for node in graph.walk():
        if node.path == path:
            return node
    raise AssertionError(f"no node at {path!r}; have {[n.path for n in graph.walk()]}")


class _UnrecognizedRecipe(base_models.NodeRecipe):
    """A well-formed recipe of a type the drawing builder has no branch for."""

    type: Literal[base_models.RecipeElementType.ATOMIC] = pydantic.Field(
        default=base_models.RecipeElementType.ATOMIC, frozen=True
    )

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class TestDepthValidation(unittest.TestCase):
    def test_negative_depth_rejected(self):
        with self.assertRaises(ValueError):
            prospective.build(std.neg.flowrep_recipe, depth=-1)


class TestAtomic(unittest.TestCase):
    def setUp(self):
        self.graph = prospective.build(std.neg.flowrep_recipe)

    def test_root_path_is_empty(self):
        self.assertEqual(self.graph.path, "")

    def test_atomic_is_always_a_leaf(self):
        self.assertTrue(self.graph.is_leaf)

    def test_kind(self):
        self.assertEqual(self.graph.kind, base_models.RecipeElementType.ATOMIC)

    def test_ports(self):
        self.assertEqual([p.label for p in self.graph.inputs], ["a"])
        self.assertEqual([p.label for p in self.graph.outputs], ["negative"])

    def test_subtitle_is_left_truncated_qualified_name(self):
        self.assertTrue(self.graph.subtitle.endswith("neg"))

    def test_no_prospective_hints(self):
        """Recipes carry no annotations; we do not import references to find any."""
        self.assertTrue(all(p.hint is None for p in self.graph.inputs))


class TestDefaultsAreItalicised(unittest.TestCase):
    def test_input_with_default_flagged(self):
        graph = prospective.build(library.increment.flowrep_recipe)
        flags = {p.label: p.has_default for p in graph.inputs}
        self.assertFalse(flags["x"])
        self.assertTrue(flags["step"])


class TestConstant(unittest.TestCase):
    def test_subtitle_is_the_repr(self):
        graph = prospective.build(constant_recipe.ConstantRecipe(constant=42))
        self.assertEqual(graph.subtitle, "42")

    def test_single_output_no_inputs(self):
        graph = prospective.build(constant_recipe.ConstantRecipe(constant=42))
        self.assertEqual(graph.inputs, ())
        self.assertEqual([p.label for p in graph.outputs], ["constant"])


class TestWorkflow(unittest.TestCase):
    def setUp(self):
        self.recipe = library.simple_workflow.flowrep_recipe
        self.graph = prospective.build(self.recipe, depth=0)

    def test_children_present_at_depth_zero(self):
        self.assertEqual(
            sorted(c.path for c in self.graph.children), sorted(self.recipe.nodes)
        )

    def test_children_are_leaves_at_depth_zero(self):
        self.assertTrue(all(c.is_leaf for c in self.graph.children))

    def test_edge_count_matches_recipe(self):
        expected = (
            len(self.recipe.input_edges)
            + len(self.recipe.edges)
            + len(self.recipe.output_edges)
        )
        self.assertEqual(len(self.graph.edges), expected)

    def test_input_edge_sources_the_parent(self):
        """A parent-input source is encoded as the empty node path."""
        target, source = next(iter(self.recipe.input_edges.items()))
        match = [
            e
            for e in self.graph.edges
            if e.target.node_path == target.node and e.target.port == target.port
        ]
        self.assertEqual(len(match), 1)
        self.assertEqual(match[0].source.node_path, "")
        self.assertEqual(match[0].source.port, source.port)

    def test_nothing_is_conditional(self):
        self.assertTrue(all(not e.conditional for e in self.graph.edges))


class TestNestedDepth(unittest.TestCase):
    """A workflow whose child is itself a workflow."""

    def setUp(self):
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

    def test_depth_zero_leaves_the_child_closed(self):
        graph = prospective.build(self.recipe, depth=0)
        self.assertTrue(_by_path(graph, "inner").is_leaf)

    def test_depth_one_opens_the_child(self):
        graph = prospective.build(self.recipe, depth=1)
        self.assertFalse(_by_path(graph, "inner").is_leaf)

    def test_grandchild_paths_are_lexical(self):
        graph = prospective.build(self.recipe, depth=1)
        inner = _by_path(graph, "inner")
        for child in inner.children:
            with self.subTest(path=child.path):
                self.assertTrue(child.path.startswith("inner."))

    def test_depth_one_grandchildren_are_leaves(self):
        graph = prospective.build(self.recipe, depth=1)
        self.assertTrue(all(c.is_leaf for c in _by_path(graph, "inner").children))


class TestOutputPassthrough(unittest.TestCase):
    """An output_edges entry sourced from the parent's own input, not a child."""

    def test_passthrough_source_is_the_parent_input(self):
        recipe = workflow_recipe.WorkflowRecipe(
            inputs=["a"],
            outputs=["passthrough"],
            nodes={},
            input_edges={},
            edges={},
            output_edges={
                edge_models.OutputTarget(port="passthrough"): edge_models.InputSource(
                    port="a"
                ),
            },
        )
        graph = prospective.build(recipe, depth=0)
        self.assertEqual(len(graph.edges), 1)
        edge = graph.edges[0]
        self.assertEqual(edge.source.node_path, "")
        self.assertEqual(edge.source.io_type, base_models.IOTypes.INPUTS)
        self.assertEqual(edge.source.port, "a")
        self.assertEqual(edge.target.node_path, "")
        self.assertEqual(edge.target.io_type, base_models.IOTypes.OUTPUTS)
        self.assertEqual(edge.target.port, "passthrough")


class TestUnrecognizedRecipe(unittest.TestCase):
    def test_unrecognized_recipe_subclass_raises(self):
        """A NodeRecipe the dispatch has no branch for still falls through."""
        with self.assertRaises(TypeError) as ctx:
            prospective.build(_UnrecognizedRecipe(inputs=[], outputs=[]))
        self.assertIn("Unrecognized recipe type", str(ctx.exception))
