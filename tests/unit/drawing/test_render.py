import shutil
import unittest

from flowrep import base_models, edge_models, std, wfms
from flowrep.drawing import model, prospective, render, retrospective, style
from flowrep.prospective import (
    constant_recipe,
    for_recipe,
    helper_models,
    if_recipe,
    while_recipe,
    workflow_recipe,
)

from flowrep_static import library

try:
    import graphviz  # noqa: F401

    _has_graphviz = True
except ImportError:
    _has_graphviz = False


def _while_recipe():
    return while_recipe.WhileRecipe(
        inputs=["x"],
        outputs=["x"],
        case=helper_models.ConditionalCase(
            condition=helper_models.LabeledRecipe(
                label="cond", recipe=library.is_positive.flowrep_recipe
            ),
            body=helper_models.LabeledRecipe(
                label="body", recipe=library.loop_inc.flowrep_recipe
            ),
        ),
        input_edges={
            edge_models.TargetHandle(node="cond", port="n"): edge_models.InputSource(
                port="x"
            ),
            edge_models.TargetHandle(node="body", port="x"): edge_models.InputSource(
                port="x"
            ),
        },
        output_edges={
            edge_models.OutputTarget(port="x"): edge_models.SourceHandle(
                node="body", port="y"
            )
        },
    )


def _for_recipe():
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


def _if_recipe():
    return if_recipe.IfRecipe(
        inputs=["n"],
        outputs=["out"],
        cases=[
            helper_models.ConditionalCase(
                condition=helper_models.LabeledRecipe(
                    label="cond0", recipe=library.is_positive.flowrep_recipe
                ),
                body=helper_models.LabeledRecipe(
                    label="body0", recipe=library.increment.flowrep_recipe
                ),
            )
        ],
        else_case=helper_models.LabeledRecipe(
            label="otherwise", recipe=library.decrement.flowrep_recipe
        ),
        input_edges={
            edge_models.TargetHandle(node="cond0", port="n"): edge_models.InputSource(
                port="n"
            ),
            edge_models.TargetHandle(node="body0", port="x"): edge_models.InputSource(
                port="n"
            ),
            edge_models.TargetHandle(
                node="otherwise", port="x"
            ): edge_models.InputSource(port="n"),
        },
        prospective_output_edges={
            edge_models.OutputTarget(port="out"): [
                edge_models.SourceHandle(node="body0", port="output_0"),
                edge_models.SourceHandle(node="otherwise", port="output_0"),
            ]
        },
    )


@unittest.skipUnless(_has_graphviz, "graphviz not installed")
class TestLeafRendering(unittest.TestCase):
    def setUp(self):
        self.source = render.render(prospective.build(std.neg.flowrep_recipe)).source

    def test_is_left_to_right(self):
        self.assertIn("rankdir=LR", self.source)

    def test_node_label_present(self):
        self.assertIn("neg", self.source)

    def test_ports_are_addressable(self):
        self.assertIn('PORT="i_a"', self.source)
        self.assertIn('PORT="o_negative"', self.source)

    def test_atomic_fill_colour_used(self):
        fill, _ = style.NODE_PALETTE[base_models.RecipeElementType.ATOMIC]
        self.assertIn(fill, self.source)


@unittest.skipUnless(_has_graphviz, "graphviz not installed")
class TestCompositeRendering(unittest.TestCase):
    def setUp(self):
        graph = prospective.build(library.simple_workflow.flowrep_recipe, depth=0)
        self.source = render.render(graph).source

    def test_root_becomes_a_cluster(self):
        self.assertIn("subgraph cluster_", self.source)

    def test_child_ids_appear_in_source(self):
        """A single-segment lexical path is a valid DOT identifier and Graphviz
        does not quote it unnecessarily; quoting is exercised separately for a
        genuinely nested (dotted) path."""
        for child in prospective.build(
            library.simple_workflow.flowrep_recipe, depth=0
        ).children:
            with self.subTest(path=child.path):
                self.assertIn(child.path, self.source)


@unittest.skipUnless(_has_graphviz, "graphviz not installed")
class TestNestedIdsAreQuoted(unittest.TestCase):
    def test_dotted_child_path_is_quoted(self):
        """A dotted lexical path is not a bare DOT identifier, so Graphviz quotes it."""
        inner = library.simple_workflow.flowrep_recipe
        outer = workflow_recipe.WorkflowRecipe(
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
        source = render.render(prospective.build(outer, depth=1)).source
        self.assertIn('"inner.typed_add_0"', source)


@unittest.skipUnless(_has_graphviz, "graphviz not installed")
class TestCycleHandling(unittest.TestCase):
    def test_back_edges_are_unconstrained(self):
        """While-loop feedback would otherwise destroy the rankdir=LR layout."""
        source = render.render(prospective.build(_while_recipe(), depth=0)).source
        self.assertIn("constraint=false", source)

    def test_acyclic_composite_has_no_unconstrained_edges(self):
        """A plain DAG must not be mistaken for a cycle just because its own
        input and its own output share the empty ``node_path`` placeholder."""
        source = render.render(
            prospective.build(library.simple_workflow.flowrep_recipe, depth=0)
        ).source
        self.assertNotIn("constraint=false", source)


@unittest.skipUnless(_has_graphviz, "graphviz not installed")
class TestConditionalEdges(unittest.TestCase):
    def test_conditional_edges_dashed(self):
        source = render.render(prospective.build(_while_recipe(), depth=0)).source
        self.assertIn("style=dashed", source)


@unittest.skipUnless(_has_graphviz, "graphviz not installed")
class TestGroupOrdering(unittest.TestCase):
    def test_groups_emit_in_reverse_declaration_order(self):
        """Under rankdir=LR, Graphviz stacks same-rank clusters bottom-up in
        declaration order, so the first-declared group ("case 0") must be the
        *last* one written to source for it to land on top."""
        source = render.render(prospective.build(_if_recipe(), depth=0)).source
        self.assertLess(source.index("else"), source.index("case 0"))


@unittest.skipUnless(_has_graphviz, "graphviz not installed")
class TestEscaping(unittest.TestCase):
    def test_special_characters_in_subtitle_are_escaped(self):
        graph = prospective.build(constant_recipe.ConstantRecipe(constant="<a & b>"))
        source = render.render(graph).source
        self.assertNotIn("<a & b>", source)
        self.assertIn("&lt;a &amp; b&gt;", source)


@unittest.skipUnless(_has_graphviz, "graphviz not installed")
class TestPortPaddingBranches(unittest.TestCase):
    def test_more_inputs_than_outputs_pads_output_column(self):
        source = render.render(prospective.build(library.combine.flowrep_recipe)).source
        self.assertIn("<TD></TD>", source)

    def test_more_outputs_than_inputs_pads_input_column(self):
        source = render.render(
            prospective.build(constant_recipe.ConstantRecipe(constant=1))
        ).source
        self.assertIn("<TD></TD>", source)


@unittest.skipUnless(_has_graphviz, "graphviz not installed")
class TestEmptyIoComposite(unittest.TestCase):
    def test_composite_with_no_own_io_draws_no_io_boxes(self):
        recipe = workflow_recipe.WorkflowRecipe(
            inputs=[],
            outputs=[],
            nodes={"c": constant_recipe.ConstantRecipe(constant=1)},
            input_edges={},
            edges={},
            output_edges={},
        )
        source = render.render(prospective.build(recipe, depth=0)).source
        self.assertNotIn("inputs.", source)
        self.assertNotIn("outputs.", source)


@unittest.skipUnless(_has_graphviz, "graphviz not installed")
class TestMultipleOwnOutputsRankSame(unittest.TestCase):
    def test_two_own_outputs_are_rank_aligned(self):
        source = render.render(
            prospective.build(library.autoencoder.flowrep_recipe, depth=0)
        ).source
        self.assertIn("rank=same", source)


@unittest.skipUnless(_has_graphviz, "graphviz not installed")
class TestBadgePreferredOverHint(unittest.TestCase):
    def test_badge_wins_when_both_set(self):
        node = model.DrawNode(
            path="",
            label="x",
            kind=base_models.RecipeElementType.ATOMIC,
            subtitle=None,
            inputs=(model.DrawPort(label="a", hint="int", badge="nested"),),
            outputs=(),
            children=(),
            edges=(),
        )
        source = render.render(node).source
        self.assertIn("nested", source)
        self.assertNotIn(">int<", source)


@unittest.skipUnless(_has_graphviz, "graphviz not installed")
class TestRetrospectiveNote(unittest.TestCase):
    def test_no_recorded_edges_note_appears_in_cluster_label(self):
        """A composite with children but no wiring surfaces an italic note.

        The toy WfMS records actualized edges, so this is not reachable through it.
        The retrospective format does not oblige every WfMS to record them, though,
        so stripping the edges back off a run node stands in for one that doesn't.
        """
        data = wfms.run_recipe(_for_recipe(), xs=[1, 2, 3])
        data.input_edges = {}
        data.edges = {}
        data.output_edges = {}
        graph = retrospective.build(data, depth=0)
        source = render.render(graph).source
        self.assertIn("no recorded edges", source)

    def test_no_note_when_the_wfms_records_its_edges(self):
        data = wfms.run_recipe(_for_recipe(), xs=[1, 2, 3])
        graph = retrospective.build(data, depth=0)
        self.assertNotIn("no recorded edges", render.render(graph).source)


@unittest.skipUnless(_has_graphviz, "graphviz not installed")
class TestActuallyRenders(unittest.TestCase):
    @unittest.skipIf(shutil.which("dot") is None, "Graphviz `dot` binary not installed")
    def test_dot_accepts_the_source(self):
        """The DOT we emit must survive a real Graphviz parse, not just look right."""
        digraph = render.render(prospective.build(_for_recipe(), depth=0))
        self.assertTrue(digraph.pipe(format="svg").startswith(b"<?xml"))

    @unittest.skipIf(shutil.which("dot") is None, "Graphviz `dot` binary not installed")
    def test_while_recipe_dot_accepts_the_source(self):
        """Cyclic (while) topology must also survive a real Graphviz parse."""
        digraph = render.render(prospective.build(_while_recipe(), depth=0))
        self.assertTrue(digraph.pipe(format="svg").startswith(b"<?xml"))

    @unittest.skipIf(shutil.which("dot") is None, "Graphviz `dot` binary not installed")
    def test_if_recipe_dot_accepts_the_source(self):
        """Grouped (if) topology must also survive a real Graphviz parse."""
        digraph = render.render(prospective.build(_if_recipe(), depth=0))
        self.assertTrue(digraph.pipe(format="svg").startswith(b"<?xml"))
