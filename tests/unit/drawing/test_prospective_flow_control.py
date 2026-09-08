import unittest

from pyiron_snippets import versions

from flowrep import edge_models, std
from flowrep.drawing import prospective
from flowrep.prospective import (
    for_recipe,
    helper_models,
    if_recipe,
    try_recipe,
    while_recipe,
)

from flowrep_static import library


def _by_path(graph, path):
    for node in graph.walk():
        if node.path == path:
            return node
    raise AssertionError(f"no node at {path!r}; have {[n.path for n in graph.walk()]}")


def _make_for() -> for_recipe.ForEachRecipe:
    return for_recipe.ForEachRecipe(
        inputs=["xs"],
        outputs=["ys", "used"],
        body_node=helper_models.LabeledRecipe(
            label="body", recipe=std.neg.flowrep_recipe
        ),
        input_edges={
            edge_models.TargetHandle(node="body", port="a"): edge_models.InputSource(
                port="xs"
            ),
        },
        output_edges={
            edge_models.OutputTarget(port="ys"): edge_models.SourceHandle(
                node="body", port="negative"
            ),
            edge_models.OutputTarget(port="used"): edge_models.InputSource(port="xs"),
        },
        nested_ports=["a"],
    )


def _make_while() -> while_recipe.WhileRecipe:
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


def _make_if() -> if_recipe.IfRecipe:
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


def _make_try() -> try_recipe.TryRecipe:
    return try_recipe.TryRecipe(
        inputs=["x", "y"],
        outputs=["out"],
        try_node=helper_models.LabeledRecipe(
            label="attempt", recipe=library.raises_custom.flowrep_recipe
        ),
        exception_cases=[
            helper_models.ExceptionCase(
                exceptions=[versions.VersionInfo.of(library.MyCustomException)],
                body=helper_models.LabeledRecipe(
                    label="handler", recipe=library.combine.flowrep_recipe
                ),
            )
        ],
        input_edges={
            edge_models.TargetHandle(node="attempt", port="x"): edge_models.InputSource(
                port="x"
            ),
            edge_models.TargetHandle(node="attempt", port="y"): edge_models.InputSource(
                port="y"
            ),
            edge_models.TargetHandle(node="handler", port="a"): edge_models.InputSource(
                port="x"
            ),
            edge_models.TargetHandle(node="handler", port="b"): edge_models.InputSource(
                port="y"
            ),
        },
        prospective_output_edges={
            edge_models.OutputTarget(port="out"): [
                edge_models.SourceHandle(
                    node="attempt", port=library.raises_custom.flowrep_recipe.outputs[0]
                ),
                edge_models.SourceHandle(
                    node="handler", port=library.combine.flowrep_recipe.outputs[0]
                ),
            ]
        },
    )


class TestForEach(unittest.TestCase):
    def setUp(self):
        self.graph = prospective.build(_make_for(), depth=0)

    def test_body_expands_once(self):
        self.assertEqual([c.path for c in self.graph.children], ["body"])

    def test_body_display_label_signals_multiplicity(self):
        self.assertEqual(self.graph.children[0].label, "body_n")

    def test_path_keeps_the_real_label(self):
        """Identity stays honest even though the display label is decorated."""
        self.assertEqual(self.graph.children[0].path, "body")

    def test_nested_port_badged(self):
        badges = {p.label: p.badge for p in self.graph.children[0].inputs}
        self.assertEqual(badges["a"], "nested")

    def test_transferred_output_links_input_to_output(self):
        matches = [
            e
            for e in self.graph.edges
            if e.source.node_path == ""
            and e.target.node_path == ""
            and e.target.port == "used"
        ]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].source.port, "xs")


class TestForEachZipped(unittest.TestCase):
    def test_zipped_port_badged(self):
        recipe = _make_for().model_copy(
            update={"nested_ports": [], "zipped_ports": ["a"]}
        )
        graph = prospective.build(recipe, depth=0)
        badges = {p.label: p.badge for p in graph.children[0].inputs}
        self.assertEqual(badges["a"], "zipped")


class TestWhile(unittest.TestCase):
    def setUp(self):
        self.graph = prospective.build(_make_while(), depth=0)

    def test_condition_and_body_both_present(self):
        self.assertEqual(sorted(c.path for c in self.graph.children), ["body", "cond"])

    def test_iteration_suffix_on_display_labels(self):
        self.assertEqual(
            sorted(c.label for c in self.graph.children), ["body_i", "cond_i"]
        )

    def test_condition_output_badged(self):
        cond = _by_path(self.graph, "cond")
        self.assertEqual([p.badge for p in cond.outputs], ["test"])

    def test_inferred_loop_back_edges_present(self):
        back = [
            e
            for e in self.graph.edges
            if e.source.node_path == "body" and e.target.node_path in ("body", "cond")
        ]
        self.assertEqual(len(back), 2)

    def test_fallback_edge_is_conditional(self):
        fallback = [
            e
            for e in self.graph.edges
            if e.source.node_path == "" and e.target.node_path == "" and e.conditional
        ]
        self.assertEqual(len(fallback), 1)
        self.assertEqual(fallback[0].source.port, "x")
        self.assertEqual(fallback[0].target.port, "x")


class TestIf(unittest.TestCase):
    def setUp(self):
        self.graph = prospective.build(_make_if(), depth=0)

    def test_all_branch_nodes_present(self):
        self.assertEqual(
            sorted(c.path for c in self.graph.children),
            ["body0", "cond0", "otherwise"],
        )

    def test_groups_pair_condition_with_body(self):
        groups = {g.label: g.members for g in self.graph.groups}
        self.assertEqual(sorted(groups["case 0"]), ["body0", "cond0"])

    def test_else_group(self):
        groups = {g.label: g.members for g in self.graph.groups}
        self.assertEqual(groups["else"], ("otherwise",))

    def test_group_order_follows_declaration(self):
        self.assertEqual([g.label for g in self.graph.groups], ["case 0", "else"])

    def test_candidate_output_edges_are_conditional(self):
        to_out = [e for e in self.graph.edges if e.target.node_path == ""]
        self.assertEqual(len(to_out), 2)
        self.assertTrue(all(e.conditional for e in to_out))

    def test_condition_output_badged(self):
        self.assertEqual(
            [p.badge for p in _by_path(self.graph, "cond0").outputs], ["test"]
        )


class TestTry(unittest.TestCase):
    def setUp(self):
        self.graph = prospective.build(_make_try(), depth=0)

    def test_try_and_handler_present(self):
        self.assertEqual(
            sorted(c.path for c in self.graph.children), ["attempt", "handler"]
        )

    def test_try_group(self):
        groups = {g.label: g.members for g in self.graph.groups}
        self.assertEqual(groups["try"], ("attempt",))

    def test_exception_group_names_the_exception(self):
        labels = [g.label for g in self.graph.groups]
        self.assertTrue(
            any(label.startswith("except ") for label in labels), msg=str(labels)
        )

    def test_candidate_output_edges_are_conditional(self):
        to_out = [e for e in self.graph.edges if e.target.node_path == ""]
        self.assertEqual(len(to_out), 2)
        self.assertTrue(all(e.conditional for e in to_out))


class TestFlowControlDepth(unittest.TestCase):
    def test_for_body_is_a_leaf_when_atomic(self):
        graph = prospective.build(_make_for(), depth=1)
        self.assertTrue(_by_path(graph, "body").is_leaf)
