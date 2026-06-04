import dataclasses
import inspect
import typing
import unittest

from pyiron_snippets import versions

from flowrep import edge_models, pysource, retrospective, wfms
from flowrep.nodes import (
    for_recipe,
    helper_models,
    if_recipe,
    try_recipe,
    while_recipe,
    workflow_recipe,
)
from flowrep.parsers import atomic_parser, workflow_parser

from flowrep_static import library, makers


@atomic_parser.atomic
def _pos_only_add(a, b, /):
    """Module-level atomic with positional-only inputs (importable by qualname)."""
    return a + b


# A conditional workflow where the condition node has no underlying python reference
# This also has conditional branches that are straight atomic nodes. This is perfectly
# valid flowrep, but something you can only get by manual construction -- parsed
# workflows currently wrap all bodies in a workflow node, whether the bodies are
# multi-step or not.
# Thus, we turn two assumptions on their head with this recipe
workflow_condition_recipe = workflow_recipe.WorkflowRecipe(
    inputs=["n"],
    outputs=["m"],
    nodes={
        "if_0": if_recipe.IfRecipe(
            inputs=["n"],
            outputs=["m"],
            cases=[
                helper_models.ConditionalCase(
                    condition=helper_models.LabeledRecipe(
                        label="condition",
                        node=workflow_recipe.WorkflowRecipe(
                            inputs=["inp"],
                            outputs=["out"],
                            nodes={},
                            input_edges={},
                            edges={},
                            output_edges={
                                edge_models.OutputTarget(
                                    port="out"
                                ): edge_models.InputSource(port="inp"),
                            },
                        ),
                    ),
                    body=helper_models.LabeledRecipe(
                        label="if_case",
                        node=library.increment.flowrep_recipe,
                    ),
                ),
            ],
            else_case=helper_models.LabeledRecipe(
                label="else_case",
                node=library.decrement.flowrep_recipe,
            ),
            input_edges={
                edge_models.TargetHandle(
                    node="condition", port="inp"
                ): edge_models.InputSource(port="n"),
                edge_models.TargetHandle(
                    node="if_case", port="x"
                ): edge_models.InputSource(port="n"),
                edge_models.TargetHandle(
                    node="else_case", port="x"
                ): edge_models.InputSource(port="n"),
            },
            prospective_output_edges={
                edge_models.OutputTarget(port="m"): [
                    edge_models.SourceHandle(node="if_case", port="output_0"),
                    edge_models.SourceHandle(node="else_case", port="output_0"),
                ]
            },
        )
    },
    input_edges={
        edge_models.TargetHandle(node="if_0", port="n"): edge_models.InputSource(
            port="n"
        )
    },
    edges={},
    output_edges={
        edge_models.OutputTarget(port="m"): edge_models.SourceHandle(
            node="if_0", port="m"
        )
    },
)


# A while-loop whose condition is a reference-free passthrough workflow and whose
# body is a bare atomic node (decrement). Like workflow_condition_recipe, this can
# only be built by hand -- the parser always wraps loop bodies in a workflow node.
workflow_while_recipe = workflow_recipe.WorkflowRecipe(
    inputs=["i"],
    outputs=["i"],
    nodes={
        "while_0": while_recipe.WhileRecipe(
            inputs=["i"],
            outputs=["i"],
            case=helper_models.ConditionalCase(
                condition=helper_models.LabeledRecipe(
                    label="condition",
                    node=workflow_recipe.WorkflowRecipe(
                        inputs=["inp"],
                        outputs=["out"],
                        nodes={},
                        input_edges={},
                        edges={},
                        output_edges={
                            edge_models.OutputTarget(
                                port="out"
                            ): edge_models.InputSource(port="inp"),
                        },
                    ),
                ),
                body=helper_models.LabeledRecipe(
                    label="while_body",
                    node=library.decrement.flowrep_recipe,
                ),
            ),
            input_edges={
                edge_models.TargetHandle(
                    node="condition", port="inp"
                ): edge_models.InputSource(port="i"),
                edge_models.TargetHandle(
                    node="while_body", port="x"
                ): edge_models.InputSource(port="i"),
            },
            output_edges={
                edge_models.OutputTarget(port="i"): edge_models.SourceHandle(
                    node="while_body", port="output_0"
                )
            },
        )
    },
    input_edges={
        edge_models.TargetHandle(node="while_0", port="i"): edge_models.InputSource(
            port="i"
        )
    },
    edges={},
    output_edges={
        edge_models.OutputTarget(port="i"): edge_models.SourceHandle(
            node="while_0", port="i"
        )
    },
)


# A try/except whose try body and except body are bare atomic nodes. Try has no
# condition node, so this isolates the atomic-branch path in _emit_branch.
workflow_try_recipe = workflow_recipe.WorkflowRecipe(
    inputs=["a", "b"],
    outputs=["z"],
    nodes={
        "try_0": try_recipe.TryRecipe(
            inputs=["a", "b"],
            outputs=["z"],
            try_node=helper_models.LabeledRecipe(
                label="try_body",
                node=library.divide.flowrep_recipe,
            ),
            exception_cases=[
                helper_models.ExceptionCase(
                    exceptions=[versions.VersionInfo.of(ZeroDivisionError)],
                    body=helper_models.LabeledRecipe(
                        label="except_body",
                        node=library.identity.flowrep_recipe,
                    ),
                )
            ],
            input_edges={
                edge_models.TargetHandle(
                    node="try_body", port="a"
                ): edge_models.InputSource(port="a"),
                edge_models.TargetHandle(
                    node="try_body", port="b"
                ): edge_models.InputSource(port="b"),
                edge_models.TargetHandle(
                    node="except_body", port="x"
                ): edge_models.InputSource(port="a"),
            },
            prospective_output_edges={
                edge_models.OutputTarget(port="z"): [
                    edge_models.SourceHandle(node="try_body", port="output_0"),
                    edge_models.SourceHandle(node="except_body", port="x"),
                ]
            },
        )
    },
    input_edges={
        edge_models.TargetHandle(node="try_0", port="a"): edge_models.InputSource(
            port="a"
        ),
        edge_models.TargetHandle(node="try_0", port="b"): edge_models.InputSource(
            port="b"
        ),
    },
    edges={},
    output_edges={
        edge_models.OutputTarget(port="z"): edge_models.SourceHandle(
            node="try_0", port="z"
        )
    },
)


# A for-each that maps a bare atomic node (increment) over its `x` port. `step`
# keeps its default and is intentionally unwired.
workflow_for_each_recipe = workflow_recipe.WorkflowRecipe(
    inputs=["xs"],
    outputs=["ys"],
    nodes={
        "for_0": for_recipe.ForEachRecipe(
            inputs=["xs"],
            outputs=["ys"],
            body_node=helper_models.LabeledRecipe(
                label="for_body",
                node=library.increment.flowrep_recipe,
            ),
            input_edges={
                edge_models.TargetHandle(
                    node="for_body", port="x"
                ): edge_models.InputSource(port="xs"),
            },
            output_edges={
                edge_models.OutputTarget(port="ys"): edge_models.SourceHandle(
                    node="for_body", port="output_0"
                )
            },
            nested_ports=["x"],
        )
    },
    input_edges={
        edge_models.TargetHandle(node="for_0", port="xs"): edge_models.InputSource(
            port="xs"
        )
    },
    edges={},
    output_edges={
        edge_models.OutputTarget(port="ys"): edge_models.SourceHandle(
            node="for_0", port="ys"
        )
    },
)


# A for-each over `increment` (body port `x`). Used as a flow-control node where a
# single-callable is required, to prove reverse-render refuses it cleanly.
def _for_each_increment(label_body: str):
    from flowrep.nodes import for_recipe  # local import keeps this near its use

    return for_recipe.ForEachRecipe(
        inputs=["xs"],
        outputs=["ys"],
        body_node=helper_models.LabeledRecipe(
            label=label_body, node=library.increment.flowrep_recipe
        ),
        input_edges={
            edge_models.TargetHandle(
                node=label_body, port="x"
            ): edge_models.InputSource(port="xs"),
        },
        output_edges={
            edge_models.OutputTarget(port="ys"): edge_models.SourceHandle(
                node=label_body, port="output_0"
            )
        },
        nested_ports=["x"],
    )


# An if-node whose CONDITION is a flow-control (for-each) recipe. Hand-built only;
# the parser never produces a flow-control condition.
if_flow_control_condition_recipe = workflow_recipe.WorkflowRecipe(
    inputs=["xs"],
    outputs=["ys"],
    nodes={
        "if_c": if_recipe.IfRecipe(
            inputs=["xs"],
            outputs=["ys"],
            cases=[
                helper_models.ConditionalCase(
                    condition=helper_models.LabeledRecipe(
                        label="flowcond", node=_for_each_increment("c_body")
                    ),
                    condition_output="ys",
                    body=helper_models.LabeledRecipe(
                        label="cbranch", node=_for_each_increment("inc_body")
                    ),
                )
            ],
            else_case=helper_models.LabeledRecipe(
                label="cebranch", node=_for_each_increment("dec_body")
            ),
            input_edges={
                edge_models.TargetHandle(
                    node="flowcond", port="xs"
                ): edge_models.InputSource(port="xs"),
                edge_models.TargetHandle(
                    node="cbranch", port="xs"
                ): edge_models.InputSource(port="xs"),
                edge_models.TargetHandle(
                    node="cebranch", port="xs"
                ): edge_models.InputSource(port="xs"),
            },
            prospective_output_edges={
                edge_models.OutputTarget(port="ys"): [
                    edge_models.SourceHandle(node="cbranch", port="ys"),
                    edge_models.SourceHandle(node="cebranch", port="ys"),
                ]
            },
        )
    },
    input_edges={
        edge_models.TargetHandle(node="if_c", port="xs"): edge_models.InputSource(
            port="xs"
        ),
    },
    edges={},
    output_edges={
        edge_models.OutputTarget(port="ys"): edge_models.SourceHandle(
            node="if_c", port="ys"
        ),
    },
)


# A while-node whose CONDITION is a flow-control (for-each) recipe.
while_flow_control_condition_recipe = workflow_recipe.WorkflowRecipe(
    inputs=["xs"],
    outputs=["xs"],
    nodes={
        "while_0": while_recipe.WhileRecipe(
            inputs=["xs"],
            outputs=["xs"],
            case=helper_models.ConditionalCase(
                condition=helper_models.LabeledRecipe(
                    label="wcond", node=_for_each_increment("wc_body")
                ),
                condition_output="ys",
                body=helper_models.LabeledRecipe(
                    label="wbody", node=_for_each_increment("wb_body")
                ),
            ),
            input_edges={
                edge_models.TargetHandle(
                    node="wcond", port="xs"
                ): edge_models.InputSource(port="xs"),
                edge_models.TargetHandle(
                    node="wbody", port="xs"
                ): edge_models.InputSource(port="xs"),
            },
            output_edges={
                edge_models.OutputTarget(port="xs"): edge_models.SourceHandle(
                    node="wbody", port="ys"
                )
            },
        )
    },
    input_edges={
        edge_models.TargetHandle(node="while_0", port="xs"): edge_models.InputSource(
            port="xs"
        )
    },
    edges={},
    output_edges={
        edge_models.OutputTarget(port="xs"): edge_models.SourceHandle(
            node="while_0", port="xs"
        )
    },
)


# An if-node whose if-branch AND else-branch are each a for-each (flow control sitting
# directly as a branch body). Condition is the atomic is_positive(n).
def _for_each(label_body: str, node):
    from flowrep.nodes import for_recipe

    return for_recipe.ForEachRecipe(
        inputs=["xs"],
        outputs=["ys"],
        body_node=helper_models.LabeledRecipe(label=label_body, node=node),
        input_edges={
            edge_models.TargetHandle(
                node=label_body, port="x"
            ): edge_models.InputSource(port="xs"),
        },
        output_edges={
            edge_models.OutputTarget(port="ys"): edge_models.SourceHandle(
                node=label_body, port="output_0"
            )
        },
        nested_ports=["x"],
    )


if_branch_is_for_each_recipe = workflow_recipe.WorkflowRecipe(
    inputs=["xs", "n"],
    outputs=["ys"],
    nodes={
        "if_0": if_recipe.IfRecipe(
            inputs=["xs", "n"],
            outputs=["ys"],
            cases=[
                helper_models.ConditionalCase(
                    condition=helper_models.LabeledRecipe(
                        label="cond", node=library.is_positive.flowrep_recipe
                    ),
                    body=helper_models.LabeledRecipe(
                        label="if_branch",
                        node=_for_each("inc_body", library.increment.flowrep_recipe),
                    ),
                )
            ],
            else_case=helper_models.LabeledRecipe(
                label="else_branch",
                node=_for_each("dec_body", library.decrement.flowrep_recipe),
            ),
            input_edges={
                edge_models.TargetHandle(
                    node="cond", port="n"
                ): edge_models.InputSource(port="n"),
                edge_models.TargetHandle(
                    node="if_branch", port="xs"
                ): edge_models.InputSource(port="xs"),
                edge_models.TargetHandle(
                    node="else_branch", port="xs"
                ): edge_models.InputSource(port="xs"),
            },
            prospective_output_edges={
                edge_models.OutputTarget(port="ys"): [
                    edge_models.SourceHandle(node="if_branch", port="ys"),
                    edge_models.SourceHandle(node="else_branch", port="ys"),
                ]
            },
        )
    },
    input_edges={
        edge_models.TargetHandle(node="if_0", port="xs"): edge_models.InputSource(
            port="xs"
        ),
        edge_models.TargetHandle(node="if_0", port="n"): edge_models.InputSource(
            port="n"
        ),
    },
    edges={},
    output_edges={
        edge_models.OutputTarget(port="ys"): edge_models.SourceHandle(
            node="if_0", port="ys"
        )
    },
)


# A for-each whose body is itself an if-node (flow control sitting directly as the
# loop body). The inner if maps increment/decrement by sign of each element.
_inner_if_recipe = if_recipe.IfRecipe(
    inputs=["x"],
    outputs=["m"],
    cases=[
        helper_models.ConditionalCase(
            condition=helper_models.LabeledRecipe(
                label="icond", node=library.is_positive.flowrep_recipe
            ),
            body=helper_models.LabeledRecipe(
                label="ibranch", node=library.increment.flowrep_recipe
            ),
        )
    ],
    else_case=helper_models.LabeledRecipe(
        label="ebranch", node=library.decrement.flowrep_recipe
    ),
    input_edges={
        edge_models.TargetHandle(node="icond", port="n"): edge_models.InputSource(
            port="x"
        ),
        edge_models.TargetHandle(node="ibranch", port="x"): edge_models.InputSource(
            port="x"
        ),
        edge_models.TargetHandle(node="ebranch", port="x"): edge_models.InputSource(
            port="x"
        ),
    },
    prospective_output_edges={
        edge_models.OutputTarget(port="m"): [
            edge_models.SourceHandle(node="ibranch", port="output_0"),
            edge_models.SourceHandle(node="ebranch", port="output_0"),
        ]
    },
)


for_body_is_if_recipe = workflow_recipe.WorkflowRecipe(
    inputs=["xs"],
    outputs=["ys"],
    nodes={
        "for_0": for_recipe.ForEachRecipe(
            inputs=["xs"],
            outputs=["ys"],
            body_node=helper_models.LabeledRecipe(
                label="inner_if", node=_inner_if_recipe
            ),
            input_edges={
                edge_models.TargetHandle(
                    node="inner_if", port="x"
                ): edge_models.InputSource(port="xs"),
            },
            output_edges={
                edge_models.OutputTarget(port="ys"): edge_models.SourceHandle(
                    node="inner_if", port="m"
                )
            },
            nested_ports=["x"],
        )
    },
    input_edges={
        edge_models.TargetHandle(node="for_0", port="xs"): edge_models.InputSource(
            port="xs"
        )
    },
    edges={},
    output_edges={
        edge_models.OutputTarget(port="ys"): edge_models.SourceHandle(
            node="for_0", port="ys"
        )
    },
)


class TestRenderedSourceAndGuard(unittest.TestCase):
    def test_rejects_recipe_with_reference(self):
        recipe = library.simple_workflow.flowrep_recipe  # has a reference
        with self.assertRaises(ValueError):
            pysource.recipe2python("rebuilt", recipe)

    def test_rendered_source_builds_callable(self):
        rs = pysource.RenderedSource(
            source="def rebuilt(a):\n    return a\n",
            namespace={},
            function_name="rebuilt",
        )
        fn = rs.build()
        self.assertEqual(fn(7), 7)

    def test_rendered_source_uses_namespace(self):
        rs = pysource.RenderedSource(
            source="def rebuilt(a=_default_a):\n    return a\n",
            namespace={"_default_a": 99},
            function_name="rebuilt",
        )
        fn = rs.build()
        self.assertEqual(fn(), 99)

    def test_rendered_source_uses_repr_defaults(self):
        rs = pysource.RenderedSource(
            source="def rebuilt(a=99):\n    return a\n",
            namespace={},
            function_name="rebuilt",
        )
        fn = rs.build()
        self.assertEqual(fn(), 99)


class TestFunctionBuilderDecorator(unittest.TestCase):
    def test_render_emits_output_labels_in_decorator(self):
        fb = pysource.FunctionBuilder(
            name="f", params=["a"], return_symbols=["a"], output_labels=["m", "n"]
        )
        self.assertIn('@flowrep.workflow("m", "n")', fb.render())

    def test_render_bare_decorator_without_labels(self):
        fb = pysource.FunctionBuilder(name="f", params=["a"], return_symbols=["a"])
        src = fb.render()
        self.assertIn("@flowrep.workflow\n", src)
        self.assertNotIn("@flowrep.workflow(", src)


class TestNameAllocator(unittest.TestCase):
    def test_fresh_returns_hint_then_suffixes(self):
        alloc = pysource._NameAllocator()
        self.assertEqual(alloc.fresh("x"), "x")
        self.assertEqual(alloc.fresh("x"), "x_0")
        self.assertEqual(alloc.fresh("x"), "x_1")

    def test_fresh_sanitises_invalid_hint(self):
        alloc = pysource._NameAllocator()
        out = alloc.fresh("output_0.bad")  # not a valid identifier
        self.assertTrue(out.isidentifier())

    def test_reserve_blocks_later_fresh_collision(self):
        alloc = pysource._NameAllocator()
        self.assertEqual(alloc.reserve("result"), "result")
        self.assertEqual(alloc.fresh("result"), "result_0")


class TestSingleAtomicDag(unittest.TestCase):
    def _free_recipe(self):
        def one_add(a, b):
            result = library.my_add(a, b)
            return result

        return one_add, makers.reference_free(one_add)

    def test_executes(self):
        original, free = self._free_recipe()
        fn = pysource.recipe2python("rebuilt", free).build()
        self.assertEqual(fn(2, 3), original(2, 3))

    def test_round_trips(self):
        _, free = self._free_recipe()
        rendered = pysource.recipe2python("rebuilt", free)
        fn = rendered.build()
        self.assertEqual(
            makers.dump_no_refs(fn.flowrep_recipe), makers.dump_no_refs(free)
        )


class TestMultiNodeDag(unittest.TestCase):
    def test_chained_and_unpacked(self):
        def chained(a, b):
            s = library.my_add(a, b)  # one output
            q, r = library.divmod_func(s, b)  # tuple unpack -> two outputs
            return q, r

        free = makers.reference_free(chained)
        rendered = pysource.recipe2python("rebuilt", free)
        fn = rendered.build()
        self.assertEqual(fn(7, 3), chained(7, 3))
        self.assertEqual(
            makers.dump_no_refs(fn.flowrep_recipe), makers.dump_no_refs(free)
        )

    def test_emits_in_dependency_order_when_nodes_unordered(self):
        # Build a recipe whose .nodes dict is *not* topologically ordered, by
        # round-tripping a normal one then re-inserting nodes in reverse.
        def chained(a, b):
            s = library.my_add(a, b)
            t = library.my_mul(s, b)
            return t

        free = makers.reference_free(chained)
        reordered_nodes = dict(reversed(list(free.nodes.items())))
        scrambled = free.model_copy(update={"nodes": reordered_nodes})
        rendered = pysource.recipe2python("rebuilt", scrambled)
        fn = rendered.build()
        self.assertEqual(fn(2, 5), chained(2, 5))


class TestSignatureParams(unittest.TestCase):
    def test_defaults_bound_as_live_objects(self):
        sentinel = object()

        def with_default(a, b=sentinel):
            r = library.identity(a)
            return r

        free = makers.reference_free(with_default)
        sig = inspect.signature(with_default)
        rendered = pysource.recipe2python("rebuilt", free, sig)
        fn = rendered.build()
        # default object survives into __defaults__ as the *same* live object
        self.assertIs(fn.__defaults__[0], sentinel)
        self.assertEqual(fn(5), with_default(5))

    def test_positional_only_and_keyword_only_markers(self):
        def kinds(x, /, y, *, z):
            r = library.my_add(x, y)
            out = library.my_add(r, z)
            return out

        free = makers.reference_free(kinds)
        sig = inspect.signature(kinds)
        rendered = pysource.recipe2python("rebuilt", free, sig)
        fn = rendered.build()
        new_sig = inspect.signature(fn)
        self.assertEqual(
            new_sig.parameters["x"].kind, inspect.Parameter.POSITIONAL_ONLY
        )
        self.assertEqual(new_sig.parameters["z"].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(fn(1, 2, z=3), kinds(1, 2, z=3))


class TestOutputsEdgeCases(unittest.TestCase):
    def test_passthrough_output_returns_input(self):
        # Output 'kept' is sourced directly from input 'a'.
        def passthrough(a, b):
            s = library.my_add(a, b)
            return s, a  # 'a' is a passthrough output

        free = makers.reference_free(passthrough)
        rendered = pysource.recipe2python("rebuilt", free)
        fn = rendered.build()
        self.assertEqual(fn(4, 6), passthrough(4, 6))
        self.assertEqual(
            makers.dump_no_refs(fn.flowrep_recipe), makers.dump_no_refs(free)
        )

    def test_duplicate_source_port_raises(self):
        # Hand-build a recipe where two outputs share one source handle.
        free = makers.reference_free(self._single_two_named_outputs_source())
        # Force two output ports onto the same source handle.

        single_src = next(iter(free.output_edges.values()))
        bad = free.model_copy(
            update={
                "outputs": ["p", "q"],
                "output_edges": {
                    edge_models.OutputTarget(port="p"): single_src,
                    edge_models.OutputTarget(port="q"): single_src,
                },
            }
        )
        # p and q both come from the same handle; no required-name conflict at the
        # top level (we use annotations there), but the source code parser guards
        # against duplicate output symbols
        rendered = pysource.recipe2python("rebuilt", bad)
        with self.assertRaisesRegex(
            ValueError,
            "Workflow python definitions must have unique returns",
        ):
            rendered.build()

    def _single_two_named_outputs_source(self):
        def one(a, b):
            s = library.my_add(a, b)
            return s

        return one


class TestNestedWorkflowNode(unittest.TestCase):
    def test_reference_free_subworkflow_node(self):
        # Build a recipe whose .nodes contains a reference-free WorkflowRecipe.
        # inner's output variable is named 'output_0' to match the output port
        # name that library.my_add exposes (and that free_outer's output_edge
        # references), making the substituted recipe internally consistent.
        def inner(a, b):
            output_0 = library.my_add(a, b)
            return output_0

        def outer(a, b):
            r = library.my_add(a, b)
            return r

        free_outer = makers.reference_free(outer)
        free_inner = makers.reference_free(inner)
        # Replace the atomic node with the reference-free sub-workflow node.
        label = next(iter(free_outer.nodes))
        nodes = dict(free_outer.nodes)
        nodes[label] = free_inner  # workflow node, reference=None
        recipe = free_outer.model_copy(update={"nodes": nodes})

        rendered = pysource.recipe2python("rebuilt", recipe)
        fn = rendered.build()
        self.assertEqual(fn(2, 3), 5)
        # Re-parse: the nested def re-parses as a workflow node (with a reference);
        # comparing with references stripped yields structural equivalence.
        self.assertEqual(
            makers.dump_no_refs(fn.flowrep_recipe), makers.dump_no_refs(recipe)
        )


class TestForEach(unittest.TestCase):
    def test_nested_for_round_trip_and_exec(self):
        def mapper(xs, k):
            acc = []
            for x in xs:
                v = library.my_mul(x, k)
                acc.append(v)
            return acc

        free = makers.reference_free(mapper)
        rendered = pysource.recipe2python("rebuilt", free)
        fn = rendered.build()
        self.assertEqual(fn([1, 2, 3], 10), mapper([1, 2, 3], 10))
        self.assertEqual(
            makers.dump_no_refs(fn.flowrep_recipe), makers.dump_no_refs(free)
        )

    def test_zipped_for(self):
        def zipper(xs, ys):
            acc = []
            for x, y in zip(xs, ys, strict=True):
                v = library.my_add(x, y)
                acc.append(v)
            return acc

        free = makers.reference_free(zipper)
        fn = pysource.recipe2python("rebuilt", free).build()
        self.assertEqual(fn([1, 2], [3, 4]), zipper([1, 2], [3, 4]))

    def test_for_each_body_without_underlying_python(self):
        self.assertEqual(
            wfms.run_recipe(workflow_for_each_recipe, xs=[1, 2, 3])
            .output_ports["ys"]
            .value,
            [2, 3, 4],
            msg="Sanity check that the recipe is fine",
        )
        rendered = pysource.recipe2python("built", workflow_for_each_recipe)
        fn = rendered.build()
        self.assertEqual(fn([1, 2, 3]), [2, 3, 4])


class TestIf(unittest.TestCase):
    def test_if_else_round_trip_and_exec(self):
        def chooser(a, b):
            if library.my_condition(a, b):  # a < b
                v = library.my_add(a, b)
            else:
                v = library.my_mul(a, b)
            return v

        free = makers.reference_free(chooser)
        rendered = pysource.recipe2python("rebuilt", free)
        fn = rendered.build()
        self.assertEqual(fn(1, 5), chooser(1, 5))
        self.assertEqual(fn(5, 1), chooser(5, 1))
        self.assertEqual(
            makers.dump_no_refs(fn.flowrep_recipe), makers.dump_no_refs(free)
        )

    def test_condition_without_underlying_python(self):
        self.assertEqual(
            wfms.run_recipe(workflow_condition_recipe, n=0).output_ports["m"].value,
            0 - 1,
            msg="Sanity check that the recipe is fine",
        )
        self.assertEqual(
            wfms.run_recipe(workflow_condition_recipe, n=1).output_ports["m"].value,
            1 + 1,
            msg="Sanity check that the recipe is fine",
        )
        pysource.recipe2python("built", workflow_condition_recipe)


class TestWhile(unittest.TestCase):
    def test_while_round_trip_and_exec(self):
        def countup(i, bound):
            while library.my_condition(i, bound):  # i < bound
                i = library.increment(i)
            return i

        free = makers.reference_free(countup)
        rendered = pysource.recipe2python("rebuilt", free)
        fn = rendered.build()
        self.assertEqual(fn(0, 5), countup(0, 5))
        self.assertEqual(
            makers.dump_no_refs(fn.flowrep_recipe), makers.dump_no_refs(free)
        )

    def test_while_body_without_underlying_python(self):
        self.assertEqual(
            wfms.run_recipe(workflow_while_recipe, i=3).output_ports["i"].value,
            0,
            msg="Sanity check that the recipe is fine",
        )
        rendered = pysource.recipe2python("built", workflow_while_recipe)
        fn = rendered.build()
        self.assertEqual(
            fn(3),
            wfms.run_recipe(workflow_while_recipe, i=3).output_ports["i"].value,
            msg="Emitted Python must match the recipe's runtime result",
        )


class TestFlowControlConditionNode(unittest.TestCase):
    def test_if_condition_flow_control_raises(self):
        with self.assertRaises(NotImplementedError) as ctx:
            pysource.recipe2python("built", if_flow_control_condition_recipe)
        self.assertIn("flowcond", str(ctx.exception))
        self.assertIn("single callable", str(ctx.exception))

    def test_while_condition_flow_control_raises(self):
        with self.assertRaises(NotImplementedError) as ctx:
            pysource.recipe2python("built", while_flow_control_condition_recipe)
        self.assertIn("wcond", str(ctx.exception))


class TestFlowControlAsBranchBody(unittest.TestCase):
    """A flow-control recipe sitting directly as a branch/loop body is inlined.

    We do NOT assert structural round-trip equality: the parser re-wraps each branch
    body in a workflow node, so recipe->python->recipe is not structurally identical.
    Instead, we assert the original recipe, the reverse-rendered function, and the
    round-trip recipe all evaluate the same (per the spec).
    """

    def _assert_three_way(self, recipe, args, kwargs):
        original = (
            wfms.run_recipe(recipe, **kwargs)
            .output_ports[next(iter(recipe.outputs))]
            .value
        )
        fn = pysource.recipe2python("built", recipe).build()
        self.assertEqual(
            fn(*args), original, msg="rendered function vs original recipe"
        )
        rt = fn.flowrep_recipe
        rt_val = (
            wfms.run_recipe(rt, **kwargs).output_ports[next(iter(rt.outputs))].value
        )
        self.assertEqual(rt_val, original, msg="round-trip recipe vs original recipe")
        return original

    def test_if_branch_is_for_each(self):
        self.assertEqual(
            self._assert_three_way(
                if_branch_is_for_each_recipe,
                args=([1, 2, 3], 1),
                kwargs=dict(xs=[1, 2, 3], n=1),
            ),
            [2, 3, 4],
            msg="positive n takes the increment for-each branch",
        )
        self.assertEqual(
            self._assert_three_way(
                if_branch_is_for_each_recipe,
                args=([1, 2, 3], -1),
                kwargs=dict(xs=[1, 2, 3], n=-1),
            ),
            [0, 1, 2],
            msg="non-positive n takes the decrement for-each branch",
        )

    def test_for_body_is_if(self):
        self.assertEqual(
            self._assert_three_way(
                for_body_is_if_recipe, args=([-2, 3],), kwargs=dict(xs=[-2, 3])
            ),
            [-3, 4],
            msg="per-element if: decrement negatives, increment positives",
        )


class TestTry(unittest.TestCase):
    def test_try_except_round_trip_and_exec(self):
        def safe_div(a, b):
            try:
                z = library.divide(a, b)
            except ZeroDivisionError:
                z = library.identity(a)
            return z

        free = makers.reference_free(safe_div)
        rendered = pysource.recipe2python("rebuilt", free)
        fn = rendered.build()
        self.assertEqual(fn(6, 3), safe_div(6, 3))
        self.assertEqual(fn(6, 0), safe_div(6, 0))
        self.assertEqual(
            makers.dump_no_refs(fn.flowrep_recipe), makers.dump_no_refs(free)
        )

    def test_try_except_with_custom_exception(self):
        def custom_exception_branch(a, b):
            try:
                z = library.raises_custom(a, b)
            except library.MyCustomException:
                z = library.identity(a)
            return z

        free = makers.reference_free(custom_exception_branch)
        rendered = pysource.recipe2python("rebuilt", free)
        self.assertIn(
            "flowrep_static.library.MyCustomException",
            rendered.source,
            msg="New source should have the FQN of the custom exception",
        )
        fn = rendered.build()
        self.assertEqual(fn(6, 3), custom_exception_branch(6, 3))
        self.assertEqual(
            makers.dump_no_refs(fn.flowrep_recipe), makers.dump_no_refs(free)
        )

    def test_try_bodies_without_underlying_python(self):
        self.assertEqual(
            wfms.run_recipe(workflow_try_recipe, a=6, b=3).output_ports["z"].value,
            2.0,
            msg="Sanity check that the recipe is fine",
        )
        self.assertEqual(
            wfms.run_recipe(workflow_try_recipe, a=6, b=0).output_ports["z"].value,
            6,
            msg="Sanity check that the except branch runs",
        )
        rendered = pysource.recipe2python("built", workflow_try_recipe)
        fn = rendered.build()
        self.assertEqual(fn(6, 3), 2.0)
        self.assertEqual(fn(6, 0), 6)


def _with_default(a, b=10):
    """Module-level workflow function with a default; importable for DagData tests."""
    r = library.my_add(a, b)
    return r


def _typed_single(a: int, b: float = 2.0) -> float:
    """Module-level typed workflow for DagData annotation tests."""
    r = library.my_add(a, b)
    return r


def _typed_multi(a: float, b: float) -> tuple[int, float]:
    """Module-level typed multi-output workflow for DagData annotation tests."""
    q, r = library.divmod_func(a, b)
    return q, r


class TestDagData(unittest.TestCase):
    def test_dagdata_round_trip(self):
        # DagData.from_recipe on a referenced recipe populates InputDataPort defaults.
        # The function must be module-level (importable) for DagData to parse defaults
        # from the reference's fully-qualified name.
        recipe_with_ref = workflow_parser.parse_workflow(_with_default)
        dagdata = retrospective.DagData.from_recipe(recipe_with_ref)
        rendered = pysource.dagdata2python("rebuilt", dagdata)
        fn = rendered.build()
        self.assertEqual(fn(5), _with_default(5))
        # default recovered from the port
        self.assertEqual(fn.__defaults__, (10,))

    def test_dagdata_propagates_types_and_default(self):
        dagdata = retrospective.DagData.from_recipe(
            workflow_parser.parse_workflow(_typed_single)
        )
        fn = pysource.dagdata2python("rebuilt", dagdata).build()
        self.assertEqual(fn(5), _typed_single(5))
        self.assertEqual(fn.__defaults__, (2.0,))
        hints = typing.get_type_hints(fn, include_extras=True)
        self.assertIs(hints["a"], int)
        self.assertIs(hints["b"], float)
        self.assertIs(hints["return"], float)

    def test_dagdata_multi_output_types_round_trip(self):
        free = makers.reference_free(_typed_multi)
        dagdata = retrospective.DagData.from_recipe(
            workflow_parser.parse_workflow(_typed_multi)
        )
        fn = pysource.dagdata2python("rebuilt", dagdata).build()
        self.assertEqual(fn(7.0, 3.0), _typed_multi(7.0, 3.0))
        self.assertEqual(
            typing.get_type_hints(fn, include_extras=True)["return"],
            tuple[int, float],
        )
        self.assertEqual(
            makers.dump_no_refs(fn.flowrep_recipe), makers.dump_no_refs(free)
        )


class TestGuardsAndEdgeCases(unittest.TestCase):
    def test_locals_qualname_raises(self):
        free = makers.reference_free(_with_default)
        label = next(iter(free.nodes))
        node = free.nodes[label]
        # VersionInfo is a dataclass; PythonReference/AtomicRecipe are pydantic models.
        bad_info = dataclasses.replace(
            node.reference.info, qualname="outer.<locals>.inner"
        )
        bad_ref = node.reference.model_copy(update={"info": bad_info})
        bad_node = node.model_copy(update={"reference": bad_ref})
        recipe = free.model_copy(update={"nodes": {label: bad_node}})
        with self.assertRaisesRegex(ValueError, "local scope"):
            pysource.recipe2python("rebuilt", recipe)

    def test_cycle_raises(self):

        def chained(a, b):
            s = library.my_add(a, b)
            t = library.my_mul(s, b)
            return t

        free = makers.reference_free(chained)
        labels = list(free.nodes)
        # Inject a back-edge to create a cycle. model_copy does not re-validate,
        # so it bypasses the recipe's acyclic check.
        cyclic_edges = dict(free.edges)
        cyclic_edges[edge_models.TargetHandle(node=labels[0], port="a")] = (
            edge_models.SourceHandle(node=labels[1], port="output_0")
        )
        cyclic = free.model_copy(update={"edges": cyclic_edges})
        with self.assertRaisesRegex(ValueError, "cycle"):
            pysource.recipe2python("rebuilt", cyclic)

    def test_trailing_positional_only_marker(self):
        def kinds(a, b, /):
            r = library.my_add(a, b)
            return r

        free = makers.reference_free(kinds)
        rendered = pysource.recipe2python("rebuilt", free, inspect.signature(kinds))
        self.assertIn("/", rendered.source)
        fn = rendered.build()
        self.assertEqual(
            inspect.signature(fn).parameters["b"].kind,
            inspect.Parameter.POSITIONAL_ONLY,
        )

    def test_positional_only_node_argument(self):
        # A referenced node with positional-only inputs is called positionally.
        def uses(x, y):
            r = _pos_only_add(x, y)
            return r

        free = makers.reference_free(uses)
        rendered = pysource.recipe2python("rebuilt", free)
        self.assertRegex(rendered.source, r"_pos_only_add\(\w+, \w+\)")
        self.assertEqual(rendered.build()(2, 3), 5)

    def test_for_each_transferred_input_output(self):
        # A zipped input forwarded straight to an output is collected per iteration.
        def fwd(xs, ks):
            acc = []
            kept = []
            for x, k in zip(xs, ks, strict=True):
                v = library.my_add(x, k)
                acc.append(v)
                kept.append(k)
            return acc, kept

        free = makers.reference_free(fwd)
        fn = pysource.recipe2python("rebuilt", free).build()
        self.assertEqual(fn([1, 2], [10, 20]), fwd([1, 2], [10, 20]))

    def test_condition_with_defaulted_unsourced_input(self):
        # increment(x, step=1): 'step' is defaulted and left unsourced as a condition.
        def f(a):
            if library.increment(a):  # noqa: SIM108
                r = library.identity(a)
            else:
                r = library.negate(a)
            return r

        free = makers.reference_free(f)
        rendered = pysource.recipe2python("rebuilt", free)
        self.assertIn("library.increment(x=", rendered.source)
        fn = rendered.build()
        self.assertEqual(fn(5), f(5))
        self.assertEqual(fn(-1), f(-1))

    def test_multi_exception_except_clause(self):
        def f(a, b):
            try:
                z = library.divide(a, b)
            except (ZeroDivisionError, ValueError):
                z = library.identity(a)
            return z

        free = makers.reference_free(f)
        rendered = pysource.recipe2python("rebuilt", free)
        self.assertIn("except (", rendered.source)
        self.assertEqual(rendered.build()(6, 0), f(6, 0))

    def test_alias_conflict_raises(self):
        # Two outputs sourced from one handle but pinned to different names cannot
        # be emitted as assignments. Drive the guard directly via _emit_workflow_body.

        body = workflow_recipe.WorkflowRecipe.model_validate(
            {
                "type": "workflow",
                "inputs": ["a", "b"],
                "outputs": ["p", "q"],
                "nodes": {"my_add_0": library.my_add.flowrep_recipe},
                "input_edges": {"my_add_0.a": "a", "my_add_0.b": "b"},
                "edges": {},
                "output_edges": {
                    "p": "my_add_0.output_0",
                    "q": "my_add_0.output_0",
                },
            }
        )
        with self.assertRaisesRegex(ValueError, "cannot be emitted as an assignment"):
            pysource._emit_workflow_body(
                body,
                {"a": "a", "b": "b"},
                {"p": "P", "q": "Q"},
                pysource._Emitter(),
                pysource._NameAllocator(),
                set(),
            )


class TestAnnotationReconstruction(unittest.TestCase):
    def test_unannotated_outputs_stay_that_way(self):
        # No signature -> Any-typed return annotation, label still pinned.
        def plain(a, b):
            r = library.my_add(a, b)
            return r

        free = makers.reference_free(plain)
        rendered = pysource.recipe2python("rebuilt", free)
        self.assertNotIn("typing.Any", rendered.source)
        self.assertNotIn("->", rendered.source)
        self.assertIn(
            '@flowrep.workflow("r")',
            rendered.source,
            msg="The output port name is pinned by the decorator",
        )
        reconstructed_sig = inspect.signature(rendered.build())
        self.assertIs(
            reconstructed_sig.return_annotation,
            inspect.Signature.empty,
            msg="No signature -> no return annotation emitted",
        )

    def test_input_annotations_resolve_including_generics(self):
        def typed(x: int, y: list[int], z: dict[str, int]):
            r = library.my_add(x, y)
            out = library.my_add(r, z)
            return out

        free = makers.reference_free(typed)
        fn = pysource.recipe2python("rebuilt", free, inspect.signature(typed)).build()
        sig = inspect.signature(fn)
        self.assertIs(sig.parameters["x"].annotation, int)
        self.assertEqual(sig.parameters["y"].annotation, list[int])
        self.assertEqual(sig.parameters["z"].annotation, dict[str, int])

    def test_annotated_input_preserved_verbatim(self):
        def f(x: typing.Annotated[int, "meta"], y: float):
            r = library.my_add(x, y)
            return r

        free = makers.reference_free(f)
        fn = pysource.recipe2python("rebuilt", free, inspect.signature(f)).build()
        hints = typing.get_type_hints(fn, include_extras=True)
        self.assertEqual(hints["x"], typing.Annotated[int, "meta"])
        self.assertIs(hints["y"], float)

    def test_annotation_and_default_both_survive(self):
        def f(x: int, y: float = 0.5):
            r = library.my_add(x, y)
            return r

        free = makers.reference_free(f)
        fn = pysource.recipe2python("rebuilt", free, inspect.signature(f)).build()
        sig = inspect.signature(fn)
        self.assertIs(sig.parameters["y"].annotation, float)
        self.assertEqual(sig.parameters["y"].default, 0.5)

    def test_single_return_annotation_is_verbatim(self):
        def typed(a, b) -> float:
            r = library.my_add(a, b)
            return r

        free = makers.reference_free(typed)
        fn = pysource.recipe2python("rebuilt", free, inspect.signature(typed)).build()
        self.assertIs(inspect.signature(fn).return_annotation, float)

    def test_multi_return_annotation_is_verbatim(self):
        def typed(a, b) -> tuple[int, float]:
            q, r = library.divmod_func(a, b)
            return q, r

        free = makers.reference_free(typed)
        fn = pysource.recipe2python("rebuilt", free, inspect.signature(typed)).build()
        self.assertEqual(typing.get_type_hints(fn)["return"], tuple[int, float])

    def test_mismatched_return_annotation_emitted_verbatim(self):
        # A non-tuple return annotation for a 2-output workflow is emitted verbatim
        # (no splitting, no typing.Any fallback); port names come from the decorator.
        def typed_bad(a, b) -> int:
            q, r = library.divmod_func(a, b)
            return q, r

        free = makers.reference_free(typed_bad)
        rendered = pysource.recipe2python("rebuilt", free, inspect.signature(typed_bad))
        self.assertIn('@flowrep.workflow("q", "r")', rendered.source)
        fn = rendered.build()
        self.assertIs(typing.get_type_hints(fn)["return"], int)
        rebuilt_recipe = workflow_parser.parse_workflow(fn)
        with self.assertRaises(
            ValueError,
            msg="Recipes don't know about annotations, so we can get a recipe from "
            "such a function, but it will fail to convert to a data object because of "
            "the mismatch between ports and the return annotation",
        ):
            retrospective.DagData.from_recipe(rebuilt_recipe)

    def test_output_port_name_pinned_via_decorator(self):
        # Return symbol is "s" (my_add's output) but the port is renamed; the
        # decorator must pin the port name regardless of the return symbol.
        def one(a, b):
            s = library.my_add(a, b)
            return s

        free = makers.reference_free(one)
        src = next(iter(free.output_edges.values()))
        renamed = free.model_copy(
            update={
                "outputs": ["renamed"],
                "output_edges": {
                    edge_models.OutputTarget(port="renamed"): src,
                },
            }
        )
        rendered = pysource.recipe2python("rebuilt", renamed)
        self.assertIn('@flowrep.workflow("renamed")', rendered.source)
        fn = rendered.build()
        self.assertEqual(fn.flowrep_recipe.outputs, ["renamed"])

    def test_input_and_return_annotations_do_not_interfere(self):
        def f(
            x: typing.Annotated[int, "in-meta"], y: float
        ) -> typing.Annotated[float, "out-meta"]:
            r = library.my_add(x, y)
            return r

        free = makers.reference_free(f)
        rendered = pysource.recipe2python("rebuilt", free, inspect.signature(f))
        self.assertIn('@flowrep.workflow("r")', rendered.source)
        fn = rendered.build()
        hints = typing.get_type_hints(fn, include_extras=True)
        self.assertEqual(hints["x"], typing.Annotated[int, "in-meta"])
        self.assertEqual(hints["return"], typing.Annotated[float, "out-meta"])
