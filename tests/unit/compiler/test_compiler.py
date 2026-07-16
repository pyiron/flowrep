import dataclasses
import gc
import inspect
import linecache
import pathlib
import re
import sys
import tempfile
import typing
import unittest

from pyiron_snippets import versions

from flowrep import base_models, edge_models, std, wfms
from flowrep.compiler import flow_control, function, source, statements, sugar
from flowrep.parsers import atomic_parser, workflow_parser
from flowrep.prospective import (
    atomic_recipe,
    constant_recipe,
    for_recipe,
    helper_models,
    if_recipe,
    try_recipe,
    while_recipe,
    workflow_recipe,
)
from flowrep.retrospective import datastructures

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
                        recipe=workflow_recipe.WorkflowRecipe(
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
                        recipe=library.increment.flowrep_recipe,
                    ),
                ),
            ],
            else_case=helper_models.LabeledRecipe(
                label="else_case",
                recipe=library.decrement.flowrep_recipe,
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
                    recipe=workflow_recipe.WorkflowRecipe(
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
                    recipe=library.decrement.flowrep_recipe,
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
                recipe=library.divide.flowrep_recipe,
            ),
            exception_cases=[
                helper_models.ExceptionCase(
                    exceptions=[versions.VersionInfo.of(ZeroDivisionError)],
                    body=helper_models.LabeledRecipe(
                        label="except_body",
                        recipe=std.identity.flowrep_recipe,
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
                recipe=library.increment.flowrep_recipe,
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
    from flowrep.prospective import for_recipe  # local import keeps this near its use

    return for_recipe.ForEachRecipe(
        inputs=["xs"],
        outputs=["ys"],
        body_node=helper_models.LabeledRecipe(
            label=label_body, recipe=library.increment.flowrep_recipe
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
                        label="flowcond", recipe=_for_each_increment("c_body")
                    ),
                    condition_output="ys",
                    body=helper_models.LabeledRecipe(
                        label="cbranch", recipe=_for_each_increment("inc_body")
                    ),
                )
            ],
            else_case=helper_models.LabeledRecipe(
                label="cebranch", recipe=_for_each_increment("dec_body")
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
                    label="wcond", recipe=_for_each_increment("wc_body")
                ),
                condition_output="ys",
                body=helper_models.LabeledRecipe(
                    label="wbody", recipe=_for_each_increment("wb_body")
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
    from flowrep.prospective import for_recipe

    return for_recipe.ForEachRecipe(
        inputs=["xs"],
        outputs=["ys"],
        body_node=helper_models.LabeledRecipe(label=label_body, recipe=node),
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
                        label="cond", recipe=library.is_positive.flowrep_recipe
                    ),
                    body=helper_models.LabeledRecipe(
                        label="if_branch",
                        recipe=_for_each("inc_body", library.increment.flowrep_recipe),
                    ),
                )
            ],
            else_case=helper_models.LabeledRecipe(
                label="else_branch",
                recipe=_for_each("dec_body", library.decrement.flowrep_recipe),
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
                label="icond", recipe=library.is_positive.flowrep_recipe
            ),
            body=helper_models.LabeledRecipe(
                label="ibranch", recipe=library.increment.flowrep_recipe
            ),
        )
    ],
    else_case=helper_models.LabeledRecipe(
        label="ebranch", recipe=library.decrement.flowrep_recipe
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
                label="inner_if", recipe=_inner_if_recipe
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
            source._workflow2python(recipe)

    def test_rendered_source_builds_callable(self):
        rs = source.RenderedSource(
            source="def rebuilt(a):\n    return a\n",
            namespace={},
            function_name="rebuilt",
        )
        fn = rs.build()
        self.assertEqual(fn(7), 7)

    def test_rendered_source_uses_namespace(self):
        rs = source.RenderedSource(
            source="def rebuilt(a=_default_a):\n    return a\n",
            namespace={"_default_a": 99},
            function_name="rebuilt",
        )
        fn = rs.build()
        self.assertEqual(fn(), 99)

    def test_rendered_source_uses_repr_defaults(self):
        rs = source.RenderedSource(
            source="def rebuilt(a=99):\n    return a\n",
            namespace={},
            function_name="rebuilt",
        )
        fn = rs.build()
        self.assertEqual(fn(), 99)

    def test_build_releases_module_and_linecache_after_gc(self):
        free = makers.make_simple_workflow_recipe().model_copy(
            update={"reference": None}
        )
        fn = source._workflow2python(free).build()
        mod_name = fn.__module__
        fname = fn.__code__.co_filename

        # Entries exist and re-parsing the live function still works.
        self.assertIn(mod_name, sys.modules)
        self.assertIn(fname, linecache.cache)
        self.assertIsNotNone(workflow_parser.parse_workflow(fn))

        del fn
        gc.collect()

        self.assertNotIn(mod_name, sys.modules)
        self.assertNotIn(fname, linecache.cache)


class TestRenderedSourceDump(unittest.TestCase):
    def setUp(self) -> None:
        # .resolve() so the tmpdir matches dump()'s resolved paths (macOS
        # /var -> /private/var symlink would otherwise break substring checks).
        self.tmpdir = pathlib.Path(
            self.enterContext(tempfile.TemporaryDirectory())
        ).resolve()

        self.rendered = source.RenderedSource(
            source="def f():\n    return 1\n",
            namespace={},
            function_name="f",
        )
        self.rendered_with_namespace = source.RenderedSource(
            source="def g():\n    return helper()\n",
            namespace={"helper": object()},
            function_name="g",
        )

    def test_writes_source_without_namespace(self) -> None:
        target = self.tmpdir / "out.py"
        message = self.rendered.dump(target)

        self.assertEqual(target.read_text(encoding="utf-8"), self.rendered.source)
        self.assertIn(str(target), message)
        self.assertNotIn("Reminder", message)

    def test_writes_source_with_namespace_and_reminds(self) -> None:
        target = self.tmpdir / "out.py"
        message = self.rendered_with_namespace.dump(
            target, allow_namespace_symbols=True
        )

        self.assertEqual(
            target.read_text(encoding="utf-8"),
            self.rendered_with_namespace.source,
        )
        self.assertIn("Reminder", message)
        for symbol in self.rendered_with_namespace.namespace:
            self.assertIn(symbol, message)

    def test_namespace_without_opt_in_raises(self) -> None:
        target = self.tmpdir / "out.py"
        with self.assertRaises(ValueError):
            self.rendered_with_namespace.dump(target)
        self.assertFalse(target.exists())

    def test_missing_extension_becomes_py(self) -> None:
        target = self.tmpdir / "out"  # no suffix
        message = self.rendered.dump(target)

        written = target.with_suffix(".py")
        self.assertTrue(written.exists())
        self.assertEqual(written.read_text(encoding="utf-8"), self.rendered.source)
        self.assertIn(str(written), message)

    def test_wrong_extension_raises(self) -> None:
        target = self.tmpdir / "out.txt"
        with self.assertRaises(ValueError):
            self.rendered.dump(target)
        self.assertFalse(target.exists())

    def test_existing_file_without_exists_ok_raises(self) -> None:
        target = self.tmpdir / "out.py"
        target.write_text("stale", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.rendered.dump(target, exists_ok=False)
        self.assertEqual(target.read_text(encoding="utf-8"), "stale")

    def test_overwrites_existing_by_default(self) -> None:
        target = self.tmpdir / "out.py"
        target.write_text("stale", encoding="utf-8")
        self.rendered.dump(target)
        self.assertEqual(target.read_text(encoding="utf-8"), self.rendered.source)


class TestFunctionBuilderDecorator(unittest.TestCase):
    def test_render_emits_output_labels_in_decorator(self):
        fb = function.FunctionBuilder(
            name="f",
            params=["a"],
            decorator="@foo.bar",
            return_symbols=["a"],
            output_labels=["m", "n"],
        )
        self.assertIn('@foo.bar("m", "n")', fb.render())

    def test_render_bare_decorator_without_labels(self):
        fb = function.FunctionBuilder(
            name="f", decorator="@foo.bar", params=["a"], return_symbols=["a"]
        )
        src = fb.render()
        self.assertIn("@foo.bar\n", src)
        self.assertNotIn("@foo.bar(", src)


class TestNameAllocator(unittest.TestCase):
    def test_fresh_returns_hint_then_suffixes(self):
        alloc = function.NameAllocator()
        self.assertEqual(alloc.fresh("x"), "x")
        self.assertEqual(alloc.fresh("x"), "x_0")
        self.assertEqual(alloc.fresh("x"), "x_1")

    def test_fresh_sanitises_invalid_hint(self):
        alloc = function.NameAllocator()
        out = alloc.fresh("output_0.bad")  # not a valid identifier
        self.assertTrue(out.isidentifier())

    def test_reserve_blocks_later_fresh_collision(self):
        alloc = function.NameAllocator()
        self.assertEqual(alloc.reserve("result"), "result")
        self.assertEqual(alloc.fresh("result"), "result_0")


class TestSingleAtomicDag(unittest.TestCase):
    def _free_recipe(self):
        def one_add(a, b):
            result = std.add(a, b)
            return result

        return one_add, makers.reference_free(one_add)

    def test_executes(self):
        original, free = self._free_recipe()
        fn = source._workflow2python(free).build()
        self.assertEqual(fn(2, 3), original(2, 3))

    def test_round_trips(self):
        _, free = self._free_recipe()
        rendered = source._workflow2python(free)
        fn = rendered.build()
        self.assertEqual(
            makers.dump_no_refs(fn.flowrep_recipe), makers.dump_no_refs(free)
        )

    def test_rendered_source_uses_custom_name(self):
        _, free = self._free_recipe()
        rs = source._workflow2python(free, function_name="my_function").source
        self.assertIn("def my_function", rs)


class TestMultiNodeDag(unittest.TestCase):
    def test_chained_and_unpacked(self):
        def chained(a, b):
            s = std.add(a, b)  # one output
            q, r = library.divmod_func(s, b)  # tuple unpack -> two outputs
            return q, r

        free = makers.reference_free(chained)
        rendered = source._workflow2python(free)
        fn = rendered.build()
        self.assertEqual(fn(7, 3), chained(7, 3))
        self.assertEqual(
            makers.dump_no_refs(fn.flowrep_recipe), makers.dump_no_refs(free)
        )

    def test_emits_in_dependency_order_when_nodes_unordered(self):
        # Build a recipe whose .nodes dict is *not* topologically ordered, by
        # round-tripping a normal one then re-inserting nodes in reverse.
        def chained(a, b):
            s = std.add(a, b)
            t = std.mul(s, b)
            return t

        free = makers.reference_free(chained)
        reordered_nodes = dict(reversed(list(free.nodes.items())))
        scrambled = free.model_copy(update={"nodes": reordered_nodes})
        rendered = source._workflow2python(scrambled)
        fn = rendered.build()
        self.assertEqual(fn(2, 5), chained(2, 5))


class TestSignatureParams(unittest.TestCase):
    def test_defaults_bound_as_live_objects(self):
        sentinel = object()

        def with_default(a, b=sentinel):
            r = library.macro_identity(a)
            return r

        free = makers.reference_free(with_default)
        sig = inspect.signature(with_default)
        rendered = source._workflow2python(free, signature=sig)
        fn = rendered.build()
        # default object survives into __defaults__ as the *same* live object
        self.assertIs(fn.__defaults__[0], sentinel)
        self.assertEqual(fn(5), with_default(5))

    def test_positional_only_and_keyword_only_markers(self):
        def kinds(x, /, y, *, z):
            r = std.add(x, y)
            out = std.add(r, z)
            return out

        free = makers.reference_free(kinds)
        sig = inspect.signature(kinds)
        rendered = source._workflow2python(free, signature=sig)
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
            s = std.add(a, b)
            return s, a  # 'a' is a passthrough output

        free = makers.reference_free(passthrough)
        rendered = source._workflow2python(free)
        fn = rendered.build()
        self.assertEqual(fn(4, 6), passthrough(4, 6))
        self.assertEqual(
            makers.dump_no_refs(fn.flowrep_recipe), makers.dump_no_refs(free)
        )


class TestNestedWorkflowNode(unittest.TestCase):
    def test_reference_free_subworkflow_node(self):
        # Build a recipe whose .nodes contains a reference-free WorkflowRecipe.
        # inner's output variable is named 'added' to match the output port
        # name that std.add exposes (and that free_outer's output_edge
        # references), making the substituted recipe internally consistent.
        def inner(a, b):
            added = std.add(a, b)
            return added

        def outer(a, b):
            r = std.add(a, b)
            return r

        free_outer = makers.reference_free(outer)
        free_inner = makers.reference_free(inner)
        # Replace the atomic node with the reference-free sub-workflow node.
        label = next(iter(free_outer.nodes))
        nodes = dict(free_outer.nodes)
        nodes[label] = free_inner  # workflow node, reference=None
        print(nodes)
        recipe = free_outer.model_copy(update={"nodes": nodes})

        rendered = source._workflow2python(recipe)
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
                v = std.mul(x, k)
                acc.append(v)
            return acc

        free = makers.reference_free(mapper)
        rendered = source._workflow2python(free)
        fn = rendered.build()
        self.assertEqual(fn([1, 2, 3], 10), mapper([1, 2, 3], 10))
        self.assertEqual(
            makers.dump_no_refs(fn.flowrep_recipe), makers.dump_no_refs(free)
        )

    def test_zipped_for(self):
        def zipper(xs, ys):
            acc = []
            for x, y in zip(xs, ys, strict=True):
                v = std.add(x, y)
                acc.append(v)
            return acc

        free = makers.reference_free(zipper)
        fn = source._workflow2python(free).build()
        self.assertEqual(fn([1, 2], [3, 4]), zipper([1, 2], [3, 4]))

    def test_for_each_body_without_underlying_python(self):
        self.assertEqual(
            wfms.run_recipe(workflow_for_each_recipe, xs=[1, 2, 3])
            .output_ports["ys"]
            .value,
            [2, 3, 4],
            msg="Sanity check that the recipe is fine",
        )
        rendered = source._workflow2python(workflow_for_each_recipe)
        fn = rendered.build()
        self.assertEqual(fn([1, 2, 3]), [2, 3, 4])


class TestIf(unittest.TestCase):
    def test_if_else_round_trip_and_exec(self):
        def chooser(a, b):
            if library.my_condition(a, b):  # noqa: SIM108
                # a < b
                v = std.add(a, b)
            else:
                v = std.mul(a, b)
            return v

        free = makers.reference_free(chooser)
        rendered = source._workflow2python(free)
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
        source._workflow2python(workflow_condition_recipe)


class TestWhile(unittest.TestCase):
    def test_while_round_trip_and_exec(self):
        def countup(i, bound):
            while library.my_condition(i, bound):  # i < bound
                i = library.increment(i)
            return i

        free = makers.reference_free(countup)
        rendered = source._workflow2python(free)
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
        rendered = source._workflow2python(workflow_while_recipe)
        fn = rendered.build()
        self.assertEqual(
            fn(3),
            wfms.run_recipe(workflow_while_recipe, i=3).output_ports["i"].value,
            msg="Emitted Python must match the recipe's runtime result",
        )


class TestFlowControlConditionNode(unittest.TestCase):
    def test_if_condition_flow_control_raises(self):
        with self.assertRaises(NotImplementedError) as ctx:
            source._workflow2python(if_flow_control_condition_recipe)
        self.assertIn("flowcond", str(ctx.exception))
        self.assertIn("single callable", str(ctx.exception))

    def test_while_condition_flow_control_raises(self):
        with self.assertRaises(NotImplementedError) as ctx:
            source._workflow2python(while_flow_control_condition_recipe)
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
        fn = source._workflow2python(recipe).build()
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
                z = std.identity(a)
            return z

        free = makers.reference_free(safe_div)
        rendered = source._workflow2python(free)
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
                z = std.identity(a)
            return z

        free = makers.reference_free(custom_exception_branch)
        rendered = source._workflow2python(free)
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
        rendered = source._workflow2python(workflow_try_recipe)
        fn = rendered.build()
        self.assertEqual(fn(6, 3), 2.0)
        self.assertEqual(fn(6, 0), 6)


def _with_default(a, b=10):
    """Module-level workflow function with a default; importable for DagData tests."""
    r = std.add(a, b)
    return r


def _typed_single(a: int, b: float = 2.0) -> float:
    """Module-level typed workflow for DagData annotation tests."""
    r = std.add(a, b)
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
        dagdata = datastructures.DagData.from_recipe(recipe_with_ref)
        rendered = source._dagdata2python(dagdata)
        fn = rendered.build()
        self.assertEqual(fn(5), _with_default(5))
        # default recovered from the port
        self.assertEqual(fn.__defaults__, (10,))

    def test_dagdata_propagates_types_and_default(self):
        dagdata = datastructures.DagData.from_recipe(
            workflow_parser.parse_workflow(_typed_single)
        )
        fn = source._dagdata2python(dagdata).build()
        self.assertEqual(fn(5), _typed_single(5))
        self.assertEqual(fn.__defaults__, (2.0,))
        hints = typing.get_type_hints(fn, include_extras=True)
        self.assertIs(hints["a"], int)
        self.assertIs(hints["b"], float)
        self.assertIs(hints["return"], float)

    def test_dagdata_multi_output_types_round_trip(self):
        free = makers.reference_free(_typed_multi)
        dagdata = datastructures.DagData.from_recipe(
            workflow_parser.parse_workflow(_typed_multi)
        )
        fn = source._dagdata2python(dagdata).build()
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
            source._workflow2python(recipe)

    def test_cycle_raises(self):

        def chained(a, b):
            s = std.add(a, b)
            t = std.mul(s, b)
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
            source._workflow2python(cyclic)

    def test_trailing_positional_only_marker(self):
        def kinds(a, b, /):
            r = std.add(a, b)
            return r

        free = makers.reference_free(kinds)
        rendered = source._workflow2python(free, signature=inspect.signature(kinds))
        self.assertIn("/", rendered.source)
        fn = rendered.build()
        self.assertEqual(
            inspect.signature(fn).parameters["b"].kind,
            inspect.Parameter.POSITIONAL_ONLY,
        )

    def test_for_each_transferred_input_output(self):
        # A zipped input forwarded straight to an output is collected per iteration.
        def fwd(xs, ks):
            acc = []
            kept = []
            for x, k in zip(xs, ks, strict=True):
                v = std.add(x, k)
                acc.append(v)
                kept.append(k)
            return acc, kept

        free = makers.reference_free(fwd)
        fn = source._workflow2python(free).build()
        self.assertEqual(fn([1, 2], [10, 20]), fwd([1, 2], [10, 20]))

    def test_condition_with_defaulted_unsourced_input(self):
        # increment(x, step=1): 'step' is defaulted and left unsourced as a condition.
        def f(a):
            if library.increment(a):  # noqa: SIM108
                r = std.identity(a)
            else:
                r = library.negate(a)
            return r

        free = makers.reference_free(f)
        rendered = source._workflow2python(free)
        self.assertIn("library.increment(x=", rendered.source)
        fn = rendered.build()
        self.assertEqual(fn(5), f(5))
        self.assertEqual(fn(-1), f(-1))

    def test_multi_exception_except_clause(self):
        def f(a, b):
            try:
                z = library.divide(a, b)
            except (ZeroDivisionError, ValueError):
                z = std.identity(a)
            return z

        free = makers.reference_free(f)
        rendered = source._workflow2python(free)
        self.assertIn("except (", rendered.source)
        self.assertEqual(rendered.build()(6, 0), f(6, 0))

    def test_positional_only_gap_raises(self):
        po = base_models.RestrictedParamKind.POSITIONAL_ONLY
        node = atomic_recipe.AtomicRecipe(
            reference=makers.make_reference(
                module="flowrep_static.library",
                qualname="two_positional_only",
                inputs_with_defaults=["a"],
                restricted_input_kinds={"a": po, "b": po},
            ),
            inputs=["a", "b"],
            outputs=["output_0"],
        )
        recipe = workflow_recipe.WorkflowRecipe(
            inputs=["x"],
            outputs=["output_0"],
            nodes={"f_0": node},
            input_edges={
                edge_models.TargetHandle(node="f_0", port="b"): edge_models.InputSource(
                    port="x"
                ),
            },
            edges={},
            output_edges={
                edge_models.OutputTarget(port="output_0"): edge_models.SourceHandle(
                    node="f_0", port="output_0"
                )
            },
            reference=None,
        )
        # 'a' has a default and is left unsourced; only 'b' (a later positional-only
        # parameter) is sourced -> no valid positional-call form, so render must raise.
        # This is an edge case that the python interpreter prevents, but can be reached
        # in manually-constructed workflows such as this one
        with self.assertRaisesRegex(ValueError, "positional-only"):
            source._workflow2python(recipe)

    def test_alias_conflict_raises(self):
        # Two outputs sourced from one handle but pinned to different names cannot
        # be emitted as assignments. Drive the guard directly via _emit_workflow_body.

        body = workflow_recipe.WorkflowRecipe.model_validate(
            {
                "type": "workflow",
                "inputs": ["a", "b"],
                "outputs": ["p", "q"],
                "nodes": {"add_0": std.add.flowrep_recipe},
                "input_edges": {"add_0.a": "a", "add_0.b": "b"},
                "edges": {},
                "output_edges": {
                    "p": "add_0.added",
                    "q": "add_0.added",
                },
            }
        )
        with self.assertRaisesRegex(ValueError, "cannot be emitted as an assignment"):
            statements.emit_workflow_body(
                body,
                {"a": "a", "b": "b"},
                {"p": "P", "q": "Q"},
                function.Emitter(),
                function.NameAllocator(),
            )


class TestAnnotationReconstruction(unittest.TestCase):
    def test_unannotated_outputs_stay_that_way(self):
        # No signature -> Any-typed return annotation, label still pinned.
        def plain(a, b):
            r = std.add(a, b)
            return r

        free = makers.reference_free(plain)
        rendered = source._workflow2python(free)
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
            r = std.add(x, y)
            out = std.add(r, z)
            return out

        free = makers.reference_free(typed)
        rendered = source._workflow2python(free, signature=inspect.signature(typed))
        self.assertIn("x: int", rendered.source)
        self.assertIn("y: list[int]", rendered.source)
        self.assertIn("z: dict[str, int]", rendered.source)
        self.assertEqual(rendered.namespace, {})
        fn = rendered.build()
        sig = inspect.signature(fn)
        self.assertIs(sig.parameters["x"].annotation, int)
        self.assertEqual(sig.parameters["y"].annotation, list[int])
        self.assertEqual(sig.parameters["z"].annotation, dict[str, int])

    def test_annotated_input_preserved_verbatim(self):
        def f(x: typing.Annotated[int, "meta"], y: float):
            r = std.add(x, y)
            return r

        free = makers.reference_free(f)
        rendered = source._workflow2python(free, signature=inspect.signature(f))
        self.assertIn("typing.Annotated[int, 'meta']", rendered.source)
        fn = rendered.build()
        hints = typing.get_type_hints(fn, include_extras=True)
        self.assertEqual(hints["x"], typing.Annotated[int, "meta"])
        self.assertIs(hints["y"], float)

    def test_annotation_and_default_both_survive(self):
        def f(x: int, y: float = 0.5):
            r = std.add(x, y)
            return r

        free = makers.reference_free(f)
        fn = source._workflow2python(free, signature=inspect.signature(f)).build()
        sig = inspect.signature(fn)
        self.assertIs(sig.parameters["y"].annotation, float)
        self.assertEqual(sig.parameters["y"].default, 0.5)

    def test_single_return_annotation_is_verbatim(self):
        def typed(a, b) -> float:
            r = std.add(a, b)
            return r

        free = makers.reference_free(typed)
        fn = source._workflow2python(free, signature=inspect.signature(typed)).build()
        self.assertIs(inspect.signature(fn).return_annotation, float)

    def test_multi_return_annotation_is_verbatim(self):
        def typed(a, b) -> tuple[int, float]:
            q, r = library.divmod_func(a, b)
            return q, r

        free = makers.reference_free(typed)
        fn = source._workflow2python(free, signature=inspect.signature(typed)).build()
        self.assertEqual(typing.get_type_hints(fn)["return"], tuple[int, float])

    def test_mismatched_return_annotation_emitted_verbatim(self):
        # A non-tuple return annotation for a 2-output workflow is emitted verbatim
        # (no splitting, no typing.Any fallback); port names come from the decorator.
        def typed_bad(a, b) -> int:
            q, r = library.divmod_func(a, b)
            return q, r

        free = makers.reference_free(typed_bad)
        rendered = source._workflow2python(free, signature=inspect.signature(typed_bad))
        self.assertIn('@flowrep.workflow("q", "r")', rendered.source)
        self.assertIn("-> int:", rendered.source)
        fn = rendered.build()
        self.assertIs(typing.get_type_hints(fn)["return"], int)
        rebuilt_recipe = workflow_parser.parse_workflow(fn)
        with self.assertRaises(
            ValueError,
            msg="Recipes don't know about annotations, so we can get a recipe from "
            "such a function, but it will fail to convert to a data object because of "
            "the mismatch between ports and the return annotation",
        ):
            datastructures.DagData.from_recipe(rebuilt_recipe)

    def test_output_port_name_pinned_via_decorator(self):
        # Return symbol is "s" (my_add's output) but the port is renamed; the
        # decorator must pin the port name regardless of the return symbol.
        def one(a, b):
            s = std.add(a, b)
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
        rendered = source._workflow2python(renamed)
        self.assertIn('@flowrep.workflow("renamed")', rendered.source)
        fn = rendered.build()
        self.assertEqual(fn.flowrep_recipe.outputs, ["renamed"])

    def test_input_and_return_annotations_do_not_interfere(self):
        def f(
            x: typing.Annotated[int, "in-meta"], y: float
        ) -> typing.Annotated[float, "out-meta"]:
            r = std.add(x, y)
            return r

        free = makers.reference_free(f)
        rendered = source._workflow2python(free, signature=inspect.signature(f))
        self.assertIn('@flowrep.workflow("r")', rendered.source)
        fn = rendered.build()
        hints = typing.get_type_hints(fn, include_extras=True)
        self.assertEqual(hints["x"], typing.Annotated[int, "in-meta"])
        self.assertEqual(hints["return"], typing.Annotated[float, "out-meta"])

    def test_plain_input_annotation_is_stringy(self):
        def typed(x: int, y: float):
            r = std.add(x, y)
            return r

        free = makers.reference_free(typed)
        rendered = source._workflow2python(free, signature=inspect.signature(typed))
        self.assertIn("x: int", rendered.source)
        self.assertIn("y: float", rendered.source)
        self.assertNotIn("_ann_x", rendered.namespace)
        self.assertNotIn("_ann_y", rendered.namespace)
        fn = rendered.build()
        sig = inspect.signature(fn)
        self.assertIs(sig.parameters["x"].annotation, int)
        self.assertIs(sig.parameters["y"].annotation, float)

    def test_literal_default_is_stringy(self):
        def f(x: int, y: float = 0.5):
            r = std.add(x, y)
            return r

        free = makers.reference_free(f)
        rendered = source._workflow2python(free, signature=inspect.signature(f))
        self.assertIn("y: float = 0.5", rendered.source)
        self.assertNotIn("_default_y", rendered.namespace)
        fn = rendered.build()
        self.assertEqual(inspect.signature(fn).parameters["y"].default, 0.5)

    def test_mixed_inline_and_namespace_default(self):
        sentinel = object()

        def f(x: int, y=sentinel):
            r = std.add(x, y)
            return r

        free = makers.reference_free(f)
        rendered = source._workflow2python(free, signature=inspect.signature(f))
        # x annotation inlines; y default cannot, so it stays namespace-bound.
        self.assertIn("x: int", rendered.source)
        self.assertNotIn("_ann_x", rendered.namespace)
        self.assertIn(
            "=_default_y",
            rendered.source,
            msg="Unstringable default should stay in reference form in the source text",
        )
        self.assertIn("_default_y", rendered.namespace)
        self.assertIs(rendered.namespace["_default_y"], sentinel)
        fn = rendered.build()
        self.assertIs(fn.__defaults__[0], sentinel)
        self.assertIs(inspect.signature(fn).parameters["x"].annotation, int)

    def test_return_annotation_is_stringy(self):
        def typed(a, b) -> float:
            r = std.add(a, b)
            return r

        free = makers.reference_free(typed)
        rendered = source._workflow2python(free, signature=inspect.signature(typed))
        self.assertIn("-> float:", rendered.source)
        self.assertNotIn("_ann_return", rendered.namespace)
        fn = rendered.build()
        self.assertIs(inspect.signature(fn).return_annotation, float)

    def test_fully_simple_recipe_has_empty_namespace(self):
        def simple(x: int, y: float = 1.0) -> float:
            r = std.add(x, y)
            return r

        free = makers.reference_free(simple)
        rendered = source._workflow2python(free, signature=inspect.signature(simple))
        self.assertEqual(rendered.namespace, {})
        fn = rendered.build()
        sig = inspect.signature(fn)
        self.assertIs(sig.parameters["x"].annotation, int)
        self.assertEqual(sig.parameters["y"].default, 1.0)
        self.assertIs(sig.return_annotation, float)

    def test_noninlinable_input_annotation_falls_back_to_namespace(self):
        class Custom:  # local class -> cannot be inlined
            pass

        def f(x: Custom):
            r = std.identity(x)
            return r

        free = makers.reference_free(f)
        rendered = source._workflow2python(free, signature=inspect.signature(f))
        self.assertIn("x: _ann_x", rendered.source)
        self.assertIs(rendered.namespace["_ann_x"], Custom)
        fn = rendered.build()
        self.assertIs(inspect.signature(fn).parameters["x"].annotation, Custom)

    def test_noninlinable_return_annotation_falls_back_to_namespace(self):
        class Custom:  # local class -> cannot be inlined
            pass

        def f(a) -> Custom:
            r = std.identity(a)
            return r

        free = makers.reference_free(f)
        rendered = source._workflow2python(free, signature=inspect.signature(f))
        self.assertIn("-> _ann_return:", rendered.source)
        self.assertIs(rendered.namespace["_ann_return"], Custom)
        fn = rendered.build()
        self.assertIs(inspect.signature(fn).return_annotation, Custom)


class TestImportHoisting(unittest.TestCase):
    """Call/exception imports are collected in the module preamble, deduplicated."""

    @staticmethod
    def _two_node_outer():
        # outer(a, b): r = combine(a, b); s = combine(r, b); return s
        # Two chained atomic nodes, both calling library.combine. These stay on the
        # library (rather than std) deliberately: the hoisting/dedup behaviour under
        # test is only observable for a third-party module, since `import flowrep` is
        # always emitted for the decorator regardless of what the nodes call.
        def outer(a, b):
            r = library.combine(a, b)
            s = library.combine(r, b)
            return s

        return makers.reference_free(outer)

    @staticmethod
    def _free_inner():
        # inner(a, b): r = combine(a, b); return r
        # Output named r to match the combine port the outer edges expect.
        def inner(a, b):
            r = library.combine(a, b)
            return r

        return makers.reference_free(inner)

    def _outer_with_one_subworkflow(self):
        # Replace the FIRST node of the 2-node outer with a reference-free
        # sub-workflow. Result: the nested def calls combine, and the top-level
        # function still calls combine directly for the second node.
        free_outer = self._two_node_outer()
        free_inner = self._free_inner()
        first_label = next(iter(free_outer.nodes))
        nodes = dict(free_outer.nodes)
        nodes[first_label] = free_inner
        return free_outer.model_copy(update={"nodes": nodes})

    def test_call_import_is_in_preamble_not_function_body(self):
        recipe = self._outer_with_one_subworkflow()
        src = source._workflow2python(recipe).source
        # Module-level (column-0) import is present...
        self.assertIn("\nimport flowrep_static.library", src)
        # ...and no indented (in-body) import remains.
        self.assertNotIn("    import flowrep_static.library", src)

    def test_duplicate_call_import_is_deduplicated(self):
        recipe = self._outer_with_one_subworkflow()
        src = source._workflow2python(recipe).source
        # Top-level function and nested def both need the library import;
        # it must appear exactly once.
        self.assertEqual(src.count("import flowrep_static.library"), 1)

    def test_nested_function_import_raises_to_top_level_preamble(self):
        recipe = self._outer_with_one_subworkflow()
        rendered = source._workflow2python(recipe)
        src = rendered.source
        # The import must appear before the first generated def/decorator,
        # i.e. in the preamble, even though one consumer is a nested def.
        first_def = src.index("@flowrep.workflow")
        import_pos = src.index("import flowrep_static.library")
        self.assertLess(import_pos, first_def)
        # And it still builds and round-trips.
        fn = rendered.build()
        self.assertEqual(
            makers.dump_no_refs(fn.flowrep_recipe), makers.dump_no_refs(recipe)
        )


class TestPublicAccess(unittest.TestCase):
    def setUp(self):
        self.recipe = makers.reference_free(makers.make_simple_workflow_recipe())
        self.dag = datastructures.DagData.from_recipe(self.recipe)

    def test_multiple_dispatch(self):
        for data in (self.recipe, self.dag):
            with self.subTest(data=data):
                rendered = source.flowrep2python(data)
                fn = rendered.build()
                self.assertEqual(fn(1, 2), _with_default(1, 2))

    def test_signature_to_dag_raises(self):
        with self.assertRaises(
            ValueError, msg="DagData infers signature information from port fields"
        ):
            source.flowrep2python(self.dag, signature="anything other than None")

    def test_unknown_type_raises(self):
        with self.assertRaises(TypeError):
            source.flowrep2python("not a flowrep recipe or data object")


class TestModuleNames(unittest.TestCase):
    """Module-scope name allocator: collector, def names, namespace, reservations."""

    def test_referenced_top_level_bindings_collects_and_skips_builtins(self):
        def safe_div(a, b):
            try:
                z = library.divide(a, b)
            except ZeroDivisionError:
                z = std.identity(a)
            return z

        def custom(a, b):
            try:
                z = library.raises_custom(a, b)
            except library.MyCustomException:
                z = std.identity(a)
            return z

        free_safe = makers.reference_free(safe_div)
        free_custom = makers.reference_free(custom)

        # All calls live in flowrep_static.library -> top binding "flowrep_static".
        # Or in the standard library
        # ZeroDivisionError is a builtin and must be skipped (no import emitted).
        self.assertEqual(
            set(function.referenced_top_level_bindings(free_safe)),
            {"flowrep_static", "flowrep"},
        )
        self.assertNotIn(
            "builtins", set(function.referenced_top_level_bindings(free_safe))
        )
        # The non-builtin custom exception still resolves to flowrep_static/flowrep (std).
        self.assertEqual(
            set(function.referenced_top_level_bindings(free_custom)),
            {"flowrep_static", "flowrep"},
        )

    def test_nested_subworkflow_def_names_do_not_collide(self):
        # Outer reference-free sub-workflow whose body is itself a reference-free
        # sub-workflow; both derive base label "add". Independent per-function
        # allocators mint "add" twice -> two module-level `def add`, the
        # second shadows the first -> the inner call recurses into itself.
        def innermost(a, b):
            added = std.add(a, b)
            return added

        def middle(a, b):
            added = std.add(a, b)
            return added

        def outer(a, b):
            r = std.add(a, b)
            return r

        free_inner = makers.reference_free(innermost)
        free_mid = makers.reference_free(middle)
        free_outer = makers.reference_free(outer)

        mlabel = next(iter(free_mid.nodes))
        mnodes = dict(free_mid.nodes)
        mnodes[mlabel] = free_inner
        free_mid = free_mid.model_copy(update={"nodes": mnodes})

        olabel = next(iter(free_outer.nodes))
        onodes = dict(free_outer.nodes)
        onodes[olabel] = free_mid
        recipe = free_outer.model_copy(update={"nodes": onodes})

        rendered = source._workflow2python(recipe)
        src = rendered.source
        self.assertIn("def add(", src)
        self.assertIn("def add_0(", src)
        def_names = re.findall(r"^def (\w+)", src, re.MULTILINE)
        self.assertEqual(
            len(def_names), len(set(def_names)), f"duplicate def names: {def_names}"
        )
        fn = rendered.build()
        self.assertEqual(fn(2, 3), 5)

    def test_sibling_subworkflow_labels_round_trip(self):
        # Two sibling reference-free sub-workflows sharing base "add" -> labels
        # add_0, add_1. The emitter must restore each nested def's __name__ to
        # the base so re-parsing reconstructs the same labels.
        def a_node(a, b):
            output_0 = std.add(a, b)
            return output_0

        def b_node(a, b):
            output_0 = std.add(a, b)
            return output_0

        free_a = makers.reference_free(a_node)
        free_b = makers.reference_free(b_node)
        nodes = {"add_0": free_a, "add_1": free_b}
        input_edges = {
            edge_models.TargetHandle(node="add_0", port="a"): edge_models.InputSource(
                port="p"
            ),
            edge_models.TargetHandle(node="add_0", port="b"): edge_models.InputSource(
                port="q"
            ),
            edge_models.TargetHandle(node="add_1", port="b"): edge_models.InputSource(
                port="q"
            ),
        }
        edges = {
            edge_models.TargetHandle(node="add_1", port="a"): edge_models.SourceHandle(
                node="add_0", port="output_0"
            ),
        }
        output_edges = {
            edge_models.OutputTarget(port="output_0"): edge_models.SourceHandle(
                node="add_1", port="output_0"
            )
        }
        recipe = workflow_recipe.WorkflowRecipe(
            inputs=["p", "q"],
            outputs=["output_0"],
            nodes=nodes,
            input_edges=input_edges,
            edges=edges,
            output_edges=output_edges,
            reference=None,
        )
        rebuilt = source._workflow2python(recipe).build()
        reparsed = workflow_parser.parse_workflow(rebuilt)
        self.assertEqual(list(recipe.nodes), list(reparsed.nodes))

    @staticmethod
    def _relabel_only_node(recipe, new_label):
        from flowrep import edge_models

        (old_label,) = recipe.nodes
        node = recipe.nodes[old_label]

        def retarget(target):
            if target.node == old_label:
                return edge_models.TargetHandle(node=new_label, port=target.port)
            return target

        def resource(source):
            if getattr(source, "node", None) == old_label:
                return edge_models.SourceHandle(node=new_label, port=source.port)
            return source

        return recipe.model_copy(
            update={
                "nodes": {new_label: node},
                "input_edges": {retarget(t): s for t, s in recipe.input_edges.items()},
                "output_edges": {
                    o: resource(s) for o, s in recipe.output_edges.items()
                },
            }
        )

    def _subworkflow_labeled(self, label):
        # Calls library.combine so that a third-party `import flowrep_static.library`
        # binding actually exists for the nested def to shadow.
        def inner(a, b):
            r = library.combine(a, b)
            return r

        def outer(a, b):
            r = library.combine(a, b)
            return r

        free_outer = makers.reference_free(outer)
        free_inner = makers.reference_free(inner)
        olabel = next(iter(free_outer.nodes))
        nodes = dict(free_outer.nodes)
        nodes[olabel] = free_inner
        recipe = free_outer.model_copy(update={"nodes": nodes})
        return self._relabel_only_node(recipe, label)

    def test_def_names_dodge_import_bindings(self):
        # "flowrep" (always-reserved base import) and "flowrep_static" (dynamic
        # pre-scan from the library calls) must not be taken by a nested def,
        # else they shadow `import flowrep` / `import flowrep_static.library`.
        for label, module_binding in (
            ("flowrep", "import flowrep\n"),
            ("flowrep_static", "import flowrep_static.library"),
        ):
            with self.subTest(label=label):
                recipe = self._subworkflow_labeled(label)
                rendered = source._workflow2python(recipe)
                self.assertNotIn(
                    f"\ndef {label}(",
                    rendered.source,
                    msg=f"nested def must not be bare '{label}'",
                )
                self.assertIn(module_binding, rendered.source)
                fn = rendered.build()  # raises if the import was shadowed
                self.assertEqual(fn(2, 3), 5)

    def test_namespace_symbols_are_module_unique(self):
        # Emit two signature-bearing functions through ONE emitter (the future
        # nested-signature case). A locally-defined class is un-inlineable, so
        # render_annotation returns None and the annotation is namespace-bound.
        # Both must survive under distinct keys, not clobber each other.
        class Custom:  # local qualname -> render_annotation returns None
            pass

        def f(a, b):
            r = std.add(a, b)
            return r

        recipe = makers.reference_free(f)
        sig = inspect.Signature(
            parameters=[
                inspect.Parameter(
                    "a", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=Custom
                ),
                inspect.Parameter("b", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            ],
            return_annotation=Custom,
        )

        emitter = function.Emitter()
        b1 = function.emit_workflow_function(recipe, "f1", emitter, sig)
        b2 = function.emit_workflow_function(recipe, "f2", emitter, sig)

        return_keys = [k for k in emitter.namespace if k.startswith("_ann_return")]
        self.assertEqual(len(return_keys), 2, emitter.namespace)
        self.assertNotEqual(b1.return_annotation, b2.return_annotation)
        self.assertIn(b1.return_annotation, emitter.namespace)
        self.assertIn(b2.return_annotation, emitter.namespace)

    def test_custom_decorator(self):
        free = makers.reference_free(_with_default)
        src = source._workflow2python(free, _workflow_decorator=("foo", "bar")).source
        self.assertIn("import foo", src, msg="decorator module should be imported")
        self.assertIn("@foo.bar", src)


class TestLoopVariableReservation(unittest.TestCase):
    def test_loop_variable_does_not_shadow_outer_symbol(self):
        # 'labeled_x' has output port 'x'; the for-loop body port is also 'x'. The
        # outer symbol must be allocated away from the loop variable, else the
        # post-loop 'combine' call reads the last loop value instead of labeled_x's
        # result.
        # Reference: wfms.run_recipe gives combine(labeled_x(10), 10) = (21, [2,3,4]).
        # Without the fix the emitter aliases the outer 'x' to the loop variable and
        # 'combine' picks up the last iteration value, giving (13, [2, 3, 4]) instead.
        # wf(...) cannot be the reference here: the original parsed function shares
        # this recipe and shadows 'x' the same way, so it would mask the bug.
        # run_recipe executes the recipe graph directly, bypassing code generation.
        @workflow_parser.workflow
        def wf(seed, xs):
            x = library.labeled_x(seed)
            acc = []
            for x in xs:
                y = library.loop_inc(x)
                acc.append(y)
            z = library.combine(x, seed)
            return z, acc

        free = makers.reference_free(wf)
        ref = wfms.run_recipe(free, seed=10, xs=[1, 2, 3])
        expected = (ref.output_ports["z"].value, ref.output_ports["acc"].value)
        rebuilt = source._workflow2python(free).build()
        self.assertEqual(rebuilt(seed=10, xs=[1, 2, 3]), expected)

    def test_nested_for_loop_variables_all_reserved(self):
        @workflow_parser.workflow
        def wf(rows):
            out = []
            for row in rows:
                inner = []
                for cell in row:
                    c = library.loop_inc(cell)
                    inner.append(c)
                out.append(inner)
            return out

        recipe = makers.reference_free(wf)
        reserved = function._inlined_pinned_symbols(recipe)
        self.assertIn("row", reserved)
        self.assertIn("cell", reserved)
        rebuilt = source._workflow2python(recipe).build()
        self.assertEqual(rebuilt(rows=[[1, 2], [3]]), wf([[1, 2], [3]]))

    def test_loop_variable_shadow_guard_raises(self):
        # A *pinned* symbol (one the allocator cannot move) named after a loop
        # variable cannot be emitted safely; the guard must raise.
        @workflow_parser.workflow
        def wf(xs):
            acc = []
            for x in xs:
                y = library.loop_inc(x)
                acc.append(y)
            return acc

        for_node = wf.flowrep_recipe.nodes["for_each_0"]
        self.assertIn("x", for_node.iterated_ports)
        with self.assertRaises(ValueError):
            flow_control._guard_loop_variable_shadowing(
                for_node, {("other_0", "output_0"): "x"}
            )


class TestSymbolNaming(unittest.TestCase):
    def test_single_output_symbol_uses_label_base(self):
        free = makers.reference_free(makers.make_simple_workflow_recipe())
        src = source._workflow2python(free).source
        # Node label "add_0" -> symbol "add" (not "output_0").
        self.assertIn("add = ", src)
        self.assertIn("return add", src)
        self.assertNotIn("output_0 = ", src)

    def test_multi_output_symbols_use_label_and_port(self):
        @workflow_parser.workflow
        def wf(v):
            lo, hi = library.split_pair(v)
            s = std.add(lo, hi)
            return s

        free = makers.reference_free(wf)
        src = source._workflow2python(free).source
        self.assertIn("split_pair_lo, split_pair_hi = ", src)
        rebuilt = source._workflow2python(free).build()
        self.assertEqual(rebuilt(v=5), wf(5))

    def test_pinned_name_wins_over_label_base(self):
        # A for-loop's collection symbol is pinned to the for-node input port name
        # ("data"), so the producing node keeps "data", not its label base.
        @workflow_parser.workflow
        def wf(seed):
            data = library.make_list(seed)
            acc = []
            for item in data:
                y = library.loop_inc(item)
                acc.append(y)
            return acc

        free = makers.reference_free(wf)
        src = source._workflow2python(free).source
        self.assertIn("data = ", src)
        self.assertNotIn("make_list = ", src)
        rebuilt = source._workflow2python(free).build()
        self.assertEqual(rebuilt(seed=3), wf(3))

    def test_single_node_branch_body_uses_label_base(self):
        # _emit_single_node_body is called for bare atomic branch bodies. When an
        # output is NOT pinned (required dict empty), the label base is used.
        # The real pipeline always pins branch/loop body outputs to the enclosing
        # flow-control node's shared symbols, so `required` is never empty in
        # production; a direct call is the only way to exercise the unforced path.
        node = library.loop_inc.flowrep_recipe
        alloc = function.NameAllocator()
        emitter = function.Emitter()
        lines, out_syms = statements._emit_single_node_body(
            node,
            "loop_inc_0",
            {"x": "x"},
            {},
            alloc,
            emitter,
        )
        # Output port is "output_0"; with label "loop_inc_0" the hint is "loop_inc".
        self.assertIn("loop_inc", out_syms.values())
        self.assertIn("loop_inc = ", lines[0])


class TestConstantInlining(unittest.TestCase):
    def _ke_recipe(self):
        def kinetic_energy(mass, velocity):
            v_2 = std.mul(velocity, velocity)
            mv_2 = std.mul(mass, v_2)
            ke = std.mul(0.5, mv_2)
            return ke

        return kinetic_energy, makers.reference_free(kinetic_energy)

    def test_inlines_literal_and_omits_assignment(self):
        _, free = self._ke_recipe()
        rendered = source._workflow2python(free)
        self.assertIn("0.5", rendered.source)
        self.assertNotIn("constant_0 =", rendered.source)

    def test_executes_with_inlined_constant(self):
        original, free = self._ke_recipe()
        fn = source._workflow2python(free).build()
        self.assertAlmostEqual(fn(2.0, 3.0), original(2.0, 3.0))

    def test_round_trips(self):
        _, free = self._ke_recipe()
        fn = source._workflow2python(free).build()
        self.assertEqual(
            makers.dump_no_refs(fn.flowrep_recipe), makers.dump_no_refs(free)
        )

    def test_compound_and_type_fidelity_round_trip(self):
        def uses_compound(x):
            y = std.mul(x, [1.0, 1, "1", {"key": [42]}])
            return y

        free = makers.reference_free(uses_compound)
        fn = source._workflow2python(free).build()
        self.assertEqual(
            makers.dump_no_refs(fn.flowrep_recipe), makers.dump_no_refs(free)
        )
        # int/float distinction survives the source round-trip
        reparsed_const = fn.flowrep_recipe.nodes["constant_0"].constant
        self.assertIs(type(reparsed_const[0]), float)
        self.assertIs(type(reparsed_const[1]), int)


class TestConstantMaterialize(unittest.TestCase):
    TH, IS, SH, OT = (
        edge_models.TargetHandle,
        edge_models.InputSource,
        edge_models.SourceHandle,
        edge_models.OutputTarget,
    )

    def _assert_canonical_round_trips(self, rendered):
        """`rendered` (from a hand-built recipe) must build, and the parsed
        canonical recipe must then compile/parse idempotently."""
        fn = rendered.build()
        canonical = fn.flowrep_recipe.model_copy(update={"reference": None})
        rebuilt = source._workflow2python(canonical).build()
        self.assertEqual(
            makers.dump_no_refs(rebuilt.flowrep_recipe),
            makers.dump_no_refs(fn.flowrep_recipe),
            msg="Canonical (post-parse) recipe must round-trip exactly",
        )
        return fn

    def test_constant_feeding_output_wraps_in_identity(self):
        wf = workflow_recipe.WorkflowRecipe(
            inputs=["a"],
            outputs=["y"],
            nodes={"constant_0": constant_recipe.ConstantRecipe(constant=5)},
            input_edges={},
            edges={},
            output_edges={
                self.OT(port="y"): self.SH(node="constant_0", port="constant")
            },
        )
        rendered = source._workflow2python(wf, function_name="returns_const")
        self.assertIn("identity(", rendered.source)
        self.assertNotIn("return 5", rendered.source)  # not inlined as a bare return
        self.assertNotIn("constant = 5", rendered.source)  # not a bare assignment
        fn = self._assert_canonical_round_trips(rendered)
        self.assertEqual(fn(99), 5)

    def test_constant_in_flow_control_body_wraps_in_identity(self):
        # Hand-built `if is_positive(n): m = 42 else: m = -1` with BARE ConstantRecipe
        # bodies (a shape only hand-construction produces).
        if_node = if_recipe.IfRecipe(
            inputs=["n"],
            outputs=["m"],
            cases=[
                helper_models.ConditionalCase(
                    condition=helper_models.LabeledRecipe(
                        label="cond", recipe=library.is_positive.flowrep_recipe
                    ),
                    body=helper_models.LabeledRecipe(
                        label="body",
                        recipe=constant_recipe.ConstantRecipe(constant=42),
                    ),
                )
            ],
            input_edges={self.TH(node="cond", port="n"): self.IS(port="n")},
            prospective_output_edges={
                self.OT(port="m"): [
                    self.SH(node="body", port="constant"),
                    self.SH(node="else_body", port="constant"),
                ]
            },
            else_case=helper_models.LabeledRecipe(
                label="else_body",
                recipe=constant_recipe.ConstantRecipe(constant=-1),
            ),
        )
        wf = workflow_recipe.WorkflowRecipe(
            inputs=["n"],
            outputs=["m"],
            nodes={"if_0": if_node},
            input_edges={self.TH(node="if_0", port="n"): self.IS(port="n")},
            edges={},
            output_edges={self.OT(port="m"): self.SH(node="if_0", port="m")},
        )
        rendered = source._workflow2python(wf, function_name="branchy")
        self.assertIn("identity(", rendered.source)
        self.assertNotIn("m = 42", rendered.source)  # not a bare assignment
        fn = self._assert_canonical_round_trips(rendered)
        self.assertEqual(fn(5), 42)
        self.assertEqual(fn(-3), -1)


class TestConstantsInConditions(unittest.TestCase):
    """Literal constants in if/elif/while conditions: execution + exact round-trip."""

    def _round_trip(self, fn, *call_args):
        free = makers.reference_free(fn)
        rendered = source._workflow2python(free)
        built = rendered.build()
        self.assertAlmostEqual(built(*call_args), fn(*call_args))
        self.assertEqual(
            makers.dump_no_refs(built.flowrep_recipe), makers.dump_no_refs(free)
        )
        return free

    def test_if_condition_literal_round_trips_and_executes(self):
        def wf(m):
            if library.my_condition(m, 0.5):  # noqa: SIM108
                y = std.identity(m)
            else:
                y = std.identity(m)
            return y

        free = self._round_trip(wf, 0.2)
        # the inlined literal appears in the rendered condition, not a bare symbol
        rendered_src = source._workflow2python(free).source
        self.assertIn("0.5", rendered_src)

    def test_elif_condition_literal_round_trips_and_executes(self):
        def wf(x, y):
            if library.my_condition(x, y):  # noqa: SIM108
                z = std.identity(x)
            elif library.my_condition(x, 5):
                z = std.identity(y)
            else:
                z = std.identity(x)
            return z

        self._round_trip(wf, 10, 3)
        self._round_trip(wf, 3, 10)

    def test_while_condition_literal_round_trips_and_executes(self):
        def wf(x):
            while library.my_condition(x, 5):
                x = std.add(x, 1)
            return x

        self._round_trip(wf, 0)

    def test_multiple_literals_in_one_condition_round_trip(self):
        def wf(m):
            if library.my_condition(0.3, 0.5):  # noqa: SIM108
                y = std.identity(m)
            else:
                y = std.identity(m)
            return y

        free = self._round_trip(wf, 7)
        constant_values = sorted(
            node.constant
            for node in free.nodes.values()
            if isinstance(node, constant_recipe.ConstantRecipe)
        )
        self.assertEqual(constant_values, [0.3, 0.5])

    def test_nested_condition_literal_round_trips(self):
        def wf(m):
            if library.my_condition(m, 0.5):
                if library.my_condition(m, 0.7):  # noqa: SIM108
                    y = std.identity(m)
                else:
                    y = std.identity(m)
            else:
                y = std.identity(m)
            return y

        self._round_trip(wf, 0.2)


class TestAttributeSugar(unittest.TestCase):
    def test_single_access_as_call_argument(self):
        def wf(x0: int, comp: library.ComplexData):
            dc = library.MyDataclass(comp, x0)
            r = std.identity(dc.x)
            return r

        free = makers.reference_free(wf)
        rendered = source._workflow2python(free)
        self.assertIn(".x", rendered.source)
        self.assertNotIn("_getattr_wrapper", rendered.source)
        fn = rendered.build()
        self.assertEqual(fn(3, library.ComplexData(val=7)), 3)
        self.assertEqual(
            makers.dump_no_refs(fn.flowrep_recipe), makers.dump_no_refs(free)
        )

    def test_chain(self):
        def wf(x0: int, comp: library.ComplexData):
            dc = library.MyDataclass(comp, x0)
            v = dc.a.val
            return v

        free = makers.reference_free(wf)
        rendered = source._workflow2python(free)
        # Each link of the chain is its own statement: a getattr node is never
        # inlined into a consumer, because that would move its name-constant
        # relative to the workflow's other constants and permute the shared
        # `constant_N` counter on re-parse.
        self.assertRegex(rendered.source, r"\n\s*(\w+) = \w+\.a\n\s*\w+ = \1\.val\n")
        self.assertNotIn(".a.val", rendered.source)
        fn = rendered.build()
        self.assertEqual(fn(3, library.ComplexData(val=7)), 7)
        self.assertEqual(
            makers.dump_no_refs(fn.flowrep_recipe), makers.dump_no_refs(free)
        )

    def test_access_interleaved_with_a_literal_round_trips(self):
        """Regression: a getattr's name-constant must not be reordered.

        Constant labels come from one shared `constant_N` counter over the whole
        workflow. Inlining `dc.a` into the `identity` call would defer its name
        constant past the literal `3`, swapping `constant_0` and `constant_1` on
        re-parse -- functionally identical, structurally different.
        """

        def wf(comp: library.ComplexData, n: int):
            dc = library.MyDataclass(comp, n)
            v = dc.a
            w = library.increment(3)
            r = std.identity(v)
            return r, w

        free = makers.reference_free(wf)
        fn = source._workflow2python(free).build()
        self.assertEqual(
            makers.dump_no_refs(fn.flowrep_recipe), makers.dump_no_refs(free)
        )

    def test_access_feeding_output_materializes_and_round_trips(self):
        def wf(x0: int, comp: library.ComplexData):
            dc = library.MyDataclass(comp, x0)
            v = dc.a
            return v

        free = makers.reference_free(wf)
        rendered = source._workflow2python(free)
        self.assertRegex(rendered.source, r"\n\s*\w+ = \w+\.a\n")
        return_lines = [
            ln for ln in rendered.source.splitlines() if ln.strip().startswith("return")
        ]
        self.assertEqual(len(return_lines), 1)
        self.assertNotIn(".", return_lines[0])
        fn = rendered.build()
        result = fn(3, library.ComplexData(val=7))
        self.assertEqual(result.val, 7)
        self.assertEqual(
            makers.dump_no_refs(fn.flowrep_recipe), makers.dump_no_refs(free)
        )

    def test_fan_out_materializes_exactly_once(self):
        def wf(x0: int, comp: library.ComplexData):
            dc = library.MyDataclass(comp, x0)
            v = dc.x
            p = std.identity(v)
            q = library.negate(v)
            return p, q

        free = makers.reference_free(wf)
        rendered = source._workflow2python(free)
        assignment_lines = [
            ln
            for ln in rendered.source.splitlines()
            if re.fullmatch(r"\w+ = \w+\.x", ln.strip())
        ]
        self.assertEqual(len(assignment_lines), 1)
        fn = rendered.build()
        self.assertEqual(fn(3, library.ComplexData(val=1)), (3, -3))
        rebuilt = fn.flowrep_recipe
        getattr_nodes = [n for n in rebuilt.nodes.values() if sugar.is_std_getattr(n)]
        self.assertEqual(len(getattr_nodes), 1)
        self.assertEqual(makers.dump_no_refs(rebuilt), makers.dump_no_refs(free))

    def test_access_appended_to_accumulator_pins_body_symbol(self):
        def wf(items: list):
            xs = []
            for item in items:
                dc = library.MyDataclass(item, 1)
                inner = dc.a
                xs.append(inner)
            return xs

        free = makers.reference_free(wf)
        rendered = source._workflow2python(free)
        self.assertRegex(rendered.source, r"\binner = \w+\.a\b")
        self.assertIn("xs.append(inner)", rendered.source)
        fn = rendered.build()
        result = fn([library.ComplexData(val=1), library.ComplexData(val=2)])
        self.assertEqual([r.val for r in result], [1, 2])
        self.assertEqual(
            makers.dump_no_refs(fn.flowrep_recipe), makers.dump_no_refs(free)
        )

    def test_access_directly_on_workflow_input(self):
        def wf(comp: library.ComplexData):
            v = comp.val
            return v

        free = makers.reference_free(wf)
        rendered = source._workflow2python(free)
        self.assertIn(".val", rendered.source)
        fn = rendered.build()
        self.assertEqual(fn(library.ComplexData(val=9)), 9)
        self.assertEqual(
            makers.dump_no_refs(fn.flowrep_recipe), makers.dump_no_refs(free)
        )

    def test_getattr_with_unwired_obj_is_not_sugared(self):
        TH, SH, OT = (
            edge_models.TargetHandle,
            edge_models.SourceHandle,
            edge_models.OutputTarget,
        )
        # WorkflowRecipe validates that every required input has a source, so an
        # unwired `obj` (no default on _getattr_wrapper) cannot be constructed
        # directly. Build a valid recipe first, then strip the edge via
        # model_copy, which -- like the analogous shape in test_sugar.py -- skips
        # validators and produces a shape only hand-construction can reach.
        wf = workflow_recipe.WorkflowRecipe(
            inputs=[],
            outputs=["attr"],
            nodes={
                "getattr_0": std.get_attr.flowrep_recipe,
                "constant_obj": constant_recipe.ConstantRecipe(constant="hi"),
                "constant_0": constant_recipe.ConstantRecipe(constant="val"),
            },
            input_edges={},
            edges={
                TH(node="getattr_0", port="obj"): SH(
                    node="constant_obj", port="constant"
                ),
                TH(node="getattr_0", port="name"): SH(
                    node="constant_0", port="constant"
                ),
            },
            output_edges={OT(port="attr"): SH(node="getattr_0", port="attr")},
        )
        stripped_edges = {
            target: sh
            for target, sh in wf.edges.items()
            if not (target.node == "getattr_0" and target.port == "obj")
        }
        wf = wf.model_copy(update={"edges": stripped_edges})
        with self.assertRaises(ValueError):
            source._workflow2python(wf, function_name="unwired_obj")

    def test_getattr_obj_fed_by_inlined_constant_is_not_sugared(self):
        TH, SH, OT = (
            edge_models.TargetHandle,
            edge_models.SourceHandle,
            edge_models.OutputTarget,
        )
        wf = workflow_recipe.WorkflowRecipe(
            inputs=[],
            outputs=["attr"],
            nodes={
                "constant_obj": constant_recipe.ConstantRecipe(constant=3),
                "constant_0": constant_recipe.ConstantRecipe(constant="real"),
                "getattr_0": std.get_attr.flowrep_recipe,
            },
            input_edges={},
            edges={
                TH(node="getattr_0", port="obj"): SH(
                    node="constant_obj", port="constant"
                ),
                TH(node="getattr_0", port="name"): SH(
                    node="constant_0", port="constant"
                ),
            },
            output_edges={OT(port="attr"): SH(node="getattr_0", port="attr")},
        )
        rendered = source._workflow2python(wf, function_name="const_obj_getattr")
        self.assertIn("get_attr", rendered.source)
        fn = rendered.build()
        self.assertEqual(fn(), 3)

    def test_hand_built_non_sugarable_getattr_emits_call_and_executes(self):
        TH, IS, SH, OT = (
            edge_models.TargetHandle,
            edge_models.InputSource,
            edge_models.SourceHandle,
            edge_models.OutputTarget,
        )
        wf = workflow_recipe.WorkflowRecipe(
            inputs=["obj_in", "name_in"],
            outputs=["attr"],
            nodes={
                "identity_0": std.identity.flowrep_recipe,
                "getattr_0": std.get_attr.flowrep_recipe,
            },
            input_edges={
                TH(node="identity_0", port="x"): IS(port="name_in"),
                TH(node="getattr_0", port="obj"): IS(port="obj_in"),
            },
            edges={
                TH(node="getattr_0", port="name"): SH(node="identity_0", port="x"),
            },
            output_edges={OT(port="attr"): SH(node="getattr_0", port="attr")},
        )
        rendered = source._workflow2python(wf, function_name="nonsugar")
        self.assertIn("get_attr", rendered.source)
        fn = rendered.build()
        self.assertEqual(fn(library.ComplexData(val=42), "val"), 42)

    def test_bound_access_as_condition_input_round_trips(self):
        def wf(x0: int, comp: library.ComplexData):
            dc = library.MyDataclass(comp, x0)
            flag = dc.x
            if library.is_positive(flag):  # noqa: SIM108
                y = std.identity(x0)
            else:
                y = library.negate(x0)
            return y

        free = makers.reference_free(wf)
        rendered = source._workflow2python(free)
        self.assertRegex(rendered.source, r"\bflag = \w+\.x\b")
        fn = rendered.build()
        self.assertEqual(fn(3, library.ComplexData(val=7)), 3)
        self.assertEqual(fn(-3, library.ComplexData(val=7)), 3)
        self.assertEqual(
            makers.dump_no_refs(fn.flowrep_recipe), makers.dump_no_refs(free)
        )


class TestPinnedSymbolReservation(unittest.TestCase):
    """The allocator must know every pinned name *before* it mints anything.

    A pin bypasses the allocator, and a pin emitted later cannot retroactively protect a
    name the allocator already minted -- so an unreserved pin silently overwrites it.
    """

    def test_flow_control_input_ports_are_reserved(self):
        @workflow_parser.workflow
        def wf(comp, seed):
            a = library.val(1)
            b = library.val(2)
            if library.is_positive(comp.val):
                m = std.identity(seed)
            else:
                m = library.negate(seed)
            c = std.add(a, b)
            return m, c

        recipe = makers.reference_free(wf)
        self.assertIn(
            "val_0",
            function._inlined_pinned_symbols(recipe),
            "the if-node's generated input port pins a getattr to `val_0`",
        )

    def test_for_body_output_ports_are_reserved(self):
        @workflow_parser.workflow
        def wf(payload, comp):
            ys = []
            for n in payload.xs:
                d = library.MyDataclass(comp, n)
                ys.append(d.x)
            return ys

        recipe = makers.reference_free(wf)
        reserved = function._inlined_pinned_symbols(recipe)
        self.assertIn("n", reserved, "loop variable")
        self.assertIn("x_0", reserved, "the for-body's generated output port")
        self.assertIn("ys", reserved, "the for node's output port")

    def test_allocator_does_not_mint_over_a_pin(self):
        """The regression itself: `b` must survive the loop that follows it."""

        @workflow_parser.workflow
        def wf(comp, seed):
            a = library.val(1)
            b = library.val(2)
            if library.is_positive(comp.val):
                m = std.identity(seed)
            else:
                m = library.negate(seed)
            c = std.add(a, b)
            return m, c

        recipe = makers.reference_free(wf)
        rebuilt = source._workflow2python(recipe).build()
        comp = library.ComplexData(val=7)
        self.assertEqual(rebuilt(comp, 4), wf(comp, 4))


if __name__ == "__main__":
    unittest.main()
