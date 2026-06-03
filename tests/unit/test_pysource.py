import dataclasses
import inspect
import typing
import unittest

from flowrep import edge_models, pysource, retrospective
from flowrep.nodes import workflow_recipe
from flowrep.parsers import atomic_parser, workflow_parser

from flowrep_static import library, makers


@atomic_parser.atomic
def _pos_only_add(a, b, /):
    """Module-level atomic with positional-only inputs (importable by qualname)."""
    return a + b


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
        self.assertEqual(hints["return"], typing.Annotated[float, {"label": "r"}])

    def test_dagdata_multi_output_types_round_trip(self):
        free = makers.reference_free(_typed_multi)
        dagdata = retrospective.DagData.from_recipe(
            workflow_parser.parse_workflow(_typed_multi)
        )
        fn = pysource.dagdata2python("rebuilt", dagdata).build()
        self.assertEqual(fn(7.0, 3.0), _typed_multi(7.0, 3.0))
        self.assertEqual(
            typing.get_type_hints(fn, include_extras=True)["return"],
            tuple[
                typing.Annotated[int, {"label": "q"}],
                typing.Annotated[float, {"label": "r"}],
            ],
        )
        self.assertEqual(
            makers.dump_no_refs(fn.flowrep_recipe), makers.dump_no_refs(free)
        )


class TestTypeAnnotations(unittest.TestCase):
    def test_input_and_single_return_annotations_propagate(self):
        def typed(a: int, b: float = 2.0) -> float:
            r = library.my_add(a, b)
            return r

        free = makers.reference_free(typed)
        fn = pysource.recipe2python("rebuilt", free, inspect.signature(typed)).build()
        self.assertEqual(fn(3), typed(3))
        hints = typing.get_type_hints(fn, include_extras=True)
        self.assertIs(hints["a"], int)
        self.assertIs(hints["b"], float)
        self.assertEqual(hints["return"], typing.Annotated[float, {"label": "r"}])

    def test_multi_output_return_annotations_propagate(self):
        def typed2(a, b) -> tuple[int, float]:
            q, r = library.divmod_func(a, b)
            return q, r

        free = makers.reference_free(typed2)
        fn = pysource.recipe2python("rebuilt", free, inspect.signature(typed2)).build()
        self.assertEqual(fn(7, 3), typed2(7, 3))
        self.assertEqual(
            typing.get_type_hints(fn, include_extras=True)["return"],
            tuple[
                typing.Annotated[int, {"label": "q"}],
                typing.Annotated[float, {"label": "r"}],
            ],
        )

    def test_unannotated_outputs_fall_back_to_any(self):
        # No signature -> Any-typed return annotation, label still pinned.
        def plain(a, b):
            r = library.my_add(a, b)
            return r

        free = makers.reference_free(plain)
        rendered = pysource.recipe2python("rebuilt", free)
        self.assertIn('typing.Annotated[typing.Any, {"label": "r"}]', rendered.source)

    def test_multi_output_without_matching_tuple_falls_back_to_any(self):
        # A non-tuple return annotation for a 2-output workflow cannot be split,
        # so both outputs fall back to typing.Any.
        def typed_bad(a, b) -> int:
            q, r = library.divmod_func(a, b)
            return q, r

        free = makers.reference_free(typed_bad)
        rendered = pysource.recipe2python("rebuilt", free, inspect.signature(typed_bad))
        self.assertEqual(rendered.source.count("typing.Any"), 2)


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

    def test_annotated_return_type_is_unwrapped(self):
        def f(a, b) -> typing.Annotated[float, "meta"]:
            r = library.my_add(a, b)
            return r

        free = makers.reference_free(f)
        fn = pysource.recipe2python("rebuilt", free, inspect.signature(f)).build()
        self.assertEqual(
            typing.get_type_hints(fn, include_extras=True)["return"],
            typing.Annotated[float, {"label": "r"}],
        )

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
