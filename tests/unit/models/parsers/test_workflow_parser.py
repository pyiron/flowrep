import sys
import types
import unittest
from typing import Annotated, Any

import pydantic
from pyiron_snippets import versions

from flowrep.models import base_models, edge_models
from flowrep.models.nodes import atomic_model, workflow_model
from flowrep.models.parsers import (
    atomic_parser,
    object_scope,
    parser_protocol,
    symbol_scope,
    workflow_parser,
)

from flowrep_static import library


def add(x: float = 2.0, y: float = 1) -> float:
    return x + y


def multiply(x: float, y: float = 5) -> float:
    return x * y


def operation(x: float, y: float) -> tuple[float, float]:
    return x + y, x - y


@atomic_parser.atomic("sum_result", "diff_result")
def labeled_operation(x, y):
    return x + y, x - y


@atomic_parser.atomic("sum_result", "diff_result")
def annotated_operation(
    x: Annotated[float, {"label": "positive"}],
    y: Annotated[float, {"label": "variable"}],
) -> tuple[
    Annotated[float, {"label": "sum_result"}],
    Annotated[float, {"label": "diff_result"}],
]:
    return x + y, x - y


@atomic_parser.atomic("aggregate_result", unpack_mode=atomic_model.UnpackMode.NONE)
def tuple_operation(x: float, y: float) -> tuple[float, float]:
    return x + y, x - y


class Outer:
    class Inner:
        @staticmethod
        def nested_func(a, b):
            return a, b


@workflow_parser.workflow
def inner_macro(a, b=10):
    c, d = operation(a, b)
    e = add(c, y=d)
    f = multiply(e)
    return f


@workflow_parser.workflow
def outer_workflow(a, b):
    y = inner_macro(a, b)
    z: float = add(y, b)  # ast.AnnAssignment
    return z


# Name that is guaranteed not to collide with a real package or stdlib module.
_UNVERSIONED_MODULE_NAME = "_flowrep_test_unversioned_mod"


def _install_unversioned_module() -> types.ModuleType:
    """Register a bare module in ``sys.modules`` that has no ``__version__``."""
    mod = types.ModuleType(_UNVERSIONED_MODULE_NAME)
    sys.modules[_UNVERSIONED_MODULE_NAME] = mod
    return mod


def _uninstall_unversioned_module() -> None:
    sys.modules.pop(_UNVERSIONED_MODULE_NAME, None)


class TestWorkflowDecorator(unittest.TestCase):
    def test_workflow_without_args(self):
        @workflow_parser.workflow
        def simple_wf(x):
            y = add(x)
            return y

        self.assertTrue(hasattr(simple_wf, "flowrep_recipe"))
        self.assertIsInstance(simple_wf.flowrep_recipe, workflow_model.WorkflowNode)
        self.assertEqual(
            simple_wf(2), add(2), msg="Should still just be regular functions"
        )

    def test_workflow_with_output_labels(self):
        @workflow_parser.workflow("result")
        def simple_wf(x):
            y = add(x)
            return y

        self.assertEqual(simple_wf.flowrep_recipe.outputs, ["result"])

    def test_workflow_with_multiple_output_labels(self):
        @workflow_parser.workflow("out_a", "out_b")
        def multi_out(x):
            a, b = operation(x, x)
            return a, b

        self.assertEqual(multi_out.flowrep_recipe.outputs, ["out_a", "out_b"])

    def test_workflow_empty_parens_infers_labels(self):
        @workflow_parser.workflow()
        def wf(x):
            result = add(x)
            return result

        self.assertEqual(wf.flowrep_recipe.outputs, ["result"])

    def test_workflow_preserves_function_behavior(self):
        @workflow_parser.workflow
        def wf(a, b):
            c = add(a, b)
            return c

        self.assertEqual(wf(3, 4), add(3, 4))


class TestWorkflowDecoratorTypeValidation(unittest.TestCase):
    def test_rejects_class_bare_decorator(self):
        with self.assertRaises(TypeError) as ctx:

            @workflow_parser.workflow
            class MyClass:
                pass

        self.assertIn("@workflow can only decorate functions", str(ctx.exception))

    def test_rejects_class_with_args(self):
        with self.assertRaises(TypeError) as ctx:

            @workflow_parser.workflow("output")
            class MyClass:
                pass

        self.assertIn("@workflow can only decorate functions", str(ctx.exception))


class TestParseWorkflowBasic(unittest.TestCase):
    def test_protocol_fulfillment(self):
        self.assertIsInstance(
            workflow_parser.WorkflowParser(
                object_scope.ScopeProxy({}),
                symbol_scope.SymbolScope({}),
                versions.VersionInfoFactory(),
            ),
            parser_protocol.BodyWalker,
        )

    def test_single_node_workflow(self):
        def wf(x):
            y = add(x)
            return y

        node = workflow_parser.parse_workflow(wf)
        self.assertEqual(node.inputs, ["x"])
        self.assertEqual(node.outputs, ["y"])
        self.assertIn("add_0", node.nodes)

    def test_chained_nodes(self):
        def wf(x):
            y = add(x)
            z = multiply(y)
            return z

        node = workflow_parser.parse_workflow(wf)
        self.assertEqual(node.inputs, ["x"])
        self.assertEqual(node.outputs, ["z"])
        self.assertIn("add_0", node.nodes)
        self.assertIn("multiply_0", node.nodes)

    def test_multiple_outputs(self):
        def wf(x, y):
            a, b = operation(x, y)
            return a, b

        node = workflow_parser.parse_workflow(wf)
        self.assertEqual(node.inputs, ["x", "y"])
        self.assertEqual(node.outputs, ["a", "b"])

    def test_keyword_arguments(self):
        def wf(a, b):
            c = add(a, y=b)
            return c

        node = workflow_parser.parse_workflow(wf)
        # Check that input edges are correctly formed
        target = edge_models.TargetHandle(node="add_0", port="y")
        self.assertIn(target, node.input_edges)
        self.assertEqual(node.input_edges[target].port, "b")

    def test_mixed_positional_and_keyword(self):
        def wf(a, b):
            c = add(a, y=b)
            return c

        node = workflow_parser.parse_workflow(wf)
        # x should come from a (positional)
        target_x = edge_models.TargetHandle(node="add_0", port="x")
        self.assertIn(target_x, node.input_edges)
        self.assertEqual(node.input_edges[target_x].port, "a")
        # y should come from b (keyword)
        target_y = edge_models.TargetHandle(node="add_0", port="y")
        self.assertIn(target_y, node.input_edges)
        self.assertEqual(node.input_edges[target_y].port, "b")


class TestParseWorkflowEdges(unittest.TestCase):
    def test_input_edges_from_workflow_inputs(self):
        def wf(x):
            y = add(x)
            return y

        node = workflow_parser.parse_workflow(wf)
        target = edge_models.TargetHandle(node="add_0", port="x")
        self.assertIn(target, node.input_edges)
        self.assertEqual(node.input_edges[target].port, "x")

    def test_internal_edges_between_nodes(self):
        def wf(x):
            y = add(x)
            z = multiply(y)
            return z

        node = workflow_parser.parse_workflow(wf)
        target = edge_models.TargetHandle(node="multiply_0", port="x")
        self.assertIn(target, node.edges)
        source = node.edges[target]
        self.assertEqual(source.node, "add_0")
        self.assertEqual(source.port, "output_0")

    def test_output_edges(self):
        def wf(x):
            y = add(x)
            return y

        node = workflow_parser.parse_workflow(wf)
        target = edge_models.OutputTarget(port="y")
        self.assertIn(target, node.output_edges)
        source = node.output_edges[target]
        self.assertEqual(source.node, "add_0")
        self.assertEqual(source.port, "output_0")

    def test_reused_symbols(self):
        """Symbol reuse should create edges using the most recent definition"""

        def wf(x):
            y = add(x)
            y = multiply(y)
            return y

        node = workflow_parser.parse_workflow(wf)
        self.assertEqual(
            node.edges,
            {
                edge_models.TargetHandle(
                    node="multiply_0", port="x"
                ): edge_models.SourceHandle(node="add_0", port="output_0")
            },
        )
        self.assertEqual(
            node.output_edges,
            {
                edge_models.OutputTarget(port="y"): edge_models.SourceHandle(
                    node="multiply_0", port="output_0"
                )
            },
        )


class TestParseWorkflowNested(unittest.TestCase):
    def test_nested_workflow_detected(self):
        def outer_wf(a):
            b = inner_macro(a)
            return b

        node = workflow_parser.parse_workflow(outer_wf)
        self.assertIn("inner_macro_0", node.nodes)
        self.assertIsInstance(node.nodes["inner_macro_0"], workflow_model.WorkflowNode)

    def test_nested_workflow_edges(self):
        def outer_wf(a):
            b = inner_macro(a)
            c = multiply(b)
            return c

        node = workflow_parser.parse_workflow(outer_wf)
        # inner_wf output -> multiply input
        target = edge_models.TargetHandle(node="multiply_0", port="x")
        self.assertIn(target, node.edges)
        self.assertEqual(node.edges[target].node, "inner_macro_0")
        self.assertEqual(node.edges[target].port, "f")


class TestNestedWorkflowFullySourcing(unittest.TestCase):
    """
    Defaults on nested macro inputs save (or fail) fully-sourced validation.

    ``inner_macro(a, b=10)`` carries ``inputs_with_defaults=["b"]``, so callers
    that omit ``b`` are rescued by the default.  Omitting ``a`` (which has no
    default) must fail.
    """

    def test_nested_macro_default_saves_unsourced_input(self):
        """inner_macro(a) passes because b=10 provides a default."""

        def wf(a):
            f = inner_macro(a)
            return f

        node = workflow_parser.parse_workflow(wf)
        inner = node.nodes["inner_macro_0"]
        # b was never wired — only a has an input edge
        wired_ports = {t.port for t in node.input_edges if t.node == "inner_macro_0"}
        self.assertIn("a", wired_ports)
        self.assertNotIn("b", wired_ports)
        # …but b is in the defaults, which is why validation passed
        self.assertIn("b", inner.inputs_with_defaults)

    def test_nested_macro_unsourced_no_default_raises(self):
        """inner_macro(b=x) fails because a has no edge and no default."""

        def wf(x):
            f = inner_macro(b=x)
            return f

        with self.assertRaises(pydantic.ValidationError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("inner_macro_0.a", str(ctx.exception))

    def test_nested_macro_all_args_supplied_passes(self):
        """inner_macro(a, b) passes when all inputs are wired."""

        def wf(a, b):
            f = inner_macro(a, b)
            return f

        node = workflow_parser.parse_workflow(wf)
        wired_ports = {t.port for t in node.input_edges if t.node == "inner_macro_0"}
        self.assertEqual(wired_ports, {"a", "b"})

    def test_doubly_nested_unsourced_raises(self):
        """outer_workflow(a) fails — outer_workflow needs both a and b."""

        def wf(a):
            z = outer_workflow(a)
            return z

        with self.assertRaises(pydantic.ValidationError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("outer_workflow_0.b", str(ctx.exception))

    def test_doubly_nested_all_args_passes(self):
        """outer_workflow(a, b) passes when both inputs are provided."""

        def wf(a, b):
            z = outer_workflow(a, b)
            return z

        node = workflow_parser.parse_workflow(wf)
        wired_ports = {t.port for t in node.input_edges if t.node == "outer_workflow_0"}
        self.assertEqual(wired_ports, {"a", "b"})


class TestParseWorkflowOutputLabels(unittest.TestCase):
    def test_explicit_labels_override_inferred(self):
        def wf(x):
            result = add(x)
            return result

        node = workflow_parser.parse_workflow(wf, "custom_output")
        self.assertEqual(node.outputs, ["custom_output"])

    def test_explicit_labels_mismatch_raises(self):
        def wf(x):
            a, b = operation(x, x)
            return a, b

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(wf, "only_one")
        self.assertIn("matching number", str(ctx.exception).lower())

    def test_workflow_labels_from_annotation(self):
        def wf(x) -> Annotated[Any, {"label": "result"}]:
            y = add(x)
            return y

        node = workflow_parser.parse_workflow(wf)
        self.assertEqual(node.outputs, ["result"])

    def test_workflow_annotations_mismatch_raises(self):
        def wf(x) -> tuple[Annotated[Any, {"label": "result"}], int]:
            y = add(x)
            return y

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(wf, "only_one")
        self.assertIn("number of elements differ", str(ctx.exception))
        self.assertIn("['result', None]", str(ctx.exception))
        self.assertIn("['y']", str(ctx.exception))


class TestParseWorkflowErrors(unittest.TestCase):
    def test_no_return_raises(self):
        def wf(x):
            y = add(x)  # noqa: F841

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("return", str(ctx.exception).lower())

    def test_multiple_returns_raises(self):
        def wf(x, flag):
            y = add(x)
            z = multiply(x)
            return y
            return z

        with self.assertRaises(ValueError):
            workflow_parser.parse_workflow(wf)

    def test_unknown_symbol_raises(self):
        def wf(x):
            y = add(unknown_var)  # noqa: F821
            return y

        with self.assertRaises(KeyError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("unknown_var", str(ctx.exception).lower())

    def test_returning_input_directly_raises(self):
        def wf(x):
            return x

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("workflow inputs", str(ctx.exception).lower())

    def test_non_call_rhs_raises(self):
        def wf(x):
            y = x + 1
            return y

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("call", str(ctx.exception).lower())

    def test_duplicate_return_symbols_raises(self):
        def wf(x):
            y = add(x)
            return y, y

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("unique", str(ctx.exception).lower())

    def test_unrecognized_node_raises(self):
        def wf(x):
            print("This is not allowed")
            y = add(x)
            return y, y

        with self.assertRaises(TypeError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("but ast found", str(ctx.exception))

    def test_too_many_symbols_raises(self):
        def wf(x):
            y, z = add(x)
            return y, z

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("Cannot map", str(ctx.exception))
        self.assertIn("output_0", str(ctx.exception))
        self.assertIn("y", str(ctx.exception))
        self.assertIn("z", str(ctx.exception))

    def test_too_few_symbols_raises(self):
        def wf(x, y):
            z = operation(x, y)
            return z

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("Cannot map", str(ctx.exception))
        self.assertIn("output_0", str(ctx.exception))
        self.assertIn("output_1", str(ctx.exception))
        self.assertIn("z", str(ctx.exception))

    def test_unrecognized_return_raises(self):
        def wf(x):
            y = add(x)  # noqa: F841
            return z  # noqa: F821

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("Return symbol 'z' is not defined", str(ctx.exception))

    def test_too_many_symbols_for_list_raises(self):
        """
        A bit silly since the python interpreter would error it too, but in case we
        parse the code before the interpreter does, it's here for completeness.
        """

        def wf(x):
            y = add(x)
            accumulator, too_much = []  # noqa: F841
            return y

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("must target exactly one symbol", str(ctx.exception))


class TestNestedAttributeResolution(unittest.TestCase):
    """Test that nested class attributes like Foo.Bar.func are resolved."""

    def test_nested_class_method(self):
        @workflow_parser.workflow
        def wf(a, b):
            x, y = Outer.Inner.nested_func(a, b)
            return x, y

        node = wf.flowrep_recipe
        self.assertIn("nested_func_0", node.nodes)
        self.assertEqual(node.inputs, ["a", "b"])
        self.assertEqual(node.outputs, ["x", "y"])


class TestAnnotatedAssignment(unittest.TestCase):
    """Test that annotated assignments (e.g., `x: int = func()`) are handled."""

    def test_annotated_assignment(self):
        @workflow_parser.workflow
        def wf(x):
            y: int = add(x)
            return y

        node = wf.flowrep_recipe
        self.assertEqual(node.outputs, ["y"])
        self.assertIn("add_0", node.nodes)


class TestWorkflowNodeNaming(unittest.TestCase):
    """Test that node naming handles collisions correctly."""

    def test_multiple_calls_same_function(self):
        @workflow_parser.workflow
        def wf(x):
            a = add(x)
            b = add(a)
            c = add(b)
            return c

        node = wf.flowrep_recipe
        self.assertIn("add_0", node.nodes)
        self.assertIn("add_1", node.nodes)
        self.assertIn("add_2", node.nodes)


class TestWorkflowWithAtomicRecipes(unittest.TestCase):
    """Test interaction with @atomic decorated functions."""

    def test_uses_atomic_recipe_labels(self):
        @workflow_parser.workflow
        def wf(a, b):
            s, d = labeled_operation(a, b)
            return s, d

        recipe = wf.flowrep_recipe
        self.assertIsInstance(
            recipe.nodes["labeled_operation_0"], atomic_model.AtomicNode
        )
        self.assertDictEqual(
            {
                edge_models.OutputTarget(port="s"): edge_models.SourceHandle(
                    node="labeled_operation_0", port="sum_result"
                ),
                edge_models.OutputTarget(port="d"): edge_models.SourceHandle(
                    node="labeled_operation_0", port="diff_result"
                ),
            },
            recipe.output_edges,
        )

    def test_uses_atomic_recipe_annotations(self):
        @workflow_parser.workflow
        def wf(a, b):
            s, d = annotated_operation(a, b)
            return s, d

        recipe = wf.flowrep_recipe
        self.assertIsInstance(
            recipe.nodes["annotated_operation_0"], atomic_model.AtomicNode
        )
        self.assertDictEqual(
            {
                edge_models.OutputTarget(port="s"): edge_models.SourceHandle(
                    node="annotated_operation_0", port="sum_result"
                ),
                edge_models.OutputTarget(port="d"): edge_models.SourceHandle(
                    node="annotated_operation_0", port="diff_result"
                ),
            },
            recipe.output_edges,
        )

    def test_uses_atomic_recipe_unpacking(self):
        @workflow_parser.workflow
        def wf(a, b):
            t = tuple_operation(a, b)
            return t

        recipe = wf.flowrep_recipe
        self.assertIsInstance(
            recipe.nodes["tuple_operation_0"], atomic_model.AtomicNode
        )
        self.assertDictEqual(
            {
                edge_models.OutputTarget(port="t"): edge_models.SourceHandle(
                    node="tuple_operation_0", port="aggregate_result"
                ),
            },
            recipe.output_edges,
        )


class TestWorkflowFullyQualifiedName(unittest.TestCase):
    """Tests that parse_workflow populates fully_qualified_name correctly."""

    def test_fqn_set_on_parsed_workflow(self):
        def wf(x):
            y = add(x)
            return y

        node = workflow_parser.parse_workflow(wf)
        self.assertEqual(
            node.fully_qualified_name,
            f"{wf.__module__}.{wf.__qualname__}",
        )

    def test_fqn_on_decorated_workflow(self):
        @workflow_parser.workflow
        def decorated_wf(x):
            y = add(x)
            return y

        self.assertEqual(
            decorated_wf.flowrep_recipe.fully_qualified_name,
            f"{decorated_wf.__module__}.{decorated_wf.__qualname__}",
        )

    def test_fqn_on_decorated_workflow_with_args(self):
        @workflow_parser.workflow("result")
        def decorated_wf(x):
            y = add(x)
            return y

        self.assertEqual(
            decorated_wf.flowrep_recipe.fully_qualified_name,
            f"{decorated_wf.__module__}.{decorated_wf.__qualname__}",
        )

    def test_fqn_nested_workflow_has_own_fqn(self):
        """inner_macro's recipe should carry its own fqn, not the outer's."""

        def outer_wf(a):
            b = inner_macro(a)
            return b

        node = workflow_parser.parse_workflow(outer_wf)
        inner = node.nodes["inner_macro_0"]
        self.assertEqual(
            inner.fully_qualified_name,
            f"{inner_macro.__module__}.{inner_macro.__qualname__}",
        )

    def test_fqn_roundtrips_through_serialization(self):
        def wf(x):
            y = add(x)
            return y

        node = workflow_parser.parse_workflow(wf)
        for mode in ["python", "json"]:
            with self.subTest(mode=mode):
                data = node.model_dump(mode=mode)
                restored = workflow_model.WorkflowNode.model_validate(data)
                self.assertEqual(
                    node.fully_qualified_name, restored.fully_qualified_name
                )


class TestWorkflowVariadicRejection(unittest.TestCase):
    """Integration tests: variadic params are rejected during parse_workflow."""

    def test_varargs_rejected(self):
        def wf(*args):
            y = add(args[0])
            return y

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("Variadic parameter kinds are not supported", str(ctx.exception))
        self.assertIn("args", str(ctx.exception))

    def test_kwargs_rejected(self):
        def wf(**kwargs):
            y = add(kwargs["x"])
            return y

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("Variadic parameter kinds are not supported", str(ctx.exception))
        self.assertIn("kwargs", str(ctx.exception))

    def test_keyword_only_accepted(self):
        def wf(a, *, key=1):
            y = add(a, key)
            return y

        node = workflow_parser.parse_workflow(wf)
        self.assertIn("key", node.inputs)
        self.assertEqual(
            node.reference.restricted_input_kinds,
            {"key": base_models.RestrictedParamKind.KEYWORD_ONLY},
        )

    def test_positional_or_keyword_omitted_from_restricted(self):
        def wf(a, b):
            c = add(a, b)
            return c

        node = workflow_parser.parse_workflow(wf)
        self.assertEqual(node.reference.restricted_input_kinds, {})


class TestParseWorkflowVersionParams(unittest.TestCase):
    """Tests that version-related params are forwarded through parse_workflow."""

    def test_version_is_populated(self):
        def my_wf(x):
            y = library.identity(x)
            return y

        node = workflow_parser.parse_workflow(my_wf)
        # The workflow function is defined in this test module; version depends
        # on whether this package exposes __version__
        self.assertIsInstance(node.reference.info.version, (str, type(None)))

    def test_fqn_is_populated(self):
        def my_wf(x):
            y = library.identity(x)
            return y

        node = workflow_parser.parse_workflow(my_wf)
        self.assertIsNotNone(node.fully_qualified_name)
        self.assertIn("my_wf", node.fully_qualified_name)

    def test_forbid_main_raises(self):
        def my_wf(x):
            y = library.identity(x)
            return y

        my_wf.__module__ = "__main__"
        with self.assertRaises(ValueError, msg="__main__") as ctx:
            workflow_parser.parse_workflow(my_wf, forbid_main=True)
        self.assertIn("__main__", str(ctx.exception))

    def test_forbid_locals_raises(self):
        def my_wf(x):
            y = library.identity(x)
            return y

        my_wf.__qualname__ = "outer.<locals>.my_wf"
        with self.assertRaises(ValueError, msg="<locals>") as ctx:
            workflow_parser.parse_workflow(my_wf, forbid_locals=True)
        self.assertIn("<locals>.my_wf", str(ctx.exception))

    def test_require_version_raises_when_missing(self):
        _install_unversioned_module()
        try:

            def my_wf(x):
                y = library.identity(x)
                return y

            my_wf.__module__ = _UNVERSIONED_MODULE_NAME
            with self.assertRaises(ValueError) as ctx:
                workflow_parser.parse_workflow(my_wf, require_version=True)
            self.assertIn("Could not find a version", str(ctx.exception))
        finally:
            _uninstall_unversioned_module()

    def test_version_scraping_is_forwarded(self):
        def my_wf(x):
            y = library.identity(x)
            return y

        custom_version = "99.0.0"
        # The test module's top-level package name
        pkg = my_wf.__module__.split(".")[0]
        scraping = {pkg: lambda _name: custom_version}
        node = workflow_parser.parse_workflow(my_wf, version_scraping=scraping)
        self.assertEqual(node.reference.info.version, custom_version)

    def test_child_nodes_not_affected_by_workflow_version_params(self):
        """Sub-workflow bodies built during parsing should not carry the
        top-level fqn/version — those are for the enclosing workflow only."""

        def my_wf(a, b):
            x = library.my_add(a, b)
            y = library.identity(x)
            return y

        node = workflow_parser.parse_workflow(my_wf)
        # The child atomic nodes should have their own fqn, not the workflow's
        for child in node.nodes.values():
            self.assertNotEqual(child.fully_qualified_name, node.fully_qualified_name)


class TestWorkflowDecoratorVersionParams(unittest.TestCase):
    """Tests that the @workflow decorator forwards version params."""

    def test_decorator_passes_version_scraping(self):
        custom_version = "42.0.0"
        module_base = add.__module__.split(".")[0]

        @workflow_parser.workflow(
            version_scraping={
                module_base: lambda _: custom_version,
            }
        )
        def my_wf(x):
            y = library.identity(x)
            return y

        self.assertEqual(my_wf.flowrep_recipe.reference.info.version, custom_version)

    def test_decorator_forbid_locals_on_inner_function(self):
        with self.assertRaises(ValueError):

            @workflow_parser.workflow(forbid_locals=True)
            def inner(x):
                y = library.identity(x)
                return y


class TestParseWorkflowHasDefault(unittest.TestCase):
    """Tests that parse_workflow populates inputs_with_defaults on the workflow and children."""

    def test_workflow_inputs_with_defaults_from_signature(self):
        def wf(a, b=5):
            c = add(a, b)
            return c

        node = workflow_parser.parse_workflow(wf)
        self.assertEqual(node.reference.inputs_with_defaults, ["b"])

    def test_workflow_no_defaults(self):
        def wf(a, b):
            c = add(a, b)
            return c

        node = workflow_parser.parse_workflow(wf)
        self.assertEqual(node.reference.inputs_with_defaults, [])

    def test_workflow_all_defaults(self):
        def wf(a=1, b=2):
            c = add(a, b)
            return c

        node = workflow_parser.parse_workflow(wf)
        self.assertEqual(node.reference.inputs_with_defaults, ["a", "b"])

    def test_child_node_inputs_with_defaults_populated(self):
        """Undecorated children parsed on-the-fly should carry inputs_with_defaults."""

        def wf(x):
            y = add(x)  # add has x=2.0, y=1 → inputs_with_defaults=["x", "y"]
            return y

        node = workflow_parser.parse_workflow(wf)
        child = node.nodes["add_0"]
        self.assertEqual(child.reference.inputs_with_defaults, ["x", "y"])

    def test_child_mixed_defaults(self):
        def wf(x):
            y = multiply(x)  # multiply has (x, y=5) → inputs_with_defaults=["y"]
            return y

        node = workflow_parser.parse_workflow(wf)
        child = node.nodes["multiply_0"]
        self.assertEqual(child.reference.inputs_with_defaults, ["y"])

    def test_decorator_preserves_inputs_with_defaults(self):
        @workflow_parser.workflow
        def wf(a, b=10):
            c = add(a, b)
            return c

        self.assertEqual(wf.flowrep_recipe.reference.inputs_with_defaults, ["b"])

    def test_roundtrip_preserves_inputs_with_defaults(self):
        def wf(a, b=5):
            c = add(a, b)
            return c

        node = workflow_parser.parse_workflow(wf)
        for mode in ["json", "python"]:
            with self.subTest(mode=mode):
                data = node.model_dump(mode=mode)
                restored = workflow_model.WorkflowNode.model_validate(data)
                self.assertEqual(restored.reference.inputs_with_defaults, ["b"])


class TestWorkflowVersionScrapingPropagation(unittest.TestCase):
    """Verify version_scraping propagates to child nodes parsed inside a workflow."""

    def _pkg(self) -> str:
        return add.__module__.split(".")[0]

    def test_scraping_propagates_to_undecorated_child(self):
        """An undecorated function parsed on-the-fly should receive the scraping map."""
        custom = "11.22.33"

        def my_wf(x):
            y = add(x)
            return y

        node = workflow_parser.parse_workflow(
            my_wf, version_scraping={self._pkg(): lambda _: custom}
        )
        child = node.nodes["add_0"]
        self.assertEqual(child.reference.info.version, custom)

    def test_scraping_does_not_override_prebuilt_recipe(self):
        """A function already decorated with @atomic keeps its own recipe."""
        custom = "99.99.99"

        def my_wf(x):
            y = library.identity(x)
            return y

        node = workflow_parser.parse_workflow(
            my_wf, version_scraping={self._pkg(): lambda _: custom}
        )
        child = node.nodes["identity_0"]
        # library.identity was decorated at import time; its recipe is fixed
        self.assertNotEqual(child.reference.info.version, custom)

    def test_scraping_propagates_through_chained_nodes(self):
        """Multiple undecorated children all receive the scraping map."""
        custom = "44.55.66"

        def my_wf(x):
            y = add(x)
            z = multiply(y)
            return z

        node = workflow_parser.parse_workflow(
            my_wf, version_scraping={self._pkg(): lambda _: custom}
        )
        self.assertEqual(node.nodes["add_0"].reference.info.version, custom)
        self.assertEqual(node.nodes["multiply_0"].reference.info.version, custom)

    def test_decorator_propagates_scraping_to_children(self):
        """@workflow decorator forwards version_scraping to child nodes."""
        custom = "77.88.99"
        pkg = add.__module__.split(".")[0]

        @workflow_parser.workflow(version_scraping={pkg: lambda _: custom})
        def my_wf(x):
            y = add(x)
            return y

        child = my_wf.flowrep_recipe.nodes["add_0"]
        self.assertEqual(child.reference.info.version, custom)


class TestLocalImports(unittest.TestCase):
    def test_local_imports(self):
        def local_import(x, y):
            import flowrep_static.local_library

            sum_ = flowrep_static.local_library.my_add(x, y)
            return sum_

        def import_as(x, y):
            import flowrep_static.local_library as ll

            sum_ = ll.my_add(x, y)
            return sum_

        def import_from(x, y):
            from flowrep_static import local_library

            sum_ = local_library.my_add(x, y)
            return sum_

        def import_from_as(x, y):
            from flowrep_static import local_library as ll

            sum_ = ll.my_add(x, y)
            return sum_

        def import_function(x, y):
            from flowrep_static.local_library import my_add

            sum_ = my_add(x, y)
            return sum_

        def import_function_as(x, y):
            from flowrep_static.local_library import my_add as ma

            sum_ = ma(x, y)
            return sum_

        for wf_func in [
            local_import,
            import_as,
            import_from,
            import_from_as,
            import_function,
            import_function_as,
        ]:
            with self.subTest(wf_func.__name__):
                node = workflow_parser.parse_workflow(wf_func)
                self.assertEqual(
                    node.nodes["my_add_0"].reference.info.module,
                    "flowrep_static.local_library",
                )
                self.assertEqual(
                    node.nodes["my_add_0"].reference.info.qualname,
                    "my_add",
                )

    def test_local_import_raises(self):
        def import_from_sibling(x, y):
            from .test_for_parser import pair

            a, b = pair(x, y)
            return a, b

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(import_from_sibling)
        self.assertIn("Relative imports are not supported", str(ctx.exception))
        self.assertIn("test_for_parser", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
