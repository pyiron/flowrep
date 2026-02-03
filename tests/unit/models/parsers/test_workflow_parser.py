import ast
import unittest
from typing import Annotated, Any

from flowrep.models import edge_models
from flowrep.models.nodes import atomic_model, workflow_model
from flowrep.models.parsers import atomic_parser, workflow_parser


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
def inner_macro(a, b):
    c, d = operation(a, b)
    e = add(c, y=d)
    f = multiply(e)
    return f


@workflow_parser.workflow
def outer_workflow(a, b):
    y = inner_macro(a, b)
    z: float = add(y, b)  # ast.AnnAssignment
    return z


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
        self.assertIn("same length", str(ctx.exception))
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

    def test_reused_symbol_raises(self):
        def wf(x):
            y = add(x)
            y = multiply(y)
            return y

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("re-use", str(ctx.exception).lower())

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
        self.assertIn("Cannot map node outputs for 'add_0'", str(ctx.exception))

    def test_too_few_symbols_raises(self):
        def wf(x, y):
            z = operation(x, y)
            return z

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("Cannot map node outputs for 'operation_0'", str(ctx.exception))


class TestParseWorkflowControlFlowNotImplemented(unittest.TestCase):
    """Control flow is not yet implemented; verify NotImplementedError is raised."""

    def test_for_loop_raises(self):
        def wf(x):
            for i in range(x):
                y = add(i)
            return y

        with self.assertRaises(NotImplementedError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("for", str(ctx.exception).lower())

    def test_while_loop_raises(self):
        def wf(x):
            while x > 0:
                x = add(x)
            return x

        with self.assertRaises(NotImplementedError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("while", str(ctx.exception).lower())

    def test_if_statement_raises(self):
        def wf(x):
            if x > 0:
                y = add(x)
            return y

        with self.assertRaises(NotImplementedError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("if", str(ctx.exception).lower())

    def test_try_except_raises(self):
        def wf(x):
            try:
                y = add(x)
            except Exception:
                y = multiply(x)
            return y

        with self.assertRaises(NotImplementedError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("try", str(ctx.exception).lower())

    def test_empty_list_assignment_raises(self):
        def wf(x):
            y = []  # noqa: F841
            return x

        with self.assertRaises(NotImplementedError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("list", str(ctx.exception).lower())


class TestScopeProxy(unittest.TestCase):
    def test_basic_access(self):
        d = {"foo": 1, "bar": 2}
        proxy = workflow_parser.ScopeProxy(d)
        self.assertEqual(proxy.foo, 1)
        self.assertEqual(proxy.bar, 2)

    def test_missing_key_raises_attribute_error(self):
        proxy = workflow_parser.ScopeProxy({})
        with self.assertRaises(AttributeError):
            _ = proxy.nonexistent


class TestUniqueSuffix(unittest.TestCase):
    def test_first_suffix(self):
        result = workflow_parser.unique_suffix("foo", [])
        self.assertEqual(result, "foo_0")

    def test_increments_on_collision(self):
        result = workflow_parser.unique_suffix("foo", ["foo_0", "foo_1"])
        self.assertEqual(result, "foo_2")

    def test_handles_gaps(self):
        result = workflow_parser.unique_suffix("foo", ["foo_0", "foo_2"])
        self.assertEqual(result, "foo_1")


class TestResolveSymbolToObject(unittest.TestCase):
    def test_simple_name(self):
        scope = workflow_parser.ScopeProxy({"add": add})

        node = ast.Name(id="add")
        result = workflow_parser.resolve_symbol_to_object(node, scope)
        self.assertIs(result, add)

    def test_attribute_chain(self):
        scope = workflow_parser.ScopeProxy({"Outer": Outer})

        # Outer.Inner.nested_func
        node = ast.Attribute(
            value=ast.Attribute(value=ast.Name(id="Outer"), attr="Inner"),
            attr="nested_func",
        )
        result = workflow_parser.resolve_symbol_to_object(node, scope)
        self.assertIs(result, Outer.Inner.nested_func)

    def test_missing_attribute_raises(self):
        scope = workflow_parser.ScopeProxy({"Outer": Outer})

        node = ast.Attribute(value=ast.Name(id="Outer"), attr="NonExistent")
        with self.assertRaises(ValueError):
            workflow_parser.resolve_symbol_to_object(node, scope)


class TestResolveSymbolsToStrings(unittest.TestCase):
    def test_single_name(self):
        node = ast.Name(id="foo")
        result = workflow_parser.resolve_symbols_to_strings(node)
        self.assertEqual(result, ["foo"])

    def test_tuple_of_names(self):
        node = ast.Tuple(elts=[ast.Name(id="a"), ast.Name(id="b")])
        result = workflow_parser.resolve_symbols_to_strings(node)
        self.assertEqual(result, ["a", "b"])

    def test_non_name_raises(self):
        node = ast.Constant(value=42)
        with self.assertRaises(TypeError):
            workflow_parser.resolve_symbols_to_strings(node)


class TestWorkflowParserStateEnforceUniqueSymbols(unittest.TestCase):
    def test_allows_new_symbols(self):
        state = workflow_parser._WorkflowParserState(inputs=["x"])
        # Should not raise
        state.enforce_unique_symbols(["y", "z"])

    def test_rejects_duplicate_of_input(self):
        state = workflow_parser._WorkflowParserState(inputs=["x"])
        with self.assertRaises(ValueError):
            state.enforce_unique_symbols(["x"])


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


if __name__ == "__main__":
    unittest.main()
