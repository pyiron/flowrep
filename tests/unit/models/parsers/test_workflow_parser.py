import ast
import unittest
from typing import Annotated, Any

from flowrep.models import edge_models
from flowrep.models.nodes import atomic_model, helper_models, workflow_model
from flowrep.models.parsers import (
    atomic_parser,
    parser_protocol,
    symbol_scope,
    workflow_parser,
)


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
    def test_protocol_fulfillment(self):
        self.assertIsInstance(
            workflow_parser.WorkflowParser(symbol_scope.SymbolScope({})),
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

    def test_reused_symbol_raises(self):
        def wf(x):
            y = add(x)
            y = multiply(y)
            return y

        with self.assertRaises(ValueError) as ctx:
            workflow_parser.parse_workflow(wf)
        self.assertIn("already in scope", str(ctx.exception).lower())

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


class TestParseWorkflowControlFlowNotImplemented(unittest.TestCase):
    """Control flow is not yet implemented; verify NotImplementedError is raised."""

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


class TestConsumeCallArguments(unittest.TestCase):
    """Tests for consume_call_arguments function."""

    def _make_labeled_node(
        self, label: str, inputs: list[str], outputs: list[str] | None = None
    ) -> helper_models.LabeledNode:
        outputs = outputs or ["output_0"]
        return helper_models.LabeledNode(
            label=label,
            node=atomic_model.AtomicNode(
                fully_qualified_name="test.module.func",
                inputs=inputs,
                outputs=outputs,
            ),
        )

    def _parse_call(self, call_str: str) -> ast.Call:
        return ast.parse(call_str, mode="eval").body

    def _make_scope(self, symbols: list[str]) -> symbol_scope.SymbolScope:
        return symbol_scope.SymbolScope(
            {s: edge_models.InputSource(port=s) for s in symbols}
        )

    def _consumed_pairs(self, scope: symbol_scope.SymbolScope) -> list[tuple[str, str]]:
        """Extract (symbol, consumer_port) pairs from scope consumptions."""
        return [(c.symbol, c.consumer_port) for c in scope._consumptions]

    def test_single_positional_arg(self):
        scope = self._make_scope(["x"])
        call = self._parse_call("func(x)")
        node = self._make_labeled_node("func_0", inputs=["a"])

        workflow_parser.consume_call_arguments(scope, call, node)

        self.assertEqual(self._consumed_pairs(scope), [("x", "a")])

    def test_multiple_positional_args(self):
        scope = self._make_scope(["x", "y", "z"])
        call = self._parse_call("func(x, y, z)")
        node = self._make_labeled_node("func_0", inputs=["a", "b", "c"])

        workflow_parser.consume_call_arguments(scope, call, node)

        self.assertEqual(
            self._consumed_pairs(scope), [("x", "a"), ("y", "b"), ("z", "c")]
        )

    def test_single_keyword_arg(self):
        scope = self._make_scope(["x"])
        call = self._parse_call("func(a=x)")
        node = self._make_labeled_node("func_0", inputs=["a"])

        workflow_parser.consume_call_arguments(scope, call, node)

        self.assertEqual(self._consumed_pairs(scope), [("x", "a")])

    def test_multiple_keyword_args(self):
        scope = self._make_scope(["x", "y"])
        call = self._parse_call("func(a=x, b=y)")
        node = self._make_labeled_node("func_0", inputs=["a", "b"])

        workflow_parser.consume_call_arguments(scope, call, node)

        self.assertEqual(self._consumed_pairs(scope), [("x", "a"), ("y", "b")])

    def test_mixed_positional_and_keyword(self):
        scope = self._make_scope(["x", "y"])
        call = self._parse_call("func(x, b=y)")
        node = self._make_labeled_node("func_0", inputs=["a", "b"])

        workflow_parser.consume_call_arguments(scope, call, node)

        self.assertEqual(self._consumed_pairs(scope), [("x", "a"), ("y", "b")])

    def test_keyword_args_out_of_order(self):
        scope = self._make_scope(["x", "y"])
        call = self._parse_call("func(b=y, a=x)")
        node = self._make_labeled_node("func_0", inputs=["a", "b"])

        workflow_parser.consume_call_arguments(scope, call, node)

        # Keywords consumed in call order, not definition order
        self.assertEqual(self._consumed_pairs(scope), [("y", "b"), ("x", "a")])

    def test_no_args(self):
        scope = self._make_scope([])
        call = self._parse_call("func()")
        node = self._make_labeled_node("func_0", inputs=[])

        workflow_parser.consume_call_arguments(scope, call, node)

        self.assertEqual(self._consumed_pairs(scope), [])

    def test_literal_positional_raises_type_error(self):
        scope = self._make_scope([])
        call = self._parse_call("func(42)")
        node = self._make_labeled_node("func_0", inputs=["a"])

        with self.assertRaises(TypeError) as ctx:
            workflow_parser.consume_call_arguments(scope, call, node)

        self.assertIn("symbolic input", str(ctx.exception))
        self.assertIn("func_0", str(ctx.exception))

    def test_literal_keyword_raises_type_error(self):
        scope = self._make_scope([])
        call = self._parse_call("func(a=42)")
        node = self._make_labeled_node("func_0", inputs=["a"])

        with self.assertRaises(TypeError) as ctx:
            workflow_parser.consume_call_arguments(scope, call, node)

        self.assertIn("symbolic input", str(ctx.exception))

    def test_expression_positional_raises_type_error(self):
        scope = self._make_scope(["x", "y"])
        call = self._parse_call("func(x + y)")
        node = self._make_labeled_node("func_0", inputs=["a"])

        with self.assertRaises(TypeError) as ctx:
            workflow_parser.consume_call_arguments(scope, call, node)

        self.assertIn("symbolic input", str(ctx.exception))

    def test_nested_call_raises_type_error(self):
        scope = self._make_scope(["x"])
        call = self._parse_call("func(other_func(x))")
        node = self._make_labeled_node("func_0", inputs=["a"])

        with self.assertRaises(TypeError) as ctx:
            workflow_parser.consume_call_arguments(scope, call, node)

        self.assertIn("symbolic input", str(ctx.exception))

    def test_attribute_access_raises_type_error(self):
        scope = self._make_scope([])
        call = self._parse_call("func(obj.attr)")
        node = self._make_labeled_node("func_0", inputs=["a"])

        with self.assertRaises(TypeError) as ctx:
            workflow_parser.consume_call_arguments(scope, call, node)

        self.assertIn("symbolic input", str(ctx.exception))

    def test_subscript_raises_type_error(self):
        scope = self._make_scope([])
        call = self._parse_call("func(arr[0])")
        node = self._make_labeled_node("func_0", inputs=["a"])

        with self.assertRaises(TypeError) as ctx:
            workflow_parser.consume_call_arguments(scope, call, node)

        self.assertIn("symbolic input", str(ctx.exception))

    def test_string_literal_keyword_raises_type_error(self):
        scope = self._make_scope([])
        call = self._parse_call("func(a='hello')")
        node = self._make_labeled_node("func_0", inputs=["a"])

        with self.assertRaises(TypeError) as ctx:
            workflow_parser.consume_call_arguments(scope, call, node)

        self.assertIn("symbolic input", str(ctx.exception))

    def test_partial_consumption_before_error(self):
        """Valid args are consumed before hitting an invalid one."""
        scope = self._make_scope(["x"])
        call = self._parse_call("func(x, 42)")
        node = self._make_labeled_node("func_0", inputs=["a", "b"])

        with self.assertRaises(TypeError):
            workflow_parser.consume_call_arguments(scope, call, node)

        # First arg was consumed before the error
        self.assertEqual(self._consumed_pairs(scope), [("x", "a")])

    def test_preserves_underscore_names(self):
        scope = self._make_scope(["_private", "__dunder__"])
        call = self._parse_call("func(_private, __dunder__)")
        node = self._make_labeled_node("func_0", inputs=["a", "b"])

        workflow_parser.consume_call_arguments(scope, call, node)

        self.assertEqual(
            self._consumed_pairs(scope), [("_private", "a"), ("__dunder__", "b")]
        )

    def test_consumed_source_is_recorded(self):
        """Verify the full SymbolConsumption, not just the symbol/port pair."""
        scope = self._make_scope(["x"])
        call = self._parse_call("func(x)")
        node = self._make_labeled_node("func_0", inputs=["a"])

        workflow_parser.consume_call_arguments(scope, call, node)

        c = scope._consumptions[0]
        self.assertEqual(c.symbol, "x")
        self.assertEqual(c.consumer_node, "func_0")
        self.assertEqual(c.consumer_port, "a")
        self.assertEqual(c.source, edge_models.InputSource(port="x"))

    def test_generates_input_edges(self):
        """Consuming InputSource symbols produces correct input_edges."""
        scope = self._make_scope(["x", "y"])
        call = self._parse_call("func(x, b=y)")
        node = self._make_labeled_node("func_0", inputs=["a", "b"])

        workflow_parser.consume_call_arguments(scope, call, node)

        self.assertEqual(
            scope.input_edges,
            {
                edge_models.TargetHandle(
                    node="func_0", port="a"
                ): edge_models.InputSource(port="x"),
                edge_models.TargetHandle(
                    node="func_0", port="b"
                ): edge_models.InputSource(port="y"),
            },
        )

    def test_generates_sibling_edges(self):
        """Consuming SourceHandle symbols produces correct sibling edges."""
        scope = symbol_scope.SymbolScope(
            {"x": edge_models.SourceHandle(node="upstream_0", port="out")}
        )
        call = self._parse_call("func(x)")
        node = self._make_labeled_node("func_0", inputs=["a"])

        workflow_parser.consume_call_arguments(scope, call, node)

        self.assertEqual(scope.input_edges, {})
        self.assertEqual(
            scope.edges,
            {
                edge_models.TargetHandle(
                    node="func_0", port="a"
                ): edge_models.SourceHandle(node="upstream_0", port="out"),
            },
        )


if __name__ == "__main__":
    unittest.main()
