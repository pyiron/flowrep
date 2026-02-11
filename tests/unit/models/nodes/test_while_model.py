"""Unit tests for WhileNode and related classes."""

import unittest

import pydantic

from flowrep.models import base_models, edge_models, subgraph_validation
from flowrep.models.nodes import (
    atomic_model,
    helper_models,
    while_model,
    workflow_model,
)


def make_atomic(inputs: list[str], outputs: list[str]) -> atomic_model.AtomicNode:
    return atomic_model.AtomicNode(
        fully_qualified_name="mod.func",
        inputs=inputs,
        outputs=outputs,
    )


def make_valid_while_node(
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    condition_inputs: list[str] | None = None,
    condition_outputs: list[str] | None = None,
    body_inputs: list[str] | None = None,
    body_outputs: list[str] | None = None,
    condition_label: str = "cond",
    body_label: str = "body",
    input_edges: dict | None = None,
    output_edges: dict | None = None,
) -> while_model.WhileNode:
    inputs = inputs if inputs is not None else ["x"]
    outputs = outputs if outputs is not None else []
    condition_inputs = condition_inputs if condition_inputs is not None else ["val"]
    condition_outputs = (
        condition_outputs if condition_outputs is not None else ["result"]
    )
    body_inputs = body_inputs if body_inputs is not None else ["inp"]
    body_outputs = body_outputs if body_outputs is not None else ["out"]

    return while_model.WhileNode(
        inputs=inputs,
        outputs=outputs,
        case=helper_models.ConditionalCase(
            condition=helper_models.LabeledNode(
                label=condition_label,
                node=make_atomic(condition_inputs, condition_outputs),
            ),
            body=helper_models.LabeledNode(
                label=body_label,
                node=make_atomic(body_inputs, body_outputs),
            ),
        ),
        input_edges=input_edges if input_edges is not None else {},
        output_edges=output_edges if output_edges is not None else {},
    )


class TestWhileNodeBasic(unittest.TestCase):
    def test_schema_generation(self):
        """model_json_schema() fails if forward refs aren't resolved."""
        while_model.WhileNode.model_json_schema()

    def test_obeys_build_subgraph_with_static_output(self):
        """WhileNode should obey build subgraph with static output."""
        node = make_valid_while_node()
        self.assertIsInstance(node, subgraph_validation.DynamicSubgraphStaticOutput)

    def test_valid_minimal(self):
        """Minimal valid WhileNode with empty edges."""
        wn = make_valid_while_node()
        self.assertEqual(wn.type, base_models.RecipeElementType.WHILE)
        self.assertEqual(wn.inputs, ["x"])
        self.assertEqual(wn.outputs, [])
        self.assertEqual(wn.input_edges, {})
        self.assertEqual(wn.output_edges, {})

    def test_valid_fully_wired(self):
        """WhileNode with all edge types populated."""
        wn = while_model.WhileNode(
            inputs=["n", "acc"],
            outputs=["acc"],
            case=helper_models.ConditionalCase(
                condition=helper_models.LabeledNode(
                    label="check",
                    node=make_atomic(["val"], ["is_positive"]),
                ),
                body=helper_models.LabeledNode(
                    label="decrement",
                    node=make_atomic(["current", "total"], ["next_val", "next_total"]),
                ),
            ),
            input_edges={
                "check.val": "n",
                "decrement.current": "n",
                "decrement.total": "acc",
            },
            output_edges={"acc": "decrement.next_total"},
        )
        self.assertEqual(len(wn.input_edges), 3)
        self.assertEqual(len(wn.output_edges), 1)


class TestWhileNodeIOValidation(unittest.TestCase):
    def test_duplicate_inputs_rejected(self):
        """Duplicate inputs are rejected."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            make_valid_while_node(inputs=["x", "y", "x"])
        self.assertIn("unique", str(ctx.exception).lower())
        self.assertIn("x", str(ctx.exception))

    def test_duplicate_outputs_rejected(self):
        """Duplicate outputs are rejected."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            make_valid_while_node(inputs=["a", "b"], outputs=["a", "b", "a"])
        self.assertIn("unique", str(ctx.exception).lower())
        self.assertIn("a", str(ctx.exception))

    def test_invalid_input_labels(self):
        """Invalid input labels are rejected."""
        for invalid in ["for", "inputs", "1bad"]:
            with (
                self.subTest(label=invalid),
                self.assertRaises(pydantic.ValidationError),
            ):
                make_valid_while_node(inputs=[invalid])

    def test_invalid_output_labels(self):
        """Invalid output labels are rejected."""
        for invalid in ["while", "outputs", "my-var"]:
            with (
                self.subTest(label=invalid),
                self.assertRaises(pydantic.ValidationError),
            ):
                make_valid_while_node(inputs=[invalid], outputs=[invalid])

    def test_outputs_must_be_subset_of_inputs(self):
        """Output labels must all appear among input labels."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            make_valid_while_node(inputs=["x"], outputs=["y"])
        self.assertIn("subset", str(ctx.exception).lower())

    def test_outputs_strict_subset_ok(self):
        """Strict subset of inputs is fine."""
        wn = make_valid_while_node(
            inputs=["a", "b", "c"],
            outputs=["a"],
            body_inputs=["inp"],
            body_outputs=["out"],
            input_edges={"body.inp": "a"},
            output_edges={"a": "body.out"},
        )
        self.assertEqual(wn.outputs, ["a"])

    def test_outputs_equal_to_inputs_ok(self):
        """Outputs identical to inputs is fine."""
        wn = make_valid_while_node(
            inputs=["a"],
            outputs=["a"],
            body_inputs=["inp"],
            body_outputs=["out"],
            input_edges={"body.inp": "a"},
            output_edges={"a": "body.out"},
        )
        self.assertEqual(wn.outputs, ["a"])


class TestWhileNodeInputEdges(unittest.TestCase):
    def test_valid_input_edge_to_condition(self):
        """Input edge to condition node."""
        wn = make_valid_while_node(
            inputs=["x"],
            condition_inputs=["val"],
            input_edges={"cond.val": "x"},
        )
        self.assertEqual(len(wn.input_edges), 1)

    def test_valid_input_edge_to_body(self):
        """Input edge to body node."""
        wn = make_valid_while_node(
            inputs=["x"],
            body_inputs=["inp"],
            input_edges={"body.inp": "x"},
        )
        self.assertEqual(len(wn.input_edges), 1)

    def test_valid_input_edges_to_both(self):
        """Input edges to both condition and body."""
        wn = make_valid_while_node(
            inputs=["a", "b"],
            condition_inputs=["val"],
            body_inputs=["inp"],
            input_edges={"cond.val": "a", "body.inp": "b"},
        )
        self.assertEqual(len(wn.input_edges), 2)

    def test_invalid_workflow_input(self):
        """Input edge source must be a workflow input."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            make_valid_while_node(
                inputs=["x"],
                condition_inputs=["val"],
                input_edges={"cond.val": "nonexistent"},
            )
        self.assertIn("Invalid input_edges source ports", str(ctx.exception))
        self.assertIn("nonexistent", str(ctx.exception))

    def test_invalid_target_node(self):
        """Input edge target node must be condition or body label."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            make_valid_while_node(
                inputs=["x"],
                input_edges={"unknown.port": "x"},
            )
        exc_str = str(ctx.exception)
        self.assertIn("Invalid input_edges target nodes", exc_str)
        self.assertIn("cond", exc_str)
        self.assertIn("body", exc_str)

    def test_invalid_target_port_on_condition(self):
        """Input edge target port must exist on condition node."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            make_valid_while_node(
                inputs=["x"],
                condition_inputs=["val"],
                input_edges={"cond.wrong": "x"},
            )
        self.assertIn("Invalid input_edges target ports", str(ctx.exception))
        self.assertIn("wrong", str(ctx.exception))

    def test_invalid_target_port_on_body(self):
        """Input edge target port must exist on body node."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            make_valid_while_node(
                inputs=["x"],
                body_inputs=["inp"],
                input_edges={"body.missing": "x"},
            )
        self.assertIn("Invalid input_edges target ports", str(ctx.exception))
        self.assertIn("missing", str(ctx.exception))


class TestWhileNodeOutputEdges(unittest.TestCase):
    def test_valid_output_edge_from_body(self):
        """Output edge from body node."""
        wn = make_valid_while_node(
            inputs=["y"],
            outputs=["y"],
            body_outputs=["out"],
            output_edges={"y": "body.out"},
        )
        self.assertEqual(len(wn.output_edges), 1)

    def test_output_edge_from_condition_rejected(self):
        """Output edges must come from the body node, not the condition."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            make_valid_while_node(
                inputs=["y"],
                outputs=["y"],
                condition_outputs=["result"],
                output_edges={"y": "cond.result"},
            )
        self.assertIn("body node", str(ctx.exception).lower())

    def test_output_edge_passthrough_rejected(self):
        """Output edges must come from the body node, not pass-through input."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            make_valid_while_node(
                inputs=["y"],
                outputs=["y"],
                output_edges={"y": "y"},
            )
        self.assertIn("body node", str(ctx.exception).lower())

    def test_invalid_workflow_output(self):
        """Output edge target must be a workflow output."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            make_valid_while_node(
                inputs=["y"],
                outputs=["y"],
                body_outputs=["out"],
                output_edges={"nonexistent": "body.out"},
            )
        self.assertIn("Invalid output target ports", str(ctx.exception))
        self.assertIn("nonexistent", str(ctx.exception))

    def test_invalid_source_node(self):
        """Output edge source node must be condition or body label."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            make_valid_while_node(
                inputs=["y"],
                outputs=["y"],
                output_edges={"y": "unknown.port"},
            )
        exc_str = str(ctx.exception)
        self.assertIn("Invalid output source nodes", exc_str)
        self.assertIn("unknown.port", exc_str)

    def test_invalid_source_port_on_body(self):
        """Output edge source port must exist on body node."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            make_valid_while_node(
                inputs=["y"],
                outputs=["y"],
                body_outputs=["out"],
                output_edges={"y": "body.missing"},
            )
        self.assertIn("Invalid output source ports", str(ctx.exception))
        self.assertIn("missing", str(ctx.exception))


class TestWhileNodeInferredIterationEdges(unittest.TestCase):
    """Tests for the derived body_body_edges and body_condition_edges properties."""

    def test_body_body_edges_inferred(self):
        """body→body edges are inferred from output_edges ∘ input_edges."""
        wn = while_model.WhileNode(
            inputs=["n", "acc"],
            outputs=["n", "acc"],
            case=helper_models.ConditionalCase(
                condition=helper_models.LabeledNode(
                    label="check",
                    node=make_atomic(["val"], ["flag"]),
                ),
                body=helper_models.LabeledNode(
                    label="step",
                    node=make_atomic(["x", "y"], ["a", "b"]),
                ),
            ),
            input_edges={
                "check.val": "n",
                "step.x": "n",
                "step.y": "acc",
            },
            output_edges={"n": "step.a", "acc": "step.b"},
        )
        bb = wn.body_body_edges
        self.assertEqual(len(bb), 2)
        self.assertEqual(
            bb[edge_models.TargetHandle(node="step", port="x")],
            edge_models.SourceHandle(node="step", port="a"),
        )
        self.assertEqual(
            bb[edge_models.TargetHandle(node="step", port="y")],
            edge_models.SourceHandle(node="step", port="b"),
        )

    def test_body_condition_edges_inferred(self):
        """body→condition edges are inferred from output_edges ∘ input_edges."""
        wn = while_model.WhileNode(
            inputs=["n", "acc"],
            outputs=["n"],
            case=helper_models.ConditionalCase(
                condition=helper_models.LabeledNode(
                    label="check",
                    node=make_atomic(["val"], ["flag"]),
                ),
                body=helper_models.LabeledNode(
                    label="step",
                    node=make_atomic(["x", "y"], ["a", "b"]),
                ),
            ),
            input_edges={
                "check.val": "n",
                "step.x": "n",
                "step.y": "acc",
            },
            output_edges={"n": "step.a"},
        )
        bc = wn.body_condition_edges
        self.assertEqual(len(bc), 1)
        self.assertEqual(
            bc[edge_models.TargetHandle(node="check", port="val")],
            edge_models.SourceHandle(node="step", port="a"),
        )

    def test_no_inferred_edges_when_no_output_overlap(self):
        """No iteration edges when input_edges don't reference output labels."""
        wn = make_valid_while_node(
            inputs=["x", "y"],
            outputs=["x"],
            body_inputs=["inp"],
            body_outputs=["out"],
            condition_inputs=["val"],
            input_edges={"body.inp": "x", "cond.val": "y"},
            output_edges={"x": "body.out"},
        )
        # cond.val comes from "y" which is not an output, so no body→condition edge
        self.assertEqual(len(wn.body_condition_edges), 0)
        # body.inp comes from "x" which IS an output, so body→body edge exists
        self.assertEqual(len(wn.body_body_edges), 1)

    def test_empty_inferred_edges_for_minimal_node(self):
        """Minimal node with no output edges produces no inferred edges."""
        wn = make_valid_while_node()
        self.assertEqual(wn.body_body_edges, {})
        self.assertEqual(wn.body_condition_edges, {})

    def test_inferred_edges_only_for_matching_target(self):
        """body_body_edges excludes condition targets and vice versa."""
        wn = while_model.WhileNode(
            inputs=["a"],
            outputs=["a"],
            case=helper_models.ConditionalCase(
                condition=helper_models.LabeledNode(
                    label="cond",
                    node=make_atomic(["v"], ["ok"]),
                ),
                body=helper_models.LabeledNode(
                    label="body",
                    node=make_atomic(["p"], ["q"]),
                ),
            ),
            input_edges={"cond.v": "a", "body.p": "a"},
            output_edges={"a": "body.q"},
        )
        # body.p ← a is in output_edges, so body→body edge
        self.assertEqual(len(wn.body_body_edges), 1)
        # cond.v ← a is also in output_edges, so body→condition edge
        self.assertEqual(len(wn.body_condition_edges), 1)
        # But they don't leak into each other
        self.assertNotIn(
            edge_models.TargetHandle(node="cond", port="v"),
            wn.body_body_edges,
        )
        self.assertNotIn(
            edge_models.TargetHandle(node="body", port="p"),
            wn.body_condition_edges,
        )


class TestWhileNodeSerialization(unittest.TestCase):
    def test_minimal_roundtrip(self):
        """Minimal WhileNode roundtrip."""
        original = make_valid_while_node()
        for mode in ["json", "python"]:
            with self.subTest(mode=mode):
                data = original.model_dump(mode=mode)
                restored = while_model.WhileNode.model_validate(data)
                self.assertEqual(original.inputs, restored.inputs)
                self.assertEqual(original.outputs, restored.outputs)
                self.assertEqual(
                    original.case.condition.label, restored.case.condition.label
                )
                self.assertEqual(original.case.body.label, restored.case.body.label)

    def test_fully_wired_roundtrip(self):
        """Fully wired WhileNode roundtrip."""
        original = while_model.WhileNode(
            inputs=["n", "acc"],
            outputs=["n", "acc"],
            case=helper_models.ConditionalCase(
                condition=helper_models.LabeledNode(
                    label="check",
                    node=make_atomic(["val"], ["flag"]),
                ),
                body=helper_models.LabeledNode(
                    label="step",
                    node=make_atomic(["x", "y"], ["a", "b"]),
                ),
            ),
            input_edges={"check.val": "n", "step.x": "n", "step.y": "acc"},
            output_edges={"n": "step.a", "acc": "step.b"},
        )
        for mode in ["json", "python"]:
            with self.subTest(mode=mode):
                data = original.model_dump(mode=mode)
                restored = while_model.WhileNode.model_validate(data)

                self.assertEqual(original.inputs, restored.inputs)
                self.assertEqual(original.outputs, restored.outputs)
                self.assertEqual(original.input_edges, restored.input_edges)
                self.assertEqual(original.output_edges, restored.output_edges)
                # Derived properties should also match
                self.assertEqual(original.body_body_edges, restored.body_body_edges)
                self.assertEqual(
                    original.body_condition_edges, restored.body_condition_edges
                )

    def test_inferred_edges_not_serialized(self):
        """body_body_edges and body_condition_edges should not appear in dump."""
        wn = while_model.WhileNode(
            inputs=["a"],
            outputs=["a"],
            case=helper_models.ConditionalCase(
                condition=helper_models.LabeledNode(
                    label="cond",
                    node=make_atomic(["v"], ["ok"]),
                ),
                body=helper_models.LabeledNode(
                    label="body",
                    node=make_atomic(["p"], ["q"]),
                ),
            ),
            input_edges={"cond.v": "a", "body.p": "a"},
            output_edges={"a": "body.q"},
        )
        data = wn.model_dump(mode="json")
        self.assertNotIn("body_body_edges", data)
        self.assertNotIn("body_condition_edges", data)

    def test_edge_serialization_format(self):
        """Edges serialize to dot-notation strings."""
        wn = while_model.WhileNode(
            inputs=["x"],
            outputs=["x"],
            case=helper_models.ConditionalCase(
                condition=helper_models.LabeledNode(
                    label="cond",
                    node=make_atomic(["inp"], ["out"]),
                ),
                body=helper_models.LabeledNode(
                    label="body",
                    node=make_atomic(["a"], ["b"]),
                ),
            ),
            input_edges={"cond.inp": "x", "body.a": "x"},
            output_edges={"x": "body.b"},
        )
        data = wn.model_dump(mode="json")

        # input_edges: target is "node.port", source is "port"
        self.assertIn("cond.inp", data["input_edges"])
        self.assertEqual(data["input_edges"]["cond.inp"], "x")

        # output_edges: target is "port", source is "node.port"
        self.assertIn("x", data["output_edges"])
        self.assertEqual(data["output_edges"]["x"], "body.b")

    def test_nested_workflow_in_body(self):
        """WhileNode can contain WorkflowNode in body."""
        inner = workflow_model.WorkflowNode(
            inputs=["a"],
            outputs=["b"],
            nodes={"leaf": make_atomic(["x"], ["y"])},
            input_edges={"leaf.x": "a"},
            edges={},
            output_edges={"b": "leaf.y"},
        )
        wn = while_model.WhileNode(
            inputs=["start"],
            outputs=["start"],
            case=helper_models.ConditionalCase(
                condition=helper_models.LabeledNode(
                    label="check",
                    node=make_atomic(["v"], ["done"]),
                ),
                body=helper_models.LabeledNode(label="process", node=inner),
            ),
            input_edges={"process.a": "start"},
            output_edges={"start": "process.b"},
        )
        data = wn.model_dump(mode="json")
        restored = while_model.WhileNode.model_validate(data)
        self.assertIsInstance(restored.case.body.node, workflow_model.WorkflowNode)


class TestWhileNodeEdgeCases(unittest.TestCase):
    def test_empty_inputs_outputs(self):
        """WhileNode with no inputs or outputs."""
        wn = while_model.WhileNode(
            inputs=[],
            outputs=[],
            case=helper_models.ConditionalCase(
                condition=helper_models.LabeledNode(
                    label="c",
                    node=make_atomic([], ["done"]),
                ),
                body=helper_models.LabeledNode(
                    label="b",
                    node=make_atomic([], []),
                ),
            ),
            input_edges={},
            output_edges={},
        )
        self.assertEqual(wn.inputs, [])
        self.assertEqual(wn.outputs, [])

    def test_condition_with_zero_outputs_rejected(self):
        """Condition must have at least one output for evaluation."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            helper_models.ConditionalCase(
                condition=helper_models.LabeledNode(
                    label="c",
                    node=make_atomic([], []),  # No outputs!
                ),
                body=helper_models.LabeledNode(
                    label="b",
                    node=make_atomic([], []),
                ),
            )
        self.assertIn("exactly one output", str(ctx.exception))

    def test_same_input_feeds_both_condition_and_body(self):
        """
        Same input can feed both condition and body via input_edges.

        This is normal — on the first iteration both get data from the loop input.
        """
        wn = while_model.WhileNode(
            inputs=["x"],
            outputs=["x"],
            case=helper_models.ConditionalCase(
                condition=helper_models.LabeledNode(
                    label="cond",
                    node=make_atomic(["val"], ["ok"]),
                ),
                body=helper_models.LabeledNode(
                    label="body",
                    node=make_atomic(["inp"], ["out"]),
                ),
            ),
            input_edges={"body.inp": "x", "cond.val": "x"},
            output_edges={"x": "body.out"},
        )
        self.assertEqual(len(wn.input_edges), 2)
        # Both inferred edge types should pick up the "x" correspondence
        self.assertEqual(len(wn.body_body_edges), 1)
        self.assertEqual(len(wn.body_condition_edges), 1)


if __name__ == "__main__":
    unittest.main()
