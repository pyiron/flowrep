"""Unit tests for WhileNode and related classes."""

import unittest

import pydantic

from flowrep.models import base_models
from flowrep.models.nodes import (
    atomic_model,
    helper_models,
    subgraph_protocols,
    union,
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
    body_body_edges: dict | None = None,
    body_condition_edges: dict | None = None,
) -> while_model.WhileNode:
    inputs = inputs if inputs is not None else ["x"]
    outputs = outputs if outputs is not None else ["y"]
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
        body_body_edges=body_body_edges if body_body_edges is not None else {},
        body_condition_edges=(
            body_condition_edges if body_condition_edges is not None else {}
        ),
    )


class TestWhileNodeBasic(unittest.TestCase):
    def test_schema_generation(self):
        """model_json_schema() fails if forward refs aren't resolved."""
        while_model.WhileNode.model_json_schema()

    def test_obeys_build_subgraph_with_static_output(self):
        """WhileNode should obey build subgraph with static output."""
        node = make_valid_while_node()
        self.assertIsInstance(node, subgraph_protocols.BuildsSubgraphWithStaticOutput)

    def test_valid_minimal(self):
        """
        Minimal valid WhileNode with empty edges.

        Subject to change -- do we want to allow a trivial while node at all?
        """
        wn = make_valid_while_node()
        self.assertEqual(wn.type, base_models.RecipeElementType.WHILE)
        self.assertEqual(wn.inputs, ["x"])
        self.assertEqual(wn.outputs, ["y"])
        self.assertEqual(wn.input_edges, {})
        self.assertEqual(wn.output_edges, {})
        self.assertEqual(wn.body_body_edges, {})
        self.assertEqual(wn.body_condition_edges, {})

    def test_valid_fully_wired(self):
        """WhileNode with all edge types populated."""
        wn = while_model.WhileNode(
            inputs=["n", "acc"],
            outputs=["result"],
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
            output_edges={"result": "decrement.next_total"},
            body_body_edges={
                "decrement.current": "decrement.next_val",
                "decrement.total": "decrement.next_total",
            },
            body_condition_edges={"check.val": "decrement.next_val"},
        )
        self.assertEqual(len(wn.input_edges), 3)
        self.assertEqual(len(wn.output_edges), 1)
        self.assertEqual(len(wn.body_body_edges), 2)
        self.assertEqual(len(wn.body_condition_edges), 1)


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
            make_valid_while_node(outputs=["a", "b", "a"])
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
                make_valid_while_node(outputs=[invalid])


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
        self.assertIn("not a workflow input", str(ctx.exception))
        self.assertIn("nonexistent", str(ctx.exception))

    def test_invalid_target_node(self):
        """Input edge target node must be condition or body label."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            make_valid_while_node(
                inputs=["x"],
                input_edges={"unknown.port": "x"},
            )
        exc_str = str(ctx.exception)
        self.assertIn("must be", exc_str)
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
        self.assertIn("has no input port", str(ctx.exception))
        self.assertIn("wrong", str(ctx.exception))

    def test_invalid_target_port_on_body(self):
        """Input edge target port must exist on body node."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            make_valid_while_node(
                inputs=["x"],
                body_inputs=["inp"],
                input_edges={"body.missing": "x"},
            )
        self.assertIn("has no input port", str(ctx.exception))
        self.assertIn("missing", str(ctx.exception))


class TestWhileNodeOutputEdges(unittest.TestCase):
    def test_valid_output_edge_from_condition(self):
        """Output edge from condition node."""
        wn = make_valid_while_node(
            outputs=["y"],
            condition_outputs=["result"],
            output_edges={"y": "cond.result"},
        )
        self.assertEqual(len(wn.output_edges), 1)

    def test_valid_output_edge_from_body(self):
        """Output edge from body node."""
        wn = make_valid_while_node(
            outputs=["y"],
            body_outputs=["out"],
            output_edges={"y": "body.out"},
        )
        self.assertEqual(len(wn.output_edges), 1)

    def test_valid_output_edges_from_both(self):
        """Output edges from both condition and body."""
        wn = make_valid_while_node(
            outputs=["a", "b"],
            condition_outputs=["flag"],
            body_outputs=["result"],
            output_edges={"a": "cond.flag", "b": "body.result"},
        )
        self.assertEqual(len(wn.output_edges), 2)

    def test_invalid_workflow_output(self):
        """Output edge target must be a workflow output."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            make_valid_while_node(
                outputs=["y"],
                body_outputs=["out"],
                output_edges={"nonexistent": "body.out"},
            )
        self.assertIn("not a workflow output", str(ctx.exception))
        self.assertIn("nonexistent", str(ctx.exception))

    def test_invalid_source_node(self):
        """Output edge source node must be condition or body label."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            make_valid_while_node(
                outputs=["y"],
                output_edges={"y": "unknown.port"},
            )
        exc_str = str(ctx.exception)
        self.assertIn("must be", exc_str)
        self.assertIn("cond", exc_str)
        self.assertIn("body", exc_str)

    def test_invalid_source_port_on_condition(self):
        """Output edge source port must exist on condition node."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            make_valid_while_node(
                outputs=["y"],
                condition_outputs=["result"],
                output_edges={"y": "cond.wrong"},
            )
        self.assertIn("has no output port", str(ctx.exception))
        self.assertIn("wrong", str(ctx.exception))

    def test_invalid_source_port_on_body(self):
        """Output edge source port must exist on body node."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            make_valid_while_node(
                outputs=["y"],
                body_outputs=["out"],
                output_edges={"y": "body.missing"},
            )
        self.assertIn("has no output port", str(ctx.exception))
        self.assertIn("missing", str(ctx.exception))


class TestWhileNodeBodyBodyEdges(unittest.TestCase):
    def test_valid_body_body_edge(self):
        """Valid edge from body output to body input."""
        wn = make_valid_while_node(
            body_inputs=["current"],
            body_outputs=["next_val"],
            body_label="loop",
            body_body_edges={"loop.current": "loop.next_val"},
        )
        self.assertEqual(len(wn.body_body_edges), 1)

    def test_valid_multiple_body_body_edges(self):
        """Multiple body-body edges for loop-carried state."""
        wn = make_valid_while_node(
            body_inputs=["a", "b"],
            body_outputs=["x", "y"],
            body_label="iter",
            body_body_edges={
                "iter.a": "iter.x",
                "iter.b": "iter.y",
            },
        )
        self.assertEqual(len(wn.body_body_edges), 2)

    def test_invalid_source_node(self):
        """Body-body edge source must be body label."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            make_valid_while_node(
                body_inputs=["inp"],
                body_outputs=["out"],
                body_label="body",
                body_body_edges={"body.inp": "wrong.out"},
            )
        self.assertIn("node must be 'body'", str(ctx.exception))
        self.assertIn("wrong", str(ctx.exception))

    def test_invalid_source_port(self):
        """Body-body edge source port must be body output."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            make_valid_while_node(
                body_inputs=["inp"],
                body_outputs=["out"],
                body_label="body",
                body_body_edges={"body.inp": "body.missing"},
            )
        self.assertIn("not an output of body node", str(ctx.exception))
        self.assertIn("missing", str(ctx.exception))

    def test_invalid_target_node(self):
        """Body-body edge target must be body label."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            make_valid_while_node(
                body_inputs=["inp"],
                body_outputs=["out"],
                body_label="body",
                body_body_edges={"wrong.inp": "body.out"},
            )
        self.assertIn("node must be 'body'", str(ctx.exception))
        self.assertIn("wrong", str(ctx.exception))

    def test_invalid_target_port(self):
        """Body-body edge target port must be body input."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            make_valid_while_node(
                body_inputs=["inp"],
                body_outputs=["out"],
                body_label="body",
                body_body_edges={"body.missing": "body.out"},
            )
        self.assertIn("not an input of body node", str(ctx.exception))
        self.assertIn("missing", str(ctx.exception))


class TestWhileNodeBodyConditionEdges(unittest.TestCase):
    def test_valid_body_condition_edge(self):
        """Valid edge from body output to condition input."""
        wn = make_valid_while_node(
            condition_inputs=["val"],
            body_outputs=["next_val"],
            condition_label="check",
            body_label="step",
            body_condition_edges={"check.val": "step.next_val"},
        )
        self.assertEqual(len(wn.body_condition_edges), 1)

    def test_valid_multiple_body_condition_edges(self):
        """Multiple edges from body to condition."""
        wn = make_valid_while_node(
            condition_inputs=["a", "b"],
            body_outputs=["x", "y"],
            condition_label="cond",
            body_label="body",
            body_condition_edges={
                "cond.a": "body.x",
                "cond.b": "body.y",
            },
        )
        self.assertEqual(len(wn.body_condition_edges), 2)

    def test_invalid_source_node(self):
        """Body-condition edge source must be body label."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            make_valid_while_node(
                condition_inputs=["val"],
                body_outputs=["out"],
                condition_label="cond",
                body_label="body",
                body_condition_edges={"cond.val": "wrong.out"},
            )
        self.assertIn("node must be 'body'", str(ctx.exception))
        self.assertIn("wrong", str(ctx.exception))

    def test_invalid_source_port(self):
        """Body-condition edge source port must be body output."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            make_valid_while_node(
                condition_inputs=["val"],
                body_outputs=["out"],
                condition_label="cond",
                body_label="body",
                body_condition_edges={"cond.val": "body.missing"},
            )
        self.assertIn("not an output of body node", str(ctx.exception))
        self.assertIn("missing", str(ctx.exception))

    def test_invalid_target_node(self):
        """Body-condition edge target must be condition label."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            make_valid_while_node(
                condition_inputs=["val"],
                body_outputs=["out"],
                condition_label="cond",
                body_label="body",
                body_condition_edges={"wrong.val": "body.out"},
            )
        self.assertIn("node must be 'cond'", str(ctx.exception))
        self.assertIn("wrong", str(ctx.exception))

    def test_invalid_target_port(self):
        """Body-condition edge target port must be condition input."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            make_valid_while_node(
                condition_inputs=["val"],
                body_outputs=["out"],
                condition_label="cond",
                body_label="body",
                body_condition_edges={"cond.missing": "body.out"},
            )
        self.assertIn("not an input of condition node", str(ctx.exception))
        self.assertIn("missing", str(ctx.exception))


class TestWhileNodeSerialization(unittest.TestCase):
    def test_minimal_roundtrip(self):
        """
        Minimal WhileNode roundtrip.

        Subject to change if we disallow trivial while loops
        """
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
            outputs=["result", "count"],
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
            output_edges={"result": "step.a", "count": "check.flag"},
            body_body_edges={"step.x": "step.a", "step.y": "step.b"},
            body_condition_edges={"check.val": "step.a"},
        )
        for mode in ["json", "python"]:
            with self.subTest(mode=mode):
                data = original.model_dump(mode=mode)
                restored = while_model.WhileNode.model_validate(data)

                self.assertEqual(original.inputs, restored.inputs)
                self.assertEqual(original.outputs, restored.outputs)
                self.assertEqual(original.input_edges, restored.input_edges)
                self.assertEqual(original.output_edges, restored.output_edges)
                self.assertEqual(original.body_body_edges, restored.body_body_edges)
                self.assertEqual(
                    original.body_condition_edges, restored.body_condition_edges
                )

    def test_edge_serialization_format(self):
        """Edges serialize to dot-notation strings."""
        wn = while_model.WhileNode(
            inputs=["x"],
            outputs=["y"],
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
            output_edges={"y": "body.b"},
            body_body_edges={"body.a": "body.b"},
            body_condition_edges={"cond.inp": "body.b"},
        )
        data = wn.model_dump(mode="json")

        # input_edges: target is "node.port", source is "port"
        self.assertIn("cond.inp", data["input_edges"])
        self.assertEqual(data["input_edges"]["cond.inp"], "x")

        # output_edges: target is "port", source is "node.port"
        self.assertIn("y", data["output_edges"])
        self.assertEqual(data["output_edges"]["y"], "body.b")

        # body_body_edges: both are "node.port"
        self.assertIn("body.a", data["body_body_edges"])
        self.assertEqual(data["body_body_edges"]["body.a"], "body.b")

        # body_condition_edges: both are "node.port"
        self.assertIn("cond.inp", data["body_condition_edges"])
        self.assertEqual(data["body_condition_edges"]["cond.inp"], "body.b")

    def test_discriminated_union_roundtrip(self):
        """WhileNode correctly deserializes via NodeType discriminator."""
        data = {
            "type": "while",
            "inputs": ["x"],
            "outputs": ["y"],
            "case": {
                "condition": {
                    "label": "c",
                    "node": {
                        "type": "atomic",
                        "fully_qualified_name": "m.f",
                        "inputs": [],
                        "outputs": ["ok"],
                    },
                },
                "body": {
                    "label": "b",
                    "node": {
                        "type": "atomic",
                        "fully_qualified_name": "m.g",
                        "inputs": [],
                        "outputs": [],
                    },
                },
            },
            "input_edges": {},
            "output_edges": {},
            "body_body_edges": {},
            "body_condition_edges": {},
        }
        node = pydantic.TypeAdapter(union.NodeType).validate_python(data)
        self.assertIsInstance(node, while_model.WhileNode)

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
            outputs=["end"],
            case=helper_models.ConditionalCase(
                condition=helper_models.LabeledNode(
                    label="check",
                    node=make_atomic(["v"], ["done"]),
                ),
                body=helper_models.LabeledNode(label="process", node=inner),
            ),
            input_edges={"process.a": "start"},
            output_edges={"end": "process.b"},
            body_body_edges={},
            body_condition_edges={},
        )
        data = wn.model_dump(mode="json")
        restored = while_model.WhileNode.model_validate(data)
        self.assertIsInstance(restored.case.body.node, workflow_model.WorkflowNode)


class TestWhileNodeEdgeCases(unittest.TestCase):
    def test_empty_inputs_outputs(self):
        """
        WhileNode with no inputs or outputs.

        More subject to change than anything else -- if this fails it's probably because
        it got outlawed
        """
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
            body_body_edges={},
            body_condition_edges={},
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

    def test_same_port_in_multiple_edge_types(self):
        """
        Same target port can appear in different edge dicts.

        This is not only valid, it is the point. On the first iteration the only data
        available is the input. At subsequent iterations, the other edges allow
        inter-iteration dataflow. It is the responsibility of the WfMS to resolve this
        behavior at runtime. Here, we only specify the loop flow.
        """

        wn = while_model.WhileNode(
            inputs=["x"],
            outputs=["y"],
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
            input_edges={"body.inp": "x"},
            output_edges={"y": "body.out"},
            body_body_edges={"body.inp": "body.out"},
            body_condition_edges={},
        )
        self.assertEqual(len(wn.input_edges), 1)
        self.assertEqual(len(wn.body_body_edges), 1)


if __name__ == "__main__":
    unittest.main()
