import unittest

import pydantic

from flowrep import base_models, edge_models, subgraph_validation
from flowrep.nodes import (
    atomic_model,
    for_model,
    helper_models,
    workflow_model,
)

from flowrep_static import makers


class TestForEachNodeBasic(unittest.TestCase):
    def test_schema_generation(self):
        """model_json_schema() fails if forward refs aren't resolved."""
        for_model.ForEachNode.model_json_schema()

    def test_obeys_build_subgraph_with_static_output(self):
        """ForEachNode should obey build subgraph with static output."""
        node = for_model.ForEachNode(
            inputs=[],
            outputs=[],
            body_node=makers.make_labeled_atomic(
                "body",
                inputs=["item"],
                outputs=["result"],
                inputs_with_defaults=["item"],
            ),
            input_edges={},
            output_edges={},
            nested_ports=["item"],
        )
        self.assertIsInstance(node, subgraph_validation.DynamicSubgraphStaticOutput)

    def test_valid_for_node_with_nested_ports(self):
        for_node = for_model.ForEachNode(
            inputs=["items"],
            outputs=["results"],
            body_node=makers.make_labeled_atomic(
                "body",
                inputs=["item"],
                outputs=["result"],
            ),
            input_edges={
                edge_models.TargetHandle(
                    node="body", port="item"
                ): edge_models.InputSource(port="items"),
            },
            output_edges={
                edge_models.OutputTarget(port="results"): edge_models.SourceHandle(
                    node="body", port="result"
                ),
            },
            nested_ports=["item"],
        )
        self.assertEqual(for_node.type, base_models.RecipeElementType.FOR)
        self.assertEqual(for_node.nested_ports, ["item"])
        self.assertEqual(for_node.zipped_ports, [])

    def test_valid_for_node_with_zipped_ports(self):
        for_node = for_model.ForEachNode(
            inputs=["xs", "ys"],
            outputs=["results"],
            body_node=makers.make_labeled_atomic(
                "body",
                inputs=["x", "y"],
                outputs=["result"],
            ),
            input_edges={
                edge_models.TargetHandle(
                    node="body", port="x"
                ): edge_models.InputSource(port="xs"),
                edge_models.TargetHandle(
                    node="body", port="y"
                ): edge_models.InputSource(port="ys"),
            },
            output_edges={
                edge_models.OutputTarget(port="results"): edge_models.SourceHandle(
                    node="body", port="result"
                ),
            },
            zipped_ports=["x", "y"],
        )
        self.assertEqual(for_node.zipped_ports, ["x", "y"])
        self.assertEqual(for_node.nested_ports, [])

    def test_valid_for_node_with_both_nested_and_zipped(self):
        for_node = for_model.ForEachNode(
            inputs=["outer", "inner1", "inner2"],
            outputs=["results"],
            body_node=makers.make_labeled_atomic(
                "body",
                inputs=["a", "b", "c"],
                outputs=["out"],
            ),
            input_edges={
                edge_models.TargetHandle(
                    node="body", port="a"
                ): edge_models.InputSource(port="outer"),
                edge_models.TargetHandle(
                    node="body", port="b"
                ): edge_models.InputSource(port="inner1"),
                edge_models.TargetHandle(
                    node="body", port="c"
                ): edge_models.InputSource(port="inner2"),
            },
            output_edges={
                edge_models.OutputTarget(port="results"): edge_models.SourceHandle(
                    node="body", port="out"
                ),
            },
            nested_ports=["a"],
            zipped_ports=["b", "c"],
        )
        self.assertEqual(for_node.nested_ports, ["a"])
        self.assertEqual(for_node.zipped_ports, ["b", "c"])

    def test_call_raises(self):
        recipe = for_model.ForEachNode(
            inputs=["x"],
            outputs=[],
            body_node=makers.make_labeled_atomic(
                "body",
                inputs=["item"],
                outputs=["result"],
                inputs_with_defaults=["item"],
            ),
            input_edges={},
            output_edges={},
            nested_ports=["item"],
        )
        with self.assertRaises(NotImplementedError):
            recipe(42)


class TestForEachNodeLoopPortValidation(unittest.TestCase):
    def test_no_iteration_ports_rejected(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            for_model.ForEachNode(
                inputs=["x"],
                outputs=["y"],
                body_node=makers.make_labeled_atomic(
                    "body",
                    inputs=["inp"],
                    outputs=["out"],
                ),
                input_edges={
                    edge_models.TargetHandle(
                        node="body", port="inp"
                    ): edge_models.InputSource(port="x"),
                },
                output_edges={
                    edge_models.OutputTarget(port="y"): edge_models.SourceHandle(
                        node="body", port="out"
                    ),
                },
                nested_ports=[],
                zipped_ports=[],
            )
        self.assertIn("at least one", str(ctx.exception).lower())

    def test_duplicate_nested_ports_rejected(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            for_model.ForEachNode(
                inputs=["items"],
                outputs=["results"],
                body_node=makers.make_labeled_atomic(
                    "body",
                    inputs=["item"],
                    outputs=["result"],
                ),
                input_edges={
                    edge_models.TargetHandle(
                        node="body", port="item"
                    ): edge_models.InputSource(port="items"),
                },
                output_edges={
                    edge_models.OutputTarget(port="results"): edge_models.SourceHandle(
                        node="body", port="result"
                    ),
                },
                nested_ports=["item", "item"],
            )
        self.assertIn("unique", str(ctx.exception).lower())

    def test_duplicate_zipped_ports_rejected(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            for_model.ForEachNode(
                inputs=["xs"],
                outputs=["results"],
                body_node=makers.make_labeled_atomic(
                    "body",
                    inputs=["x"],
                    outputs=["result"],
                ),
                input_edges={
                    edge_models.TargetHandle(
                        node="body", port="x"
                    ): edge_models.InputSource(port="xs"),
                },
                output_edges={
                    edge_models.OutputTarget(port="results"): edge_models.SourceHandle(
                        node="body", port="result"
                    ),
                },
                zipped_ports=["x", "x"],
            )
        self.assertIn("unique", str(ctx.exception).lower())

    def test_overlapping_nested_and_zipped_rejected(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            for_model.ForEachNode(
                inputs=["items"],
                outputs=["results"],
                body_node=makers.make_labeled_atomic(
                    "body",
                    inputs=["item"],
                    outputs=["result"],
                ),
                input_edges={
                    edge_models.TargetHandle(
                        node="body", port="item"
                    ): edge_models.InputSource(port="items"),
                },
                output_edges={
                    edge_models.OutputTarget(port="results"): edge_models.SourceHandle(
                        node="body", port="result"
                    ),
                },
                nested_ports=["item"],
                zipped_ports=["item"],
            )
        self.assertIn("overlap", str(ctx.exception).lower())
        self.assertIn("item", str(ctx.exception))

    def test_iteration_port_not_on_body_node_rejected(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            for_model.ForEachNode(
                inputs=["items"],
                outputs=["results"],
                body_node=makers.make_labeled_atomic(
                    "body",
                    inputs=["item"],
                    outputs=["result"],
                ),
                input_edges={
                    edge_models.TargetHandle(
                        node="body", port="item"
                    ): edge_models.InputSource(port="items"),
                },
                output_edges={
                    edge_models.OutputTarget(port="results"): edge_models.SourceHandle(
                        node="body", port="result"
                    ),
                },
                nested_ports=["nonexistent"],
            )
        self.assertIn("nonexistent", str(ctx.exception))


class TestForEachNodeInputEdges(unittest.TestCase):
    def test_input_edge_wrong_target_node_rejected(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            for_model.ForEachNode(
                inputs=["items"],
                outputs=["results"],
                body_node=makers.make_labeled_atomic(
                    "body",
                    inputs=["item"],
                    outputs=["result"],
                ),
                input_edges={
                    edge_models.TargetHandle(
                        node="wrong_node", port="item"
                    ): edge_models.InputSource(port="items"),
                },
                output_edges={
                    edge_models.OutputTarget(port="results"): edge_models.SourceHandle(
                        node="body", port="result"
                    ),
                },
                nested_ports=["item"],
            )
        self.assertIn("body", str(ctx.exception))

    def test_input_edge_wrong_target_port_rejected(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            for_model.ForEachNode(
                inputs=["items"],
                outputs=["results"],
                body_node=makers.make_labeled_atomic(
                    "body",
                    inputs=["item"],
                    outputs=["result"],
                    inputs_with_defaults=["item"],
                ),
                input_edges={
                    edge_models.TargetHandle(
                        node="body", port="wrong_port"
                    ): edge_models.InputSource(port="items"),
                },
                output_edges={
                    edge_models.OutputTarget(port="results"): edge_models.SourceHandle(
                        node="body", port="result"
                    ),
                },
                nested_ports=["item"],
            )
        self.assertIn("wrong_port", str(ctx.exception))

    def test_input_edge_invalid_source_port_rejected(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            for_model.ForEachNode(
                inputs=["items"],
                outputs=["results"],
                body_node=makers.make_labeled_atomic(
                    "body",
                    inputs=["item"],
                    outputs=["result"],
                ),
                input_edges={
                    edge_models.TargetHandle(
                        node="body", port="item"
                    ): edge_models.InputSource(port="nonexistent"),
                },
                output_edges={
                    edge_models.OutputTarget(port="results"): edge_models.SourceHandle(
                        node="body", port="result"
                    ),
                },
                nested_ports=["item"],
            )
        self.assertIn("nonexistent", str(ctx.exception))


class TestForEachNodeFullySourcing(unittest.TestCase):
    """Tests for validate_internal_data_completeness on for-node body."""

    def test_body_unsourced_no_default_raises(self):
        """Body input without edge or default → rejected."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            for_model.ForEachNode(
                inputs=["xs"],
                outputs=["results"],
                body_node=makers.make_labeled_atomic(
                    "body",
                    inputs=["x", "extra"],
                    outputs=["result"],
                ),
                input_edges={
                    edge_models.TargetHandle(
                        node="body", port="x"
                    ): edge_models.InputSource(port="xs"),
                },
                output_edges={
                    edge_models.OutputTarget(port="results"): edge_models.SourceHandle(
                        node="body", port="result"
                    ),
                },
                nested_ports=["x"],
            )
        self.assertIn("body.extra", str(ctx.exception))

    def test_body_unsourced_with_default_passes(self):
        """Body input without edge but with default → accepted."""
        node = for_model.ForEachNode(
            inputs=["xs"],
            outputs=["results"],
            body_node=makers.make_labeled_atomic(
                "body",
                inputs=["x", "extra"],
                outputs=["result"],
                inputs_with_defaults=["extra"],
            ),
            input_edges={
                edge_models.TargetHandle(
                    node="body", port="x"
                ): edge_models.InputSource(port="xs"),
            },
            output_edges={
                edge_models.OutputTarget(port="results"): edge_models.SourceHandle(
                    node="body", port="result"
                ),
            },
            nested_ports=["x"],
        )
        self.assertIn("extra", node.body_node.node.inputs)

    def test_body_mixed_sourcing_one_unsourced_raises(self):
        """Multiple body inputs: edged, defaulted, and unsourced → fails."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            for_model.ForEachNode(
                inputs=["xs"],
                outputs=["results"],
                body_node=makers.make_labeled_atomic(
                    "body",
                    inputs=["x", "y", "z"],
                    outputs=["result"],
                    inputs_with_defaults=["y"],
                ),
                input_edges={
                    edge_models.TargetHandle(
                        node="body", port="x"
                    ): edge_models.InputSource(port="xs"),
                    # y: no edge, but has default — ok
                    # z: no edge, no default — fail
                },
                output_edges={
                    edge_models.OutputTarget(port="results"): edge_models.SourceHandle(
                        node="body", port="result"
                    ),
                },
                nested_ports=["x"],
            )
        exc_str = str(ctx.exception)
        self.assertIn("body.z", exc_str)
        self.assertNotIn("body.y", exc_str)


class TestForEachNodeOutputEdges(unittest.TestCase):
    def test_output_edge_wrong_source_node_rejected(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            for_model.ForEachNode(
                inputs=["items"],
                outputs=["results"],
                body_node=makers.make_labeled_atomic(
                    "body",
                    inputs=["item"],
                    outputs=["result"],
                ),
                input_edges={
                    edge_models.TargetHandle(
                        node="body", port="item"
                    ): edge_models.InputSource(port="items"),
                },
                output_edges={
                    edge_models.OutputTarget(port="results"): edge_models.SourceHandle(
                        node="wrong_node", port="result"
                    ),
                },
                nested_ports=["item"],
            )
        self.assertIn("Invalid output source nodes", str(ctx.exception))
        self.assertIn("wrong_node", str(ctx.exception))

    def test_output_edge_wrong_source_port_rejected(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            for_model.ForEachNode(
                inputs=["items"],
                outputs=["results"],
                body_node=makers.make_labeled_atomic(
                    "body",
                    inputs=["item"],
                    outputs=["result"],
                ),
                input_edges={
                    edge_models.TargetHandle(
                        node="body", port="item"
                    ): edge_models.InputSource(port="items"),
                },
                output_edges={
                    edge_models.OutputTarget(port="results"): edge_models.SourceHandle(
                        node="body", port="wrong_port"
                    ),
                },
                nested_ports=["item"],
            )
        self.assertIn("wrong_port", str(ctx.exception))

    def test_output_edge_invalid_target_port_rejected(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            for_model.ForEachNode(
                inputs=["items"],
                outputs=["results"],
                body_node=makers.make_labeled_atomic(
                    "body",
                    inputs=["item"],
                    outputs=["result"],
                ),
                input_edges={
                    edge_models.TargetHandle(
                        node="body", port="item"
                    ): edge_models.InputSource(port="items"),
                },
                output_edges={
                    edge_models.OutputTarget(
                        port="nonexistent"
                    ): edge_models.SourceHandle(node="body", port="result"),
                },
                nested_ports=["item"],
            )
        self.assertIn("nonexistent", str(ctx.exception))


class TestForEachNodeTransferEdges(unittest.TestCase):
    def test_valid_transfer_edge(self):
        """Transfer edges should forward iterated inputs to outputs."""
        for_node = for_model.ForEachNode(
            inputs=["items", "broadcast"],
            outputs=["results", "original_items"],
            body_node=makers.make_labeled_atomic(
                "body",
                inputs=["item", "static"],
                outputs=["result"],
            ),
            input_edges={
                edge_models.TargetHandle(
                    node="body", port="item"
                ): edge_models.InputSource(port="items"),
                edge_models.TargetHandle(
                    node="body", port="static"
                ): edge_models.InputSource(port="broadcast"),
            },
            output_edges={
                edge_models.OutputTarget(port="results"): edge_models.SourceHandle(
                    node="body", port="result"
                ),
                edge_models.OutputTarget(
                    port="original_items"
                ): edge_models.InputSource(port="items"),
            },
            nested_ports=["item"],
        )
        self.assertEqual(len(for_node.output_edges), 2)

    def test_transfer_source_not_in_inputs_rejected(self):
        """Transfer edge sources must be ForEachNode inputs."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            for_model.ForEachNode(
                inputs=["items"],
                outputs=["results", "forwarded"],
                body_node=makers.make_labeled_atomic(
                    "body",
                    inputs=["item"],
                    outputs=["result"],
                ),
                input_edges={
                    edge_models.TargetHandle(
                        node="body", port="item"
                    ): edge_models.InputSource(port="items"),
                },
                output_edges={
                    edge_models.OutputTarget(port="results"): edge_models.SourceHandle(
                        node="body", port="result"
                    ),
                    edge_models.OutputTarget(port="forwarded"): edge_models.InputSource(
                        port="nonexistent"
                    ),
                },
                nested_ports=["item"],
            )
        self.assertIn("nonexistent", str(ctx.exception))
        self.assertIn("inputs", str(ctx.exception).lower())

    def test_transfer_target_not_in_outputs_rejected(self):
        """Transfer edge targets must be ForEachNode outputs."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            for_model.ForEachNode(
                inputs=["items"],
                outputs=["results"],
                body_node=makers.make_labeled_atomic(
                    "body",
                    inputs=["item"],
                    outputs=["result"],
                ),
                input_edges={
                    edge_models.TargetHandle(
                        node="body", port="item"
                    ): edge_models.InputSource(port="items"),
                },
                output_edges={
                    edge_models.OutputTarget(port="results"): edge_models.SourceHandle(
                        node="body", port="result"
                    ),
                    edge_models.OutputTarget(
                        port="nonexistent"
                    ): edge_models.InputSource(port="items"),
                },
                nested_ports=["item"],
            )
        self.assertIn("Invalid output target ports", str(ctx.exception))
        self.assertIn("nonexistent", str(ctx.exception).lower())


class TestForEachNodeSerialization(unittest.TestCase):
    def test_roundtrip(self):
        original = for_model.ForEachNode(
            inputs=["items", "multiplier"],
            outputs=["results", "original_items"],
            body_node=makers.make_labeled_atomic(
                "body",
                inputs=["item", "mult"],
                outputs=["result"],
            ),
            input_edges={
                edge_models.TargetHandle(
                    node="body", port="item"
                ): edge_models.InputSource(port="items"),
                edge_models.TargetHandle(
                    node="body", port="mult"
                ): edge_models.InputSource(port="multiplier"),
            },
            output_edges={
                edge_models.OutputTarget(port="results"): edge_models.SourceHandle(
                    node="body", port="result"
                ),
                edge_models.OutputTarget(
                    port="original_items"
                ): edge_models.InputSource(port="items"),
            },
            nested_ports=["item"],
        )
        for mode in ["python", "json"]:
            with self.subTest(mode=mode):
                data = original.model_dump(mode=mode)
                restored = for_model.ForEachNode.model_validate(data)

                self.assertEqual(original.inputs, restored.inputs)
                self.assertEqual(original.outputs, restored.outputs)
                self.assertEqual(original.nested_ports, restored.nested_ports)
                self.assertEqual(original.zipped_ports, restored.zipped_ports)
                self.assertEqual(original.body_node.label, restored.body_node.label)
                self.assertEqual(original.input_edges, restored.input_edges)
                self.assertEqual(original.output_edges, restored.output_edges)


class TestForEachNodeComposition(unittest.TestCase):
    def test_workflow_body_node(self):
        inner_workflow = workflow_model.WorkflowNode(
            inputs=["x"],
            outputs=["y"],
            nodes={
                "leaf": makers.make_atomic(
                    inputs=["inp"],
                    outputs=["out"],
                ),
            },
            input_edges={
                edge_models.TargetHandle(
                    node="leaf", port="inp"
                ): edge_models.InputSource(port="x"),
            },
            edges={},
            output_edges={
                edge_models.OutputTarget(port="y"): edge_models.SourceHandle(
                    node="leaf", port="out"
                ),
            },
        )

        for_node = for_model.ForEachNode(
            inputs=["items"],
            outputs=["results"],
            body_node=helper_models.LabeledNode(
                label="body",
                node=inner_workflow,
            ),
            input_edges={
                edge_models.TargetHandle(
                    node="body", port="x"
                ): edge_models.InputSource(port="items"),
            },
            output_edges={
                edge_models.OutputTarget(port="results"): edge_models.SourceHandle(
                    node="body", port="y"
                ),
            },
            nested_ports=["x"],
        )
        self.assertIsInstance(for_node.body_node.node, workflow_model.WorkflowNode)

    def test_for_node_as_workflow_child(self):
        for_node = for_model.ForEachNode(
            inputs=["items"],
            outputs=["results"],
            body_node=makers.make_labeled_atomic(
                "body",
                inputs=["item"],
                outputs=["result"],
            ),
            input_edges={
                edge_models.TargetHandle(
                    node="body", port="item"
                ): edge_models.InputSource(port="items"),
            },
            output_edges={
                edge_models.OutputTarget(port="results"): edge_models.SourceHandle(
                    node="body", port="result"
                ),
            },
            nested_ports=["item"],
        )

        workflow = workflow_model.WorkflowNode(
            inputs=["data"],
            outputs=["processed"],
            nodes={"for_node": for_node},
            input_edges={
                edge_models.TargetHandle(
                    node="for_node", port="items"
                ): edge_models.InputSource(port="data"),
            },
            edges={},
            output_edges={
                edge_models.OutputTarget(port="processed"): edge_models.SourceHandle(
                    node="for_node", port="results"
                ),
            },
        )

        self.assertIsInstance(workflow.nodes["for_node"], for_model.ForEachNode)
        self.assertEqual(workflow.inputs, ["data"])
        self.assertEqual(workflow.outputs, ["processed"])

    def test_for_node_chained_in_workflow(self):
        for_node = for_model.ForEachNode(
            inputs=["items"],
            outputs=["results"],
            body_node=makers.make_labeled_atomic(
                "body", inputs=["item"], outputs=["result"], qualname="transform"
            ),
            input_edges={
                edge_models.TargetHandle(
                    node="body", port="item"
                ): edge_models.InputSource(port="items"),
            },
            output_edges={
                edge_models.OutputTarget(port="results"): edge_models.SourceHandle(
                    node="body", port="result"
                ),
            },
            nested_ports=["item"],
        )

        workflow = workflow_model.WorkflowNode(
            inputs=["raw_data"],
            outputs=["final"],
            nodes={
                "preprocess": makers.make_atomic(
                    inputs=["data"],
                    outputs=["items"],
                    qualname="preprocess",
                ),
                "for_node": for_node,
                "postprocess": makers.make_atomic(
                    inputs=["results"],
                    outputs=["output"],
                    qualname="postprocess",
                ),
            },
            input_edges={
                edge_models.TargetHandle(
                    node="preprocess", port="data"
                ): edge_models.InputSource(port="raw_data"),
            },
            edges={
                edge_models.TargetHandle(
                    node="for_node", port="items"
                ): edge_models.SourceHandle(node="preprocess", port="items"),
                edge_models.TargetHandle(
                    node="postprocess", port="results"
                ): edge_models.SourceHandle(node="for_node", port="results"),
            },
            output_edges={
                edge_models.OutputTarget(port="final"): edge_models.SourceHandle(
                    node="postprocess", port="output"
                ),
            },
        )

        self.assertEqual(len(workflow.nodes), 3)
        self.assertIsInstance(workflow.nodes["for_node"], for_model.ForEachNode)
        self.assertIsInstance(workflow.nodes["preprocess"], atomic_model.AtomicNode)
        self.assertIsInstance(workflow.nodes["postprocess"], atomic_model.AtomicNode)


class TestForEachNodeOutputProperties(unittest.TestCase):
    """Tests for outputs from input sources."""

    def _make_body(self, inputs, outputs) -> dict:
        return makers.make_labeled_atomic(
            "body", inputs=inputs, outputs=outputs
        ).model_dump(mode="json")

    def test_body_outputs_appear_in_neither(self):
        """SourceHandle outputs are neither pass-through nor transferred."""
        node = for_model.ForEachNode.model_validate(
            {
                "type": "for",
                "inputs": ["xs"],
                "outputs": ["results"],
                "body_node": self._make_body(["x"], ["y"]),
                "input_edges": {"body.x": "xs"},
                "output_edges": {"results": "body.y"},
                "nested_ports": ["x"],
            }
        )
        self.assertEqual(node.transferred_outputs, {})

    def test_transferred_from_nested_port(self):
        node = for_model.ForEachNode.model_validate(
            {
                "type": "for",
                "inputs": ["xs"],
                "outputs": ["results", "collected_xs"],
                "body_node": self._make_body(["x"], ["y"]),
                "input_edges": {"body.x": "xs"},
                "output_edges": {
                    "results": "body.y",
                    "collected_xs": "xs",
                },
                "nested_ports": ["x"],
            }
        )
        self.assertEqual(len(node.transferred_outputs), 1)
        target = edge_models.OutputTarget(port="collected_xs")
        self.assertIn(target, node.transferred_outputs)

    def test_transferred_from_zipped_port(self):
        node = for_model.ForEachNode.model_validate(
            {
                "type": "for",
                "inputs": ["xs", "ys"],
                "outputs": ["results", "fwd_xs", "fwd_ys"],
                "body_node": self._make_body(["x", "y"], ["z"]),
                "input_edges": {"body.x": "xs", "body.y": "ys"},
                "output_edges": {
                    "results": "body.z",
                    "fwd_xs": "xs",
                    "fwd_ys": "ys",
                },
                "zipped_ports": ["x", "y"],
            }
        )
        self.assertEqual(len(node.transferred_outputs), 2)

    def test_pass_through_from_broadcast_input_raises(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            for_model.ForEachNode.model_validate(
                {
                    "type": "for",
                    "inputs": ["c", "xs"],
                    "outputs": ["results", "fwd_c"],
                    "body_node": self._make_body(["x", "c"], ["y"]),
                    "input_edges": {"body.x": "xs", "body.c": "c"},
                    "output_edges": {
                        "results": "body.y",
                        "fwd_c": "c",
                    },
                    "nested_ports": ["x"],
                }
            )
        self.assertIn("input sources are only allowed if", str(ctx.exception))

    def test_no_input_source_outputs(self):
        """All outputs from body → both properties empty."""
        node = for_model.ForEachNode.model_validate(
            {
                "type": "for",
                "inputs": ["xs"],
                "outputs": ["a", "b"],
                "body_node": self._make_body(["x"], ["p", "q"]),
                "input_edges": {"body.x": "xs"},
                "output_edges": {"a": "body.p", "b": "body.q"},
                "nested_ports": ["x"],
            }
        )
        self.assertEqual(node.transferred_outputs, {})
