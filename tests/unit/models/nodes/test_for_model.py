import unittest

import pydantic

from flowrep.models import edges_model
from flowrep.models.nodes import (
    atomic_model,
    base_models,
    for_model,
    helper_models,
    union,
    workflow_model,
)


class TestForNodeBasic(unittest.TestCase):
    def test_valid_for_node_with_nested_ports(self):
        for_node = for_model.ForNode(
            inputs=["items"],
            outputs=["results"],
            body_node=helper_models.LabeledNode(
                label="body",
                node=atomic_model.AtomicNode(
                    fully_qualified_name="mod.func",
                    inputs=["item"],
                    outputs=["result"],
                ),
            ),
            input_edges={
                edges_model.TargetHandle(
                    node="body", port="item"
                ): edges_model.InputSource(port="items"),
            },
            output_edges={
                edges_model.OutputTarget(port="results"): edges_model.SourceHandle(
                    node="body", port="result"
                ),
            },
            nested_ports=["item"],
        )
        self.assertEqual(for_node.type, base_models.RecipeElementType.FOR)
        self.assertEqual(for_node.nested_ports, ["item"])
        self.assertEqual(for_node.zipped_ports, [])

    def test_valid_for_node_with_zipped_ports(self):
        for_node = for_model.ForNode(
            inputs=["xs", "ys"],
            outputs=["results"],
            body_node=helper_models.LabeledNode(
                label="body",
                node=atomic_model.AtomicNode(
                    fully_qualified_name="mod.func",
                    inputs=["x", "y"],
                    outputs=["result"],
                ),
            ),
            input_edges={
                edges_model.TargetHandle(
                    node="body", port="x"
                ): edges_model.InputSource(port="xs"),
                edges_model.TargetHandle(
                    node="body", port="y"
                ): edges_model.InputSource(port="ys"),
            },
            output_edges={
                edges_model.OutputTarget(port="results"): edges_model.SourceHandle(
                    node="body", port="result"
                ),
            },
            zipped_ports=["x", "y"],
        )
        self.assertEqual(for_node.zipped_ports, ["x", "y"])
        self.assertEqual(for_node.nested_ports, [])

    def test_valid_for_node_with_both_nested_and_zipped(self):
        for_node = for_model.ForNode(
            inputs=["outer", "inner1", "inner2"],
            outputs=["results"],
            body_node=helper_models.LabeledNode(
                label="body",
                node=atomic_model.AtomicNode(
                    fully_qualified_name="mod.func",
                    inputs=["a", "b", "c"],
                    outputs=["out"],
                ),
            ),
            input_edges={
                edges_model.TargetHandle(
                    node="body", port="a"
                ): edges_model.InputSource(port="outer"),
                edges_model.TargetHandle(
                    node="body", port="b"
                ): edges_model.InputSource(port="inner1"),
                edges_model.TargetHandle(
                    node="body", port="c"
                ): edges_model.InputSource(port="inner2"),
            },
            output_edges={
                edges_model.OutputTarget(port="results"): edges_model.SourceHandle(
                    node="body", port="out"
                ),
            },
            nested_ports=["a"],
            zipped_ports=["b", "c"],
        )
        self.assertEqual(for_node.nested_ports, ["a"])
        self.assertEqual(for_node.zipped_ports, ["b", "c"])


class TestForNodeLoopPortValidation(unittest.TestCase):
    def test_no_loop_ports_rejected(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            for_model.ForNode(
                inputs=["x"],
                outputs=["y"],
                body_node=helper_models.LabeledNode(
                    label="body",
                    node=atomic_model.AtomicNode(
                        fully_qualified_name="mod.func",
                        inputs=["inp"],
                        outputs=["out"],
                    ),
                ),
                input_edges={
                    edges_model.TargetHandle(
                        node="body", port="inp"
                    ): edges_model.InputSource(port="x"),
                },
                output_edges={
                    edges_model.OutputTarget(port="y"): edges_model.SourceHandle(
                        node="body", port="out"
                    ),
                },
                nested_ports=[],
                zipped_ports=[],
            )
        self.assertIn("at least one", str(ctx.exception).lower())

    def test_duplicate_nested_ports_rejected(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            for_model.ForNode(
                inputs=["items"],
                outputs=["results"],
                body_node=helper_models.LabeledNode(
                    label="body",
                    node=atomic_model.AtomicNode(
                        fully_qualified_name="mod.func",
                        inputs=["item"],
                        outputs=["result"],
                    ),
                ),
                input_edges={
                    edges_model.TargetHandle(
                        node="body", port="item"
                    ): edges_model.InputSource(port="items"),
                },
                output_edges={
                    edges_model.OutputTarget(port="results"): edges_model.SourceHandle(
                        node="body", port="result"
                    ),
                },
                nested_ports=["item", "item"],
            )
        self.assertIn("unique", str(ctx.exception).lower())

    def test_duplicate_zipped_ports_rejected(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            for_model.ForNode(
                inputs=["xs"],
                outputs=["results"],
                body_node=helper_models.LabeledNode(
                    label="body",
                    node=atomic_model.AtomicNode(
                        fully_qualified_name="mod.func",
                        inputs=["x"],
                        outputs=["result"],
                    ),
                ),
                input_edges={
                    edges_model.TargetHandle(
                        node="body", port="x"
                    ): edges_model.InputSource(port="xs"),
                },
                output_edges={
                    edges_model.OutputTarget(port="results"): edges_model.SourceHandle(
                        node="body", port="result"
                    ),
                },
                zipped_ports=["x", "x"],
            )
        self.assertIn("unique", str(ctx.exception).lower())

    def test_overlapping_nested_and_zipped_rejected(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            for_model.ForNode(
                inputs=["items"],
                outputs=["results"],
                body_node=helper_models.LabeledNode(
                    label="body",
                    node=atomic_model.AtomicNode(
                        fully_qualified_name="mod.func",
                        inputs=["item"],
                        outputs=["result"],
                    ),
                ),
                input_edges={
                    edges_model.TargetHandle(
                        node="body", port="item"
                    ): edges_model.InputSource(port="items"),
                },
                output_edges={
                    edges_model.OutputTarget(port="results"): edges_model.SourceHandle(
                        node="body", port="result"
                    ),
                },
                nested_ports=["item"],
                zipped_ports=["item"],
            )
        self.assertIn("overlap", str(ctx.exception).lower())
        self.assertIn("item", str(ctx.exception))

    def test_loop_port_not_on_body_node_rejected(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            for_model.ForNode(
                inputs=["items"],
                outputs=["results"],
                body_node=helper_models.LabeledNode(
                    label="body",
                    node=atomic_model.AtomicNode(
                        fully_qualified_name="mod.func",
                        inputs=["item"],
                        outputs=["result"],
                    ),
                ),
                input_edges={
                    edges_model.TargetHandle(
                        node="body", port="item"
                    ): edges_model.InputSource(port="items"),
                },
                output_edges={
                    edges_model.OutputTarget(port="results"): edges_model.SourceHandle(
                        node="body", port="result"
                    ),
                },
                nested_ports=["nonexistent"],
            )
        self.assertIn("nonexistent", str(ctx.exception))


class TestForNodeInputEdges(unittest.TestCase):
    def test_input_edge_wrong_target_node_rejected(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            for_model.ForNode(
                inputs=["items"],
                outputs=["results"],
                body_node=helper_models.LabeledNode(
                    label="body",
                    node=atomic_model.AtomicNode(
                        fully_qualified_name="mod.func",
                        inputs=["item"],
                        outputs=["result"],
                    ),
                ),
                input_edges={
                    edges_model.TargetHandle(
                        node="wrong_node", port="item"
                    ): edges_model.InputSource(port="items"),
                },
                output_edges={
                    edges_model.OutputTarget(port="results"): edges_model.SourceHandle(
                        node="body", port="result"
                    ),
                },
                nested_ports=["item"],
            )
        self.assertIn("body", str(ctx.exception))

    def test_input_edge_wrong_target_port_rejected(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            for_model.ForNode(
                inputs=["items"],
                outputs=["results"],
                body_node=helper_models.LabeledNode(
                    label="body",
                    node=atomic_model.AtomicNode(
                        fully_qualified_name="mod.func",
                        inputs=["item"],
                        outputs=["result"],
                    ),
                ),
                input_edges={
                    edges_model.TargetHandle(
                        node="body", port="wrong_port"
                    ): edges_model.InputSource(port="items"),
                },
                output_edges={
                    edges_model.OutputTarget(port="results"): edges_model.SourceHandle(
                        node="body", port="result"
                    ),
                },
                nested_ports=["item"],
            )
        self.assertIn("wrong_port", str(ctx.exception))

    def test_input_edge_invalid_source_port_rejected(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            for_model.ForNode(
                inputs=["items"],
                outputs=["results"],
                body_node=helper_models.LabeledNode(
                    label="body",
                    node=atomic_model.AtomicNode(
                        fully_qualified_name="mod.func",
                        inputs=["item"],
                        outputs=["result"],
                    ),
                ),
                input_edges={
                    edges_model.TargetHandle(
                        node="body", port="item"
                    ): edges_model.InputSource(port="nonexistent"),
                },
                output_edges={
                    edges_model.OutputTarget(port="results"): edges_model.SourceHandle(
                        node="body", port="result"
                    ),
                },
                nested_ports=["item"],
            )
        self.assertIn("nonexistent", str(ctx.exception))


class TestForNodeOutputEdges(unittest.TestCase):
    def test_output_edge_wrong_source_node_rejected(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            for_model.ForNode(
                inputs=["items"],
                outputs=["results"],
                body_node=helper_models.LabeledNode(
                    label="body",
                    node=atomic_model.AtomicNode(
                        fully_qualified_name="mod.func",
                        inputs=["item"],
                        outputs=["result"],
                    ),
                ),
                input_edges={
                    edges_model.TargetHandle(
                        node="body", port="item"
                    ): edges_model.InputSource(port="items"),
                },
                output_edges={
                    edges_model.OutputTarget(port="results"): edges_model.SourceHandle(
                        node="wrong_node", port="result"
                    ),
                },
                nested_ports=["item"],
            )
        self.assertIn("body", str(ctx.exception))

    def test_output_edge_wrong_source_port_rejected(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            for_model.ForNode(
                inputs=["items"],
                outputs=["results"],
                body_node=helper_models.LabeledNode(
                    label="body",
                    node=atomic_model.AtomicNode(
                        fully_qualified_name="mod.func",
                        inputs=["item"],
                        outputs=["result"],
                    ),
                ),
                input_edges={
                    edges_model.TargetHandle(
                        node="body", port="item"
                    ): edges_model.InputSource(port="items"),
                },
                output_edges={
                    edges_model.OutputTarget(port="results"): edges_model.SourceHandle(
                        node="body", port="wrong_port"
                    ),
                },
                nested_ports=["item"],
            )
        self.assertIn("wrong_port", str(ctx.exception))

    def test_output_edge_invalid_target_port_rejected(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            for_model.ForNode(
                inputs=["items"],
                outputs=["results"],
                body_node=helper_models.LabeledNode(
                    label="body",
                    node=atomic_model.AtomicNode(
                        fully_qualified_name="mod.func",
                        inputs=["item"],
                        outputs=["result"],
                    ),
                ),
                input_edges={
                    edges_model.TargetHandle(
                        node="body", port="item"
                    ): edges_model.InputSource(port="items"),
                },
                output_edges={
                    edges_model.OutputTarget(
                        port="nonexistent"
                    ): edges_model.SourceHandle(node="body", port="result"),
                },
                nested_ports=["item"],
            )
        self.assertIn("nonexistent", str(ctx.exception))


class TestForNodeTransferEdges(unittest.TestCase):
    def test_valid_transfer_edge(self):
        """Transfer edges should forward looped inputs to outputs."""
        for_node = for_model.ForNode(
            inputs=["items"],
            outputs=["results", "original_items"],
            body_node=helper_models.LabeledNode(
                label="body",
                node=atomic_model.AtomicNode(
                    fully_qualified_name="mod.func",
                    inputs=["item"],
                    outputs=["result"],
                ),
            ),
            input_edges={
                edges_model.TargetHandle(
                    node="body", port="item"
                ): edges_model.InputSource(port="items"),
            },
            output_edges={
                edges_model.OutputTarget(port="results"): edges_model.SourceHandle(
                    node="body", port="result"
                ),
            },
            nested_ports=["item"],
            transfer_edges={
                edges_model.OutputTarget(
                    port="original_items"
                ): edges_model.InputSource(port="items"),
            },
        )
        self.assertEqual(len(for_node.transfer_edges), 1)

    def test_transfer_source_not_in_inputs_rejected(self):
        """Transfer edge sources must be ForNode inputs."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            for_model.ForNode(
                inputs=["items"],
                outputs=["results", "forwarded"],
                body_node=helper_models.LabeledNode(
                    label="body",
                    node=atomic_model.AtomicNode(
                        fully_qualified_name="mod.func",
                        inputs=["item"],
                        outputs=["result"],
                    ),
                ),
                input_edges={
                    edges_model.TargetHandle(
                        node="body", port="item"
                    ): edges_model.InputSource(port="items"),
                },
                output_edges={
                    edges_model.OutputTarget(port="results"): edges_model.SourceHandle(
                        node="body", port="result"
                    ),
                },
                nested_ports=["item"],
                transfer_edges={
                    edges_model.OutputTarget(port="forwarded"): edges_model.InputSource(
                        port="nonexistent"
                    ),
                },
            )
        self.assertIn("nonexistent", str(ctx.exception))
        self.assertIn("inputs", str(ctx.exception).lower())

    def test_transfer_target_not_in_outputs_rejected(self):
        """Transfer edge targets must be ForNode outputs."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            for_model.ForNode(
                inputs=["items"],
                outputs=["results"],
                body_node=helper_models.LabeledNode(
                    label="body",
                    node=atomic_model.AtomicNode(
                        fully_qualified_name="mod.func",
                        inputs=["item"],
                        outputs=["result"],
                    ),
                ),
                input_edges={
                    edges_model.TargetHandle(
                        node="body", port="item"
                    ): edges_model.InputSource(port="items"),
                },
                output_edges={
                    edges_model.OutputTarget(port="results"): edges_model.SourceHandle(
                        node="body", port="result"
                    ),
                },
                nested_ports=["item"],
                transfer_edges={
                    edges_model.OutputTarget(
                        port="nonexistent"
                    ): edges_model.InputSource(port="items"),
                },
            )
        self.assertIn("nonexistent", str(ctx.exception))
        self.assertIn("outputs", str(ctx.exception).lower())

    def test_transfer_source_not_looped_rejected(self):
        """Transfer edge sources must be looped (in nested_ports or zipped_ports)."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            for_model.ForNode(
                inputs=["items", "static_value"],
                outputs=["results", "forwarded"],
                body_node=helper_models.LabeledNode(
                    label="body",
                    node=atomic_model.AtomicNode(
                        fully_qualified_name="mod.func",
                        inputs=["item", "static"],
                        outputs=["result"],
                    ),
                ),
                input_edges={
                    edges_model.TargetHandle(
                        node="body", port="item"
                    ): edges_model.InputSource(port="items"),
                    edges_model.TargetHandle(
                        node="body", port="static"
                    ): edges_model.InputSource(port="static_value"),
                },
                output_edges={
                    edges_model.OutputTarget(port="results"): edges_model.SourceHandle(
                        node="body", port="result"
                    ),
                },
                nested_ports=["item"],
                transfer_edges={
                    edges_model.OutputTarget(port="forwarded"): edges_model.InputSource(
                        port="static_value"
                    ),
                },
            )
        self.assertIn("static_value", str(ctx.exception))
        self.assertIn("looped", str(ctx.exception).lower())

    def test_transfer_target_collides_with_output_edge_rejected(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            for_model.ForNode(
                inputs=["items"],
                outputs=["results"],
                body_node=helper_models.LabeledNode(
                    label="body",
                    node=atomic_model.AtomicNode(
                        fully_qualified_name="mod.func",
                        inputs=["item"],
                        outputs=["result"],
                    ),
                ),
                input_edges={
                    edges_model.TargetHandle(
                        node="body", port="item"
                    ): edges_model.InputSource(port="items"),
                },
                output_edges={
                    edges_model.OutputTarget(port="results"): edges_model.SourceHandle(
                        node="body", port="result"
                    ),
                },
                nested_ports=["item"],
                transfer_edges={
                    edges_model.OutputTarget(port="results"): edges_model.InputSource(
                        port="items"
                    ),
                },
            )
        self.assertIn("results", str(ctx.exception))
        self.assertIn("conflict", str(ctx.exception).lower())


class TestForNodeSerialization(unittest.TestCase):
    def test_roundtrip(self):
        original = for_model.ForNode(
            inputs=["items", "multiplier"],
            outputs=["results", "original_items"],
            body_node=helper_models.LabeledNode(
                label="body",
                node=atomic_model.AtomicNode(
                    fully_qualified_name="mod.func",
                    inputs=["item", "mult"],
                    outputs=["result"],
                ),
            ),
            input_edges={
                edges_model.TargetHandle(
                    node="body", port="item"
                ): edges_model.InputSource(port="items"),
                edges_model.TargetHandle(
                    node="body", port="mult"
                ): edges_model.InputSource(port="multiplier"),
            },
            output_edges={
                edges_model.OutputTarget(port="results"): edges_model.SourceHandle(
                    node="body", port="result"
                ),
            },
            nested_ports=["item"],
            transfer_edges={
                edges_model.OutputTarget(
                    port="original_items"
                ): edges_model.InputSource(port="items"),
            },
        )
        for mode in ["python", "json"]:
            with self.subTest(mode=mode):
                data = original.model_dump(mode=mode)
                restored = for_model.ForNode.model_validate(data)

                self.assertEqual(original.inputs, restored.inputs)
                self.assertEqual(original.outputs, restored.outputs)
                self.assertEqual(original.nested_ports, restored.nested_ports)
                self.assertEqual(original.zipped_ports, restored.zipped_ports)
                self.assertEqual(original.body_node.label, restored.body_node.label)
                self.assertEqual(original.input_edges, restored.input_edges)
                self.assertEqual(original.output_edges, restored.output_edges)
                self.assertEqual(original.transfer_edges, restored.transfer_edges)

    def test_discriminated_union_roundtrip(self):
        """ForNode should be correctly identified via type discriminator."""
        data = {
            "type": base_models.RecipeElementType.FOR,
            "inputs": ["items"],
            "outputs": ["results"],
            "body_node": {
                "label": "body",
                "node": {
                    "type": "atomic",
                    "fully_qualified_name": "mod.func",
                    "inputs": ["item"],
                    "outputs": ["result"],
                },
            },
            "input_edges": {"body.item": "items"},
            "output_edges": {"results": "body.result"},
            "nested_ports": ["item"],
            "zipped_ports": [],
            "transfer_edges": {},
        }
        node = pydantic.TypeAdapter(union.NodeType).validate_python(data)
        self.assertIsInstance(node, for_model.ForNode)


class TestForNodeComposition(unittest.TestCase):
    def test_workflow_body_node(self):
        inner_workflow = workflow_model.WorkflowNode(
            inputs=["x"],
            outputs=["y"],
            nodes={
                "leaf": atomic_model.AtomicNode(
                    fully_qualified_name="mod.func",
                    inputs=["inp"],
                    outputs=["out"],
                )
            },
            input_edges={
                edges_model.TargetHandle(
                    node="leaf", port="inp"
                ): edges_model.InputSource(port="x"),
            },
            edges={},
            output_edges={
                edges_model.OutputTarget(port="y"): edges_model.SourceHandle(
                    node="leaf", port="out"
                ),
            },
        )

        for_node = for_model.ForNode(
            inputs=["items"],
            outputs=["results"],
            body_node=helper_models.LabeledNode(
                label="body",
                node=inner_workflow,
            ),
            input_edges={
                edges_model.TargetHandle(
                    node="body", port="x"
                ): edges_model.InputSource(port="items"),
            },
            output_edges={
                edges_model.OutputTarget(port="results"): edges_model.SourceHandle(
                    node="body", port="y"
                ),
            },
            nested_ports=["x"],
        )
        self.assertIsInstance(for_node.body_node.node, workflow_model.WorkflowNode)

    def test_for_node_as_workflow_child(self):
        for_node = for_model.ForNode(
            inputs=["items"],
            outputs=["results"],
            body_node=helper_models.LabeledNode(
                label="body",
                node=atomic_model.AtomicNode(
                    fully_qualified_name="mod.func",
                    inputs=["item"],
                    outputs=["result"],
                ),
            ),
            input_edges={
                edges_model.TargetHandle(
                    node="body", port="item"
                ): edges_model.InputSource(port="items"),
            },
            output_edges={
                edges_model.OutputTarget(port="results"): edges_model.SourceHandle(
                    node="body", port="result"
                ),
            },
            nested_ports=["item"],
        )

        workflow = workflow_model.WorkflowNode(
            inputs=["data"],
            outputs=["processed"],
            nodes={"loop": for_node},
            input_edges={
                edges_model.TargetHandle(
                    node="loop", port="items"
                ): edges_model.InputSource(port="data"),
            },
            edges={},
            output_edges={
                edges_model.OutputTarget(port="processed"): edges_model.SourceHandle(
                    node="loop", port="results"
                ),
            },
        )

        self.assertIsInstance(workflow.nodes["loop"], for_model.ForNode)
        self.assertEqual(workflow.inputs, ["data"])
        self.assertEqual(workflow.outputs, ["processed"])

    def test_for_node_chained_in_workflow(self):
        for_node = for_model.ForNode(
            inputs=["items"],
            outputs=["results"],
            body_node=helper_models.LabeledNode(
                label="body",
                node=atomic_model.AtomicNode(
                    fully_qualified_name="mod.transform",
                    inputs=["item"],
                    outputs=["result"],
                ),
            ),
            input_edges={
                edges_model.TargetHandle(
                    node="body", port="item"
                ): edges_model.InputSource(port="items"),
            },
            output_edges={
                edges_model.OutputTarget(port="results"): edges_model.SourceHandle(
                    node="body", port="result"
                ),
            },
            nested_ports=["item"],
        )

        workflow = workflow_model.WorkflowNode(
            inputs=["raw_data"],
            outputs=["final"],
            nodes={
                "preprocess": atomic_model.AtomicNode(
                    fully_qualified_name="mod.preprocess",
                    inputs=["data"],
                    outputs=["items"],
                ),
                "loop": for_node,
                "postprocess": atomic_model.AtomicNode(
                    fully_qualified_name="mod.postprocess",
                    inputs=["results"],
                    outputs=["output"],
                ),
            },
            input_edges={
                edges_model.TargetHandle(
                    node="preprocess", port="data"
                ): edges_model.InputSource(port="raw_data"),
            },
            edges={
                edges_model.TargetHandle(
                    node="loop", port="items"
                ): edges_model.SourceHandle(node="preprocess", port="items"),
                edges_model.TargetHandle(
                    node="postprocess", port="results"
                ): edges_model.SourceHandle(node="loop", port="results"),
            },
            output_edges={
                edges_model.OutputTarget(port="final"): edges_model.SourceHandle(
                    node="postprocess", port="output"
                ),
            },
        )

        self.assertEqual(len(workflow.nodes), 3)
        self.assertIsInstance(workflow.nodes["loop"], for_model.ForNode)
        self.assertIsInstance(workflow.nodes["preprocess"], atomic_model.AtomicNode)
        self.assertIsInstance(workflow.nodes["postprocess"], atomic_model.AtomicNode)
