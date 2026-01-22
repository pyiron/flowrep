import unittest

import pydantic

from flowrep.models import model


class TestForNodeBasic(unittest.TestCase):
    def test_valid_for_node_with_nested_ports(self):
        for_node = model.ForNode(
            inputs=["items"],
            outputs=["results"],
            body_node=model.LabeledNode(
                label="body",
                node=model.AtomicNode(
                    fully_qualified_name="mod.func",
                    inputs=["item"],
                    outputs=["result"],
                ),
            ),
            input_edges={
                model.TargetHandle(node="body", port="item"): model.InputSource(
                    port="items"
                ),
            },
            output_edges={
                model.OutputTarget(port="results"): model.SourceHandle(
                    node="body", port="result"
                ),
            },
            nested_ports=["item"],
        )
        self.assertEqual(for_node.type, model.RecipeElementType.FOR)
        self.assertEqual(for_node.nested_ports, ["item"])
        self.assertEqual(for_node.zipped_ports, [])

    def test_valid_for_node_with_zipped_ports(self):
        for_node = model.ForNode(
            inputs=["xs", "ys"],
            outputs=["results"],
            body_node=model.LabeledNode(
                label="body",
                node=model.AtomicNode(
                    fully_qualified_name="mod.func",
                    inputs=["x", "y"],
                    outputs=["result"],
                ),
            ),
            input_edges={
                model.TargetHandle(node="body", port="x"): model.InputSource(port="xs"),
                model.TargetHandle(node="body", port="y"): model.InputSource(port="ys"),
            },
            output_edges={
                model.OutputTarget(port="results"): model.SourceHandle(
                    node="body", port="result"
                ),
            },
            zipped_ports=["x", "y"],
        )
        self.assertEqual(for_node.zipped_ports, ["x", "y"])
        self.assertEqual(for_node.nested_ports, [])

    def test_valid_for_node_with_both_nested_and_zipped(self):
        for_node = model.ForNode(
            inputs=["outer", "inner1", "inner2"],
            outputs=["results"],
            body_node=model.LabeledNode(
                label="body",
                node=model.AtomicNode(
                    fully_qualified_name="mod.func",
                    inputs=["a", "b", "c"],
                    outputs=["out"],
                ),
            ),
            input_edges={
                model.TargetHandle(node="body", port="a"): model.InputSource(
                    port="outer"
                ),
                model.TargetHandle(node="body", port="b"): model.InputSource(
                    port="inner1"
                ),
                model.TargetHandle(node="body", port="c"): model.InputSource(
                    port="inner2"
                ),
            },
            output_edges={
                model.OutputTarget(port="results"): model.SourceHandle(
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
            model.ForNode(
                inputs=["x"],
                outputs=["y"],
                body_node=model.LabeledNode(
                    label="body",
                    node=model.AtomicNode(
                        fully_qualified_name="mod.func",
                        inputs=["inp"],
                        outputs=["out"],
                    ),
                ),
                input_edges={
                    model.TargetHandle(node="body", port="inp"): model.InputSource(
                        port="x"
                    ),
                },
                output_edges={
                    model.OutputTarget(port="y"): model.SourceHandle(
                        node="body", port="out"
                    ),
                },
                nested_ports=[],
                zipped_ports=[],
            )
        self.assertIn("at least one", str(ctx.exception).lower())

    def test_duplicate_nested_ports_rejected(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.ForNode(
                inputs=["items"],
                outputs=["results"],
                body_node=model.LabeledNode(
                    label="body",
                    node=model.AtomicNode(
                        fully_qualified_name="mod.func",
                        inputs=["item"],
                        outputs=["result"],
                    ),
                ),
                input_edges={
                    model.TargetHandle(node="body", port="item"): model.InputSource(
                        port="items"
                    ),
                },
                output_edges={
                    model.OutputTarget(port="results"): model.SourceHandle(
                        node="body", port="result"
                    ),
                },
                nested_ports=["item", "item"],
            )
        self.assertIn("unique", str(ctx.exception).lower())

    def test_duplicate_zipped_ports_rejected(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.ForNode(
                inputs=["xs"],
                outputs=["results"],
                body_node=model.LabeledNode(
                    label="body",
                    node=model.AtomicNode(
                        fully_qualified_name="mod.func",
                        inputs=["x"],
                        outputs=["result"],
                    ),
                ),
                input_edges={
                    model.TargetHandle(node="body", port="x"): model.InputSource(
                        port="xs"
                    ),
                },
                output_edges={
                    model.OutputTarget(port="results"): model.SourceHandle(
                        node="body", port="result"
                    ),
                },
                zipped_ports=["x", "x"],
            )
        self.assertIn("unique", str(ctx.exception).lower())

    def test_overlapping_nested_and_zipped_rejected(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.ForNode(
                inputs=["items"],
                outputs=["results"],
                body_node=model.LabeledNode(
                    label="body",
                    node=model.AtomicNode(
                        fully_qualified_name="mod.func",
                        inputs=["item"],
                        outputs=["result"],
                    ),
                ),
                input_edges={
                    model.TargetHandle(node="body", port="item"): model.InputSource(
                        port="items"
                    ),
                },
                output_edges={
                    model.OutputTarget(port="results"): model.SourceHandle(
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
            model.ForNode(
                inputs=["items"],
                outputs=["results"],
                body_node=model.LabeledNode(
                    label="body",
                    node=model.AtomicNode(
                        fully_qualified_name="mod.func",
                        inputs=["item"],
                        outputs=["result"],
                    ),
                ),
                input_edges={
                    model.TargetHandle(node="body", port="item"): model.InputSource(
                        port="items"
                    ),
                },
                output_edges={
                    model.OutputTarget(port="results"): model.SourceHandle(
                        node="body", port="result"
                    ),
                },
                nested_ports=["nonexistent"],
            )
        self.assertIn("nonexistent", str(ctx.exception))


class TestForNodeInputEdges(unittest.TestCase):
    def test_input_edge_wrong_target_node_rejected(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.ForNode(
                inputs=["items"],
                outputs=["results"],
                body_node=model.LabeledNode(
                    label="body",
                    node=model.AtomicNode(
                        fully_qualified_name="mod.func",
                        inputs=["item"],
                        outputs=["result"],
                    ),
                ),
                input_edges={
                    model.TargetHandle(
                        node="wrong_node", port="item"
                    ): model.InputSource(port="items"),
                },
                output_edges={
                    model.OutputTarget(port="results"): model.SourceHandle(
                        node="body", port="result"
                    ),
                },
                nested_ports=["item"],
            )
        self.assertIn("body", str(ctx.exception))

    def test_input_edge_wrong_target_port_rejected(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.ForNode(
                inputs=["items"],
                outputs=["results"],
                body_node=model.LabeledNode(
                    label="body",
                    node=model.AtomicNode(
                        fully_qualified_name="mod.func",
                        inputs=["item"],
                        outputs=["result"],
                    ),
                ),
                input_edges={
                    model.TargetHandle(
                        node="body", port="wrong_port"
                    ): model.InputSource(port="items"),
                },
                output_edges={
                    model.OutputTarget(port="results"): model.SourceHandle(
                        node="body", port="result"
                    ),
                },
                nested_ports=["item"],
            )
        self.assertIn("wrong_port", str(ctx.exception))

    def test_input_edge_invalid_source_port_rejected(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.ForNode(
                inputs=["items"],
                outputs=["results"],
                body_node=model.LabeledNode(
                    label="body",
                    node=model.AtomicNode(
                        fully_qualified_name="mod.func",
                        inputs=["item"],
                        outputs=["result"],
                    ),
                ),
                input_edges={
                    model.TargetHandle(node="body", port="item"): model.InputSource(
                        port="nonexistent"
                    ),
                },
                output_edges={
                    model.OutputTarget(port="results"): model.SourceHandle(
                        node="body", port="result"
                    ),
                },
                nested_ports=["item"],
            )
        self.assertIn("nonexistent", str(ctx.exception))


class TestForNodeOutputEdges(unittest.TestCase):
    def test_output_edge_wrong_source_node_rejected(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.ForNode(
                inputs=["items"],
                outputs=["results"],
                body_node=model.LabeledNode(
                    label="body",
                    node=model.AtomicNode(
                        fully_qualified_name="mod.func",
                        inputs=["item"],
                        outputs=["result"],
                    ),
                ),
                input_edges={
                    model.TargetHandle(node="body", port="item"): model.InputSource(
                        port="items"
                    ),
                },
                output_edges={
                    model.OutputTarget(port="results"): model.SourceHandle(
                        node="wrong_node", port="result"
                    ),
                },
                nested_ports=["item"],
            )
        self.assertIn("body", str(ctx.exception))

    def test_output_edge_wrong_source_port_rejected(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.ForNode(
                inputs=["items"],
                outputs=["results"],
                body_node=model.LabeledNode(
                    label="body",
                    node=model.AtomicNode(
                        fully_qualified_name="mod.func",
                        inputs=["item"],
                        outputs=["result"],
                    ),
                ),
                input_edges={
                    model.TargetHandle(node="body", port="item"): model.InputSource(
                        port="items"
                    ),
                },
                output_edges={
                    model.OutputTarget(port="results"): model.SourceHandle(
                        node="body", port="wrong_port"
                    ),
                },
                nested_ports=["item"],
            )
        self.assertIn("wrong_port", str(ctx.exception))

    def test_output_edge_invalid_target_port_rejected(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.ForNode(
                inputs=["items"],
                outputs=["results"],
                body_node=model.LabeledNode(
                    label="body",
                    node=model.AtomicNode(
                        fully_qualified_name="mod.func",
                        inputs=["item"],
                        outputs=["result"],
                    ),
                ),
                input_edges={
                    model.TargetHandle(node="body", port="item"): model.InputSource(
                        port="items"
                    ),
                },
                output_edges={
                    model.OutputTarget(port="nonexistent"): model.SourceHandle(
                        node="body", port="result"
                    ),
                },
                nested_ports=["item"],
            )
        self.assertIn("nonexistent", str(ctx.exception))


class TestForNodeTransferEdges(unittest.TestCase):
    def test_valid_transfer_edge(self):
        """Transfer edges should forward looped inputs to outputs."""
        for_node = model.ForNode(
            inputs=["items"],
            outputs=["results", "original_items"],
            body_node=model.LabeledNode(
                label="body",
                node=model.AtomicNode(
                    fully_qualified_name="mod.func",
                    inputs=["item"],
                    outputs=["result"],
                ),
            ),
            input_edges={
                model.TargetHandle(node="body", port="item"): model.InputSource(
                    port="items"
                ),
            },
            output_edges={
                model.OutputTarget(port="results"): model.SourceHandle(
                    node="body", port="result"
                ),
            },
            nested_ports=["item"],
            transfer_edges={
                model.OutputTarget(port="original_items"): model.InputSource(
                    port="items"
                ),
            },
        )
        self.assertEqual(len(for_node.transfer_edges), 1)

    def test_transfer_source_not_in_inputs_rejected(self):
        """Transfer edge sources must be ForNode inputs."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.ForNode(
                inputs=["items"],
                outputs=["results", "forwarded"],
                body_node=model.LabeledNode(
                    label="body",
                    node=model.AtomicNode(
                        fully_qualified_name="mod.func",
                        inputs=["item"],
                        outputs=["result"],
                    ),
                ),
                input_edges={
                    model.TargetHandle(node="body", port="item"): model.InputSource(
                        port="items"
                    ),
                },
                output_edges={
                    model.OutputTarget(port="results"): model.SourceHandle(
                        node="body", port="result"
                    ),
                },
                nested_ports=["item"],
                transfer_edges={
                    model.OutputTarget(port="forwarded"): model.InputSource(
                        port="nonexistent"
                    ),
                },
            )
        self.assertIn("nonexistent", str(ctx.exception))
        self.assertIn("inputs", str(ctx.exception).lower())

    def test_transfer_target_not_in_outputs_rejected(self):
        """Transfer edge targets must be ForNode outputs."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.ForNode(
                inputs=["items"],
                outputs=["results"],
                body_node=model.LabeledNode(
                    label="body",
                    node=model.AtomicNode(
                        fully_qualified_name="mod.func",
                        inputs=["item"],
                        outputs=["result"],
                    ),
                ),
                input_edges={
                    model.TargetHandle(node="body", port="item"): model.InputSource(
                        port="items"
                    ),
                },
                output_edges={
                    model.OutputTarget(port="results"): model.SourceHandle(
                        node="body", port="result"
                    ),
                },
                nested_ports=["item"],
                transfer_edges={
                    model.OutputTarget(port="nonexistent"): model.InputSource(
                        port="items"
                    ),
                },
            )
        self.assertIn("nonexistent", str(ctx.exception))
        self.assertIn("outputs", str(ctx.exception).lower())

    def test_transfer_source_not_looped_rejected(self):
        """Transfer edge sources must be looped (in nested_ports or zipped_ports)."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.ForNode(
                inputs=["items", "static_value"],
                outputs=["results", "forwarded"],
                body_node=model.LabeledNode(
                    label="body",
                    node=model.AtomicNode(
                        fully_qualified_name="mod.func",
                        inputs=["item", "static"],
                        outputs=["result"],
                    ),
                ),
                input_edges={
                    model.TargetHandle(node="body", port="item"): model.InputSource(
                        port="items"
                    ),
                    model.TargetHandle(node="body", port="static"): model.InputSource(
                        port="static_value"
                    ),
                },
                output_edges={
                    model.OutputTarget(port="results"): model.SourceHandle(
                        node="body", port="result"
                    ),
                },
                nested_ports=["item"],
                transfer_edges={
                    model.OutputTarget(port="forwarded"): model.InputSource(
                        port="static_value"
                    ),
                },
            )
        self.assertIn("static_value", str(ctx.exception))
        self.assertIn("looped", str(ctx.exception).lower())

    def test_transfer_target_collides_with_output_edge_rejected(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.ForNode(
                inputs=["items"],
                outputs=["results"],
                body_node=model.LabeledNode(
                    label="body",
                    node=model.AtomicNode(
                        fully_qualified_name="mod.func",
                        inputs=["item"],
                        outputs=["result"],
                    ),
                ),
                input_edges={
                    model.TargetHandle(node="body", port="item"): model.InputSource(
                        port="items"
                    ),
                },
                output_edges={
                    model.OutputTarget(port="results"): model.SourceHandle(
                        node="body", port="result"
                    ),
                },
                nested_ports=["item"],
                transfer_edges={
                    model.OutputTarget(port="results"): model.InputSource(port="items"),
                },
            )
        self.assertIn("results", str(ctx.exception))
        self.assertIn("conflict", str(ctx.exception).lower())


class TestForNodeSerialization(unittest.TestCase):
    def test_roundtrip(self):
        original = model.ForNode(
            inputs=["items", "multiplier"],
            outputs=["results", "original_items"],
            body_node=model.LabeledNode(
                label="body",
                node=model.AtomicNode(
                    fully_qualified_name="mod.func",
                    inputs=["item", "mult"],
                    outputs=["result"],
                ),
            ),
            input_edges={
                model.TargetHandle(node="body", port="item"): model.InputSource(
                    port="items"
                ),
                model.TargetHandle(node="body", port="mult"): model.InputSource(
                    port="multiplier"
                ),
            },
            output_edges={
                model.OutputTarget(port="results"): model.SourceHandle(
                    node="body", port="result"
                ),
            },
            nested_ports=["item"],
            transfer_edges={
                model.OutputTarget(port="original_items"): model.InputSource(
                    port="items"
                ),
            },
        )
        for mode in ["python", "json"]:
            with self.subTest(mode=mode):
                data = original.model_dump(mode=mode)
                restored = model.ForNode.model_validate(data)

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
            "type": model.RecipeElementType.FOR,
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
        node = pydantic.TypeAdapter(model.NodeType).validate_python(data)
        self.assertIsInstance(node, model.ForNode)


class TestForNodeComposition(unittest.TestCase):
    def test_workflow_body_node(self):
        inner_workflow = model.WorkflowNode(
            inputs=["x"],
            outputs=["y"],
            nodes={
                "leaf": model.AtomicNode(
                    fully_qualified_name="mod.func",
                    inputs=["inp"],
                    outputs=["out"],
                )
            },
            input_edges={
                model.TargetHandle(node="leaf", port="inp"): model.InputSource(
                    port="x"
                ),
            },
            edges={},
            output_edges={
                model.OutputTarget(port="y"): model.SourceHandle(
                    node="leaf", port="out"
                ),
            },
        )

        for_node = model.ForNode(
            inputs=["items"],
            outputs=["results"],
            body_node=model.LabeledNode(
                label="body",
                node=inner_workflow,
            ),
            input_edges={
                model.TargetHandle(node="body", port="x"): model.InputSource(
                    port="items"
                ),
            },
            output_edges={
                model.OutputTarget(port="results"): model.SourceHandle(
                    node="body", port="y"
                ),
            },
            nested_ports=["x"],
        )
        self.assertIsInstance(for_node.body_node.node, model.WorkflowNode)

    def test_for_node_as_workflow_child(self):
        for_node = model.ForNode(
            inputs=["items"],
            outputs=["results"],
            body_node=model.LabeledNode(
                label="body",
                node=model.AtomicNode(
                    fully_qualified_name="mod.func",
                    inputs=["item"],
                    outputs=["result"],
                ),
            ),
            input_edges={
                model.TargetHandle(node="body", port="item"): model.InputSource(
                    port="items"
                ),
            },
            output_edges={
                model.OutputTarget(port="results"): model.SourceHandle(
                    node="body", port="result"
                ),
            },
            nested_ports=["item"],
        )

        workflow = model.WorkflowNode(
            inputs=["data"],
            outputs=["processed"],
            nodes={"loop": for_node},
            input_edges={
                model.TargetHandle(node="loop", port="items"): model.InputSource(
                    port="data"
                ),
            },
            edges={},
            output_edges={
                model.OutputTarget(port="processed"): model.SourceHandle(
                    node="loop", port="results"
                ),
            },
        )

        self.assertIsInstance(workflow.nodes["loop"], model.ForNode)
        self.assertEqual(workflow.inputs, ["data"])
        self.assertEqual(workflow.outputs, ["processed"])

    def test_for_node_chained_in_workflow(self):
        for_node = model.ForNode(
            inputs=["items"],
            outputs=["results"],
            body_node=model.LabeledNode(
                label="body",
                node=model.AtomicNode(
                    fully_qualified_name="mod.transform",
                    inputs=["item"],
                    outputs=["result"],
                ),
            ),
            input_edges={
                model.TargetHandle(node="body", port="item"): model.InputSource(
                    port="items"
                ),
            },
            output_edges={
                model.OutputTarget(port="results"): model.SourceHandle(
                    node="body", port="result"
                ),
            },
            nested_ports=["item"],
        )

        workflow = model.WorkflowNode(
            inputs=["raw_data"],
            outputs=["final"],
            nodes={
                "preprocess": model.AtomicNode(
                    fully_qualified_name="mod.preprocess",
                    inputs=["data"],
                    outputs=["items"],
                ),
                "loop": for_node,
                "postprocess": model.AtomicNode(
                    fully_qualified_name="mod.postprocess",
                    inputs=["results"],
                    outputs=["output"],
                ),
            },
            input_edges={
                model.TargetHandle(node="preprocess", port="data"): model.InputSource(
                    port="raw_data"
                ),
            },
            edges={
                model.TargetHandle(node="loop", port="items"): model.SourceHandle(
                    node="preprocess", port="items"
                ),
                model.TargetHandle(
                    node="postprocess", port="results"
                ): model.SourceHandle(node="loop", port="results"),
            },
            output_edges={
                model.OutputTarget(port="final"): model.SourceHandle(
                    node="postprocess", port="output"
                ),
            },
        )

        self.assertEqual(len(workflow.nodes), 3)
        self.assertIsInstance(workflow.nodes["loop"], model.ForNode)
        self.assertIsInstance(workflow.nodes["preprocess"], model.AtomicNode)
        self.assertIsInstance(workflow.nodes["postprocess"], model.AtomicNode)
