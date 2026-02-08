import unittest

import pydantic

from flowrep.models import base_models, edge_models, subgraph_validation
from flowrep.models.nodes import atomic_model, workflow_model


class TestWorkflowNodeStructure(unittest.TestCase):
    """Tests for protocol validation and schema availability."""

    def test_schema_generation(self):
        """model_json_schema() fails if forward refs aren't resolved."""
        workflow_model.WorkflowNode.model_json_schema()

    def test_obeys_has_static_subgraph(self):
        wf = workflow_model.WorkflowNode(
            inputs=[],
            outputs=[],
            nodes={},
            input_edges={},
            edges={},
            output_edges={},
        )
        self.assertIsInstance(wf, subgraph_validation.StaticSubgraphOwner)


class TestWorkflowNodeInputEdges(unittest.TestCase):
    """Tests for input_edges validation (workflow inputs -> child inputs)."""

    def test_valid_input_edge(self):
        """Input edge from workflow input to child input."""
        wf = workflow_model.WorkflowNode(
            inputs=["x"],
            outputs=["y"],
            nodes={
                "child": atomic_model.AtomicNode(
                    fully_qualified_name="mod.func",
                    inputs=["inp"],
                    outputs=["out"],
                )
            },
            input_edges={
                edge_models.TargetHandle(
                    node="child", port="inp"
                ): edge_models.InputSource(port="x"),
            },
            edges={},
            output_edges={
                edge_models.OutputTarget(port="y"): edge_models.SourceHandle(
                    node="child", port="out"
                ),
            },
        )
        self.assertEqual(len(wf.input_edges), 1)

    def test_input_edge_invalid_workflow_input(self):
        """Input edge referencing nonexistent workflow input."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            workflow_model.WorkflowNode(
                inputs=["x"],
                outputs=["y"],
                nodes={
                    "child": atomic_model.AtomicNode(
                        fully_qualified_name="mod.func",
                        inputs=["inp"],
                        outputs=["out"],
                    )
                },
                input_edges={
                    edge_models.TargetHandle(
                        node="child", port="inp"
                    ): edge_models.InputSource(port="nonexistent"),
                },
                edges={},
                output_edges={},
            )
        self.assertIn("Invalid input_edges source ports", str(ctx.exception))

    def test_input_edge_invalid_child_node(self):
        """Input edge referencing nonexistent child node."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            workflow_model.WorkflowNode(
                inputs=["x"],
                outputs=["y"],
                nodes={
                    "child": atomic_model.AtomicNode(
                        fully_qualified_name="mod.func",
                        inputs=["inp"],
                        outputs=["out"],
                    )
                },
                input_edges={
                    edge_models.TargetHandle(
                        node="nonexistent", port="inp"
                    ): edge_models.InputSource(port="x"),
                },
                edges={},
                output_edges={},
            )
        self.assertIn("Invalid input_edges target nodes", str(ctx.exception))
        self.assertIn("nonexistent", str(ctx.exception))

    def test_input_edge_invalid_child_port(self):
        """Input edge referencing nonexistent port on child."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            workflow_model.WorkflowNode(
                inputs=["x"],
                outputs=["y"],
                nodes={
                    "child": atomic_model.AtomicNode(
                        fully_qualified_name="mod.func",
                        inputs=["inp"],
                        outputs=["out"],
                    )
                },
                input_edges={
                    edge_models.TargetHandle(
                        node="child", port="wrong"
                    ): edge_models.InputSource(port="x"),
                },
                edges={},
                output_edges={},
            )
        self.assertIn("Invalid input_edges target ports", str(ctx.exception))
        self.assertIn("wrong", str(ctx.exception))


class TestWorkflowNodeOutputEdges(unittest.TestCase):
    """Tests for output_edges validation (child outputs -> workflow outputs)."""

    def test_valid_output_edge(self):
        """Output edge from child output to workflow output."""
        wf = workflow_model.WorkflowNode(
            inputs=["x"],
            outputs=["y"],
            nodes={
                "child": atomic_model.AtomicNode(
                    fully_qualified_name="mod.func",
                    inputs=["inp"],
                    outputs=["out"],
                )
            },
            input_edges={
                edge_models.TargetHandle(
                    node="child", port="inp"
                ): edge_models.InputSource(port="x"),
            },
            edges={},
            output_edges={
                edge_models.OutputTarget(port="y"): edge_models.SourceHandle(
                    node="child", port="out"
                ),
            },
        )
        self.assertEqual(len(wf.output_edges), 1)

    def test_output_edge_invalid_workflow_output(self):
        """Output edge referencing nonexistent workflow output."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            workflow_model.WorkflowNode(
                inputs=["x"],
                outputs=["y"],
                nodes={
                    "child": atomic_model.AtomicNode(
                        fully_qualified_name="mod.func",
                        inputs=["inp"],
                        outputs=["out"],
                    )
                },
                input_edges={},
                edges={},
                output_edges={
                    edge_models.OutputTarget(
                        port="nonexistent"
                    ): edge_models.SourceHandle(node="child", port="out"),
                },
            )
        self.assertIn("Invalid output target ports", str(ctx.exception))
        self.assertIn("nonexistent", str(ctx.exception))

    def test_output_edge_invalid_child_node(self):
        """Output edge referencing nonexistent child node."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            workflow_model.WorkflowNode(
                inputs=["x"],
                outputs=["y"],
                nodes={
                    "child": atomic_model.AtomicNode(
                        fully_qualified_name="mod.func",
                        inputs=["inp"],
                        outputs=["out"],
                    )
                },
                input_edges={},
                edges={},
                output_edges={
                    edge_models.OutputTarget(port="y"): edge_models.SourceHandle(
                        node="nonexistent", port="out"
                    ),
                },
            )
        self.assertIn("Invalid output source nodes", str(ctx.exception))
        self.assertIn("nonexistent", str(ctx.exception))

    def test_output_edge_invalid_child_port(self):
        """Output edge referencing nonexistent port on child."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            workflow_model.WorkflowNode(
                inputs=["x"],
                outputs=["y"],
                nodes={
                    "child": atomic_model.AtomicNode(
                        fully_qualified_name="mod.func",
                        inputs=["inp"],
                        outputs=["out"],
                    )
                },
                input_edges={},
                edges={},
                output_edges={
                    edge_models.OutputTarget(port="y"): edge_models.SourceHandle(
                        node="child", port="wrong_port"
                    ),
                    # 'wrong_port' doesn't exist
                },
            )
        self.assertIn("Invalid output source ports", str(ctx.exception))
        self.assertIn("wrong_port", str(ctx.exception))


class TestWorkflowNodeInternalEdges(unittest.TestCase):
    """Tests for edges validation (child outputs -> child inputs)."""

    def test_valid_internal_edge(self):
        """Edge between two child nodes."""
        wf = workflow_model.WorkflowNode(
            inputs=["x"],
            outputs=["y"],
            nodes={
                "a": atomic_model.AtomicNode(
                    fully_qualified_name="mod.f",
                    inputs=["inp"],
                    outputs=["out"],
                ),
                "b": atomic_model.AtomicNode(
                    fully_qualified_name="mod.g",
                    inputs=["inp"],
                    outputs=["out"],
                ),
            },
            input_edges={
                edge_models.TargetHandle(node="a", port="inp"): edge_models.InputSource(
                    port="x"
                ),
            },
            edges={
                edge_models.TargetHandle(
                    node="b", port="inp"
                ): edge_models.SourceHandle(node="a", port="out"),
            },
            output_edges={
                edge_models.OutputTarget(port="y"): edge_models.SourceHandle(
                    node="b", port="out"
                ),
            },
        )
        self.assertEqual(len(wf.edges), 1)

    def test_internal_edge_invalid_source_node(self):
        """Internal edge from nonexistent source node."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            workflow_model.WorkflowNode(
                inputs=["x"],
                outputs=[],
                nodes={
                    "child": atomic_model.AtomicNode(
                        fully_qualified_name="mod.func",
                        inputs=["inp"],
                        outputs=["out"],
                    )
                },
                input_edges={},
                edges={
                    edge_models.TargetHandle(
                        node="child", port="inp"
                    ): edge_models.SourceHandle(node="nonexistent", port="out"),
                },
                output_edges={},
            )
        self.assertIn("Invalid edge source node", str(ctx.exception))
        self.assertIn("nonexistent", str(ctx.exception))

    def test_internal_edge_invalid_target_node(self):
        """Internal edge to nonexistent target node."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            workflow_model.WorkflowNode(
                inputs=["x"],
                outputs=[],
                nodes={
                    "child": atomic_model.AtomicNode(
                        fully_qualified_name="mod.func",
                        inputs=["inp"],
                        outputs=["out"],
                    )
                },
                input_edges={},
                edges={
                    edge_models.TargetHandle(
                        node="nonexistent", port="inp"
                    ): edge_models.SourceHandle(node="child", port="out"),
                },
                output_edges={},
            )
        self.assertIn("Invalid edge target node", str(ctx.exception))
        self.assertIn("nonexistent", str(ctx.exception))

    def test_internal_edge_invalid_source_port(self):
        """Internal edge from nonexistent source port."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            workflow_model.WorkflowNode(
                inputs=["x"],
                outputs=[],
                nodes={
                    "a": atomic_model.AtomicNode(
                        fully_qualified_name="mod.f",
                        inputs=["inp"],
                        outputs=["out"],
                    ),
                    "b": atomic_model.AtomicNode(
                        fully_qualified_name="mod.g",
                        inputs=["inp"],
                        outputs=["out"],
                    ),
                },
                input_edges={},
                edges={
                    edge_models.TargetHandle(
                        node="b", port="inp"
                    ): edge_models.SourceHandle(node="a", port="wrong"),
                },
                output_edges={},
            )
        self.assertIn("Invalid edge source port", str(ctx.exception))
        self.assertIn("wrong", str(ctx.exception))

    def test_internal_edge_invalid_target_port(self):
        """Internal edge to nonexistent target port."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            workflow_model.WorkflowNode(
                inputs=["x"],
                outputs=[],
                nodes={
                    "a": atomic_model.AtomicNode(
                        fully_qualified_name="mod.f",
                        inputs=["inp"],
                        outputs=["out"],
                    ),
                    "b": atomic_model.AtomicNode(
                        fully_qualified_name="mod.g",
                        inputs=["inp"],
                        outputs=["out"],
                    ),
                },
                input_edges={},
                edges={
                    edge_models.TargetHandle(
                        node="b", port="wrong"
                    ): edge_models.SourceHandle(node="a", port="out"),
                },
                output_edges={},
            )
        self.assertIn("Invalid edge target port", str(ctx.exception))
        self.assertIn("wrong", str(ctx.exception))


class TestWorkflowNodeMultiplePorts(unittest.TestCase):
    """Tests for workflows with multiple input/output ports."""

    def test_multiple_ports_valid(self):
        """Valid edges with multiple input/output ports."""
        wf = workflow_model.WorkflowNode(
            inputs=["a", "b"],
            outputs=["x", "y"],
            nodes={
                "node1": atomic_model.AtomicNode(
                    fully_qualified_name="mod.func",
                    inputs=["in1", "in2"],
                    outputs=["out1", "out2"],
                )
            },
            input_edges={
                edge_models.TargetHandle(
                    node="node1", port="in1"
                ): edge_models.InputSource(port="a"),
                edge_models.TargetHandle(
                    node="node1", port="in2"
                ): edge_models.InputSource(port="b"),
            },
            edges={},
            output_edges={
                edge_models.OutputTarget(port="x"): edge_models.SourceHandle(
                    node="node1", port="out1"
                ),
                edge_models.OutputTarget(port="y"): edge_models.SourceHandle(
                    node="node1", port="out2"
                ),
            },
        )
        self.assertEqual(len(wf.input_edges), 2)
        self.assertEqual(len(wf.output_edges), 2)


class TestWorkflowNodeReservedNames(unittest.TestCase):
    """Tests for reserved node name validation."""

    def test_reserved_node_names(self):
        """Node labels cannot use reserved names."""
        test_cases = [
            ("for", "Python keyword"),
            ("while", "Python keyword"),
            *[(reserved, "reserved name") for reserved in base_models.RESERVED_NAMES],
            ("1invalid", "not an identifier"),
            ("my-var", "not an identifier"),
            ("my var", "not an identifier"),
            ("", "not an identifier"),
        ]

        for invalid_label, reason in test_cases:
            with self.subTest(label=invalid_label, reason=reason):
                with self.assertRaises(pydantic.ValidationError) as ctx:
                    workflow_model.WorkflowNode(
                        inputs=["a"],
                        outputs=["b"],
                        nodes={
                            invalid_label: atomic_model.AtomicNode(
                                fully_qualified_name="m.f",
                                inputs=[],
                                outputs=[],
                            )
                        },
                        input_edges={},
                        edges={},
                        output_edges={},
                    )
                exc_str = str(ctx.exception)
                self.assertIn(invalid_label, exc_str)


class TestWorkflowNodeAcyclic(unittest.TestCase):
    """Tests for DAG validation."""

    def test_simple_cycle_rejected(self):
        """A -> B -> A should fail."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            workflow_model.WorkflowNode(
                inputs=["x"],
                outputs=["y"],
                nodes={
                    "a": atomic_model.AtomicNode(
                        fully_qualified_name="m.f",
                        inputs=["inp", "feedback"],
                        outputs=["out"],
                    ),
                    "b": atomic_model.AtomicNode(
                        fully_qualified_name="m.g",
                        inputs=["inp"],
                        outputs=["out", "out2"],
                    ),
                },
                input_edges={
                    edge_models.TargetHandle(
                        node="a", port="inp"
                    ): edge_models.InputSource(port="x"),
                },
                edges={
                    edge_models.TargetHandle(
                        node="b", port="inp"
                    ): edge_models.SourceHandle(node="a", port="out"),
                    edge_models.TargetHandle(
                        node="a", port="feedback"
                    ): edge_models.SourceHandle(node="b", port="out"),
                },
                output_edges={
                    edge_models.OutputTarget(port="y"): edge_models.SourceHandle(
                        node="b", port="out2"
                    ),
                },
            )
        self.assertIn("cycle", str(ctx.exception).lower())

    def test_self_loop_rejected(self):
        """A -> A should fail."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            workflow_model.WorkflowNode(
                inputs=["x"],
                outputs=["y"],
                nodes={
                    "a": atomic_model.AtomicNode(
                        fully_qualified_name="m.f",
                        inputs=["inp", "feedback"],
                        outputs=["out", "out2"],
                    )
                },
                input_edges={
                    edge_models.TargetHandle(
                        node="a", port="inp"
                    ): edge_models.InputSource(port="x"),
                },
                edges={
                    edge_models.TargetHandle(
                        node="a", port="feedback"
                    ): edge_models.SourceHandle(node="a", port="out"),
                },
                output_edges={
                    edge_models.OutputTarget(port="y"): edge_models.SourceHandle(
                        node="a", port="out2"
                    ),
                },
            )
        self.assertIn("cycle", str(ctx.exception).lower())

    def test_valid_dag(self):
        """Linear chain should pass."""
        wf = workflow_model.WorkflowNode(
            inputs=["x"],
            outputs=["y"],
            nodes={
                "a": atomic_model.AtomicNode(
                    fully_qualified_name="m.f",
                    inputs=["inp"],
                    outputs=["out"],
                ),
                "b": atomic_model.AtomicNode(
                    fully_qualified_name="m.g",
                    inputs=["inp"],
                    outputs=["out"],
                ),
                "c": atomic_model.AtomicNode(
                    fully_qualified_name="m.h",
                    inputs=["inp"],
                    outputs=["out"],
                ),
            },
            input_edges={
                edge_models.TargetHandle(node="a", port="inp"): edge_models.InputSource(
                    port="x"
                ),
            },
            edges={
                edge_models.TargetHandle(
                    node="b", port="inp"
                ): edge_models.SourceHandle(node="a", port="out"),
                edge_models.TargetHandle(
                    node="c", port="inp"
                ): edge_models.SourceHandle(node="b", port="out"),
            },
            output_edges={
                edge_models.OutputTarget(port="y"): edge_models.SourceHandle(
                    node="c", port="out"
                ),
            },
        )
        self.assertEqual(len(wf.nodes), 3)


class TestNestedWorkflow(unittest.TestCase):
    """Tests for nested workflow structures."""

    def test_nested_construction(self):
        """Nested workflows should validate recursively."""
        inner = workflow_model.WorkflowNode(
            inputs=["a"],
            outputs=["b"],
            nodes={
                "leaf": atomic_model.AtomicNode(
                    fully_qualified_name="m.f",
                    inputs=["inp"],
                    outputs=["out"],
                )
            },
            input_edges={
                edge_models.TargetHandle(
                    node="leaf", port="inp"
                ): edge_models.InputSource(port="a"),
            },
            edges={},
            output_edges={
                edge_models.OutputTarget(port="b"): edge_models.SourceHandle(
                    node="leaf", port="out"
                ),
            },
        )

        outer = workflow_model.WorkflowNode(
            inputs=["x", "y"],
            outputs=["z"],
            nodes={"inner": inner},
            input_edges={
                edge_models.TargetHandle(
                    node="inner", port="a"
                ): edge_models.InputSource(port="x"),
            },
            edges={},
            output_edges={
                edge_models.OutputTarget(port="z"): edge_models.SourceHandle(
                    node="inner", port="b"
                ),
            },
        )
        self.assertIsInstance(outer.nodes["inner"], workflow_model.WorkflowNode)

    def test_nested_invalid_fqn_bubbles_up(self):
        """Validation errors in nested nodes should propagate."""
        with self.assertRaises(pydantic.ValidationError):
            workflow_model.WorkflowNode(
                inputs=["x"],
                outputs=["y"],
                nodes={
                    "inner": workflow_model.WorkflowNode(
                        inputs=["a"],
                        outputs=["b"],
                        nodes={
                            "bad": atomic_model.AtomicNode(
                                fully_qualified_name="noDot",
                                inputs=[],
                                outputs=[],
                            )
                        },
                        input_edges={},
                        edges={},
                        output_edges={},
                    ),
                },
                input_edges={},
                edges={},
                output_edges={},
            )

    def test_nested_workflow_port_validation(self):
        """Port validation works for nested workflows."""
        inner = workflow_model.WorkflowNode(
            inputs=["inner_in"],
            outputs=["inner_out"],
            nodes={
                "leaf": atomic_model.AtomicNode(
                    fully_qualified_name="m.f",
                    inputs=["x"],
                    outputs=["y"],
                )
            },
            input_edges={
                edge_models.TargetHandle(
                    node="leaf", port="x"
                ): edge_models.InputSource(port="inner_in"),
            },
            edges={},
            output_edges={
                edge_models.OutputTarget(port="inner_out"): edge_models.SourceHandle(
                    node="leaf", port="y"
                ),
            },
        )

        outer = workflow_model.WorkflowNode(
            inputs=["outer_in"],
            outputs=["outer_out"],
            nodes={"inner": inner},
            input_edges={
                edge_models.TargetHandle(
                    node="inner", port="inner_in"
                ): edge_models.InputSource(port="outer_in"),
            },
            edges={},
            output_edges={
                edge_models.OutputTarget(port="outer_out"): edge_models.SourceHandle(
                    node="inner", port="inner_out"
                ),
            },
        )
        self.assertEqual(len(outer.nodes), 1)

    def test_nested_workflow_invalid_port(self):
        """Port validation catches errors in nested workflow edges."""
        inner = workflow_model.WorkflowNode(
            inputs=["inner_in"],
            outputs=["inner_out"],
            nodes={
                "leaf": atomic_model.AtomicNode(
                    fully_qualified_name="m.f",
                    inputs=["x"],
                    outputs=["y"],
                )
            },
            input_edges={
                edge_models.TargetHandle(
                    node="leaf", port="x"
                ): edge_models.InputSource(port="inner_in"),
            },
            edges={},
            output_edges={
                edge_models.OutputTarget(port="inner_out"): edge_models.SourceHandle(
                    node="leaf", port="y"
                ),
            },
        )

        with self.assertRaises(pydantic.ValidationError) as ctx:
            workflow_model.WorkflowNode(
                inputs=["outer_in"],
                outputs=["outer_out"],
                nodes={"inner": inner},
                input_edges={
                    edge_models.TargetHandle(
                        node="inner", port="wrong_port"
                    ): edge_models.InputSource(port="outer_in"),
                },
                edges={},
                output_edges={
                    edge_models.OutputTarget(
                        port="outer_out"
                    ): edge_models.SourceHandle(node="inner", port="inner_out"),
                },
            )
        self.assertIn("Invalid input_edges target ports", str(ctx.exception))
        self.assertIn("wrong_port", str(ctx.exception))


class TestEmptyWorkflow(unittest.TestCase):
    """Edge cases with empty collections."""

    def test_empty_nodes_and_edges(self):
        wf = workflow_model.WorkflowNode(
            inputs=[],
            outputs=[],
            nodes={},
            input_edges={},
            edges={},
            output_edges={},
        )
        self.assertEqual(wf.nodes, {})
        self.assertEqual(wf.input_edges, {})
        self.assertEqual(wf.edges, {})
        self.assertEqual(wf.output_edges, {})

    def test_empty_inputs_outputs(self):
        wf = workflow_model.WorkflowNode(
            inputs=[],
            outputs=[],
            nodes={
                "n": atomic_model.AtomicNode(
                    fully_qualified_name="m.f",
                    inputs=[],
                    outputs=[],
                )
            },
            input_edges={},
            edges={},
            output_edges={},
        )
        self.assertEqual(wf.inputs, [])
        self.assertEqual(wf.outputs, [])


class TestWorkflowNodeSerialization(unittest.TestCase):
    def test_workflow_python_roundtrip(self):
        """Roundtrip for WorkflowNode."""
        original = workflow_model.WorkflowNode(
            inputs=["x"],
            outputs=["y"],
            nodes={
                "n": atomic_model.AtomicNode(
                    fully_qualified_name="m.f",
                    inputs=["inp"],
                    outputs=["out"],
                )
            },
            input_edges={
                edge_models.TargetHandle(node="n", port="inp"): edge_models.InputSource(
                    port="x"
                ),
            },
            edges={},
            output_edges={
                edge_models.OutputTarget(port="y"): edge_models.SourceHandle(
                    node="n", port="out"
                ),
            },
        )
        for mode in ["python", "json"]:
            with self.subTest(mode=mode):
                data = original.model_dump(mode=mode)
                restored = workflow_model.WorkflowNode.model_validate(data)
                self.assertEqual(original.inputs, restored.inputs)
                self.assertEqual(original.outputs, restored.outputs)
                self.assertEqual(len(original.nodes), len(restored.nodes))
                self.assertEqual(original.input_edges, restored.input_edges)
                self.assertEqual(original.edges, restored.edges)
                self.assertEqual(original.output_edges, restored.output_edges)

    def test_edge_serialization_format(self):
        """Edges serialize handles to dot-notation strings."""
        original = workflow_model.WorkflowNode(
            inputs=["x", "y"],
            outputs=["z", "w"],
            nodes={
                "a": atomic_model.AtomicNode(
                    fully_qualified_name="m.f",
                    inputs=["i1", "i2"],
                    outputs=["o1", "o2"],
                ),
                "b": atomic_model.AtomicNode(
                    fully_qualified_name="m.g",
                    inputs=["inp"],
                    outputs=["out"],
                ),
            },
            input_edges={
                edge_models.TargetHandle(node="a", port="i1"): edge_models.InputSource(
                    port="x"
                ),
                edge_models.TargetHandle(node="a", port="i2"): edge_models.InputSource(
                    port="y"
                ),
            },
            edges={
                edge_models.TargetHandle(
                    node="b", port="inp"
                ): edge_models.SourceHandle(node="a", port="o1"),
            },
            output_edges={
                edge_models.OutputTarget(port="z"): edge_models.SourceHandle(
                    node="a", port="o2"
                ),
                edge_models.OutputTarget(port="w"): edge_models.SourceHandle(
                    node="b", port="out"
                ),
            },
        )
        data = original.model_dump(mode="json")

        # input_edges: keys are "node.port", values are just "port"
        self.assertIsInstance(data["input_edges"], dict)
        self.assertIn("a.i1", data["input_edges"])
        self.assertEqual(data["input_edges"]["a.i1"], "x")

        # edges: both keys and values are "node.port"
        self.assertIsInstance(data["edges"], dict)
        self.assertIn("b.inp", data["edges"])
        self.assertEqual(data["edges"]["b.inp"], "a.o1")

        # output_edges: keys are just "port", values are "node.port"
        self.assertIsInstance(data["output_edges"], dict)
        self.assertIn("z", data["output_edges"])
        self.assertEqual(data["output_edges"]["z"], "a.o2")


if __name__ == "__main__":
    unittest.main()
