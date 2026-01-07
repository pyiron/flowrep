"""Unit tests for flowrep.model"""

import unittest

import pydantic

from flowrep import model


class TestAtomicNode(unittest.TestCase):
    """Tests for AtomicNode validation."""

    def test_valid_fnc(self):
        node = model.AtomicNode(
            fully_qualified_name="module.func",
            inputs=[],
            outputs=[],
        )
        self.assertEqual(node.fully_qualified_name, "module.func")
        self.assertEqual(node.type, "atomic")

    def test_valid_fnc_deep(self):
        node = model.AtomicNode(
            fully_qualified_name="a.b.c.d",
            inputs=[],
            outputs=[],
        )
        self.assertEqual(node.fully_qualified_name, "a.b.c.d")

    def test_fnc_no_period(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.AtomicNode(
                fully_qualified_name="noDot",
                inputs=[],
                outputs=[],
            )
        self.assertIn("at least one period", str(ctx.exception))

    def test_fnc_empty_string(self):
        with self.assertRaises(pydantic.ValidationError):
            model.AtomicNode(
                fully_qualified_name="",
                inputs=[],
                outputs=[],
            )

    def test_fnc_empty_part(self):
        """e.g., 'module.' or '.func' or 'a..b'"""
        for bad in ["module.", ".func", "a..b"]:
            with self.assertRaises(
                pydantic.ValidationError, msg=f"Should reject {bad!r}"
            ):
                model.AtomicNode(
                    fully_qualified_name=bad,
                    inputs=[],
                    outputs=[],
                )


class TestAtomicNodeUnpacking(unittest.TestCase):
    """Tests for unpack_mode validation."""

    def test_default_unpack_mode(self):
        """Default unpack_mode should be 'tuple'."""
        node = model.AtomicNode(
            fully_qualified_name="mod.func",
            inputs=["x"],
            outputs=["a", "b"],
        )
        self.assertEqual(node.unpack_mode, model.UnpackMode.TUPLE)

    def test_tuple_mode_multiple_outputs(self):
        """Multiple outputs allowed with unpack_mode='tuple'."""
        node = model.AtomicNode(
            fully_qualified_name="mod.func",
            inputs=[],
            outputs=["a", "b", "c"],
            unpack_mode=model.UnpackMode.TUPLE,
        )
        self.assertEqual(len(node.outputs), 3)
        self.assertEqual(node.unpack_mode, model.UnpackMode.TUPLE)

    def test_dataclass_mode_multiple_outputs(self):
        """Multiple outputs allowed with unpack_mode='dataclass'."""
        node = model.AtomicNode(
            fully_qualified_name="mod.func",
            inputs=[],
            outputs=["a", "b", "c"],
            unpack_mode=model.UnpackMode.DATACLASS,
        )
        self.assertEqual(len(node.outputs), 3)
        self.assertEqual(node.unpack_mode, model.UnpackMode.DATACLASS)

    def test_none_mode_multiple_outputs_rejected(self):
        """Multiple outputs rejected when unpack_mode='none'."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.AtomicNode(
                fully_qualified_name="mod.func",
                inputs=[],
                outputs=["a", "b"],
                unpack_mode=model.UnpackMode.NONE,
            )
        self.assertIn("exactly one element", str(ctx.exception))
        self.assertIn(f"unpack_mode={model.UnpackMode.NONE.value}", str(ctx.exception))

    def test_none_mode_single_output_valid(self):
        """Single output valid with unpack_mode='none'."""
        node = model.AtomicNode(
            fully_qualified_name="mod.func",
            inputs=["x"],
            outputs=["result"],
            unpack_mode=model.UnpackMode.NONE,
        )
        self.assertEqual(node.outputs, ["result"])
        self.assertEqual(node.unpack_mode, model.UnpackMode.NONE)

    def test_none_mode_zero_outputs_valid(self):
        """Zero outputs valid with unpack_mode='none'."""
        node = model.AtomicNode(
            fully_qualified_name="mod.func",
            inputs=[],
            outputs=[],
            unpack_mode=model.UnpackMode.NONE,
        )
        self.assertEqual(len(node.outputs), 0)
        self.assertEqual(node.unpack_mode, model.UnpackMode.NONE)

    def test_tuple_mode_zero_outputs_valid(self):
        """Zero outputs valid with unpack_mode='tuple'."""
        node = model.AtomicNode(
            fully_qualified_name="mod.func",
            inputs=["x"],
            outputs=[],
            unpack_mode=model.UnpackMode.TUPLE,
        )
        self.assertEqual(len(node.outputs), 0)

    def test_all_unpack_modes_valid_literal(self):
        """All three unpack modes should be valid."""
        for mode in ["none", "tuple", "dataclass"]:
            node = model.AtomicNode(
                fully_qualified_name="mod.func",
                inputs=[],
                outputs=["out"],
                unpack_mode=mode,
            )
            self.assertEqual(node.unpack_mode, mode)


class TestWorkflowNodeEdgeValidation(unittest.TestCase):
    """Tests for WorkflowNode edge validation."""

    def _make_workflow(self, edges):
        """Helper to create a workflow with given edges and one atomic child."""
        return model.WorkflowNode(
            inputs=["x"],
            outputs=["y"],
            nodes={
                "child": model.AtomicNode(
                    fully_qualified_name="mod.func",
                    inputs=["in"],
                    outputs=["out"],
                )
            },
            edges=edges,
        )

    def test_valid_edges(self):
        wf = self._make_workflow(
            {
                ("child", "in"): "x",  # tuple <- str (input to child)
                "y": ("child", "out"),  # str <- tuple (output from child)
            }
        )
        self.assertEqual(len(wf.edges), 2)

    def test_edge_both_strings_rejected(self):
        """
        dict[str, str] should fail - it would represent pass-through data.
        Don't do that, just make an edge going around this node!
        """
        with self.assertRaises(pydantic.ValidationError) as ctx:
            self._make_workflow({"y": "x"})
        self.assertIn(
            "both target and source cannot be plain strings", str(ctx.exception)
        )

    def test_edge_tuple_wrong_length(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            self._make_workflow({("child", "a", "extra"): "x"})
        self.assertIn("2 items", str(ctx.exception))

    def test_edge_reference_nonexistent_node(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            self._make_workflow({("nonexistent", "port"): "x"})
        self.assertIn("not a child node", str(ctx.exception))


class TestWorkflowNodePortValidation(unittest.TestCase):
    """Tests for port name validation in edges."""

    def test_invalid_input_port_name(self):
        """Edge references non-existent input port on child node."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.WorkflowNode(
                inputs=["x"],
                outputs=["y"],
                nodes={
                    "child": model.AtomicNode(
                        fully_qualified_name="mod.func",
                        inputs=["in"],
                        outputs=["out"],
                    )
                },
                edges={
                    ("child", "wrong_port"): "x",  # 'wrong_port' doesn't exist
                },
            )
        self.assertIn("has no input port", str(ctx.exception))
        self.assertIn("wrong_port", str(ctx.exception))

    def test_invalid_output_port_name(self):
        """Edge references non-existent output port on child node."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.WorkflowNode(
                inputs=["x"],
                outputs=["y"],
                nodes={
                    "child": model.AtomicNode(
                        fully_qualified_name="mod.func",
                        inputs=["in"],
                        outputs=["out"],
                    )
                },
                edges={
                    ("child", "in"): "x",
                    "y": ("child", "wrong_port"),  # 'wrong_port' doesn't exist
                },
            )
        self.assertIn("has no output port", str(ctx.exception))
        self.assertIn("wrong_port", str(ctx.exception))

    def test_invalid_workflow_input_name(self):
        """Edge references non-existent workflow input."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.WorkflowNode(
                inputs=["x"],
                outputs=["y"],
                nodes={
                    "child": model.AtomicNode(
                        fully_qualified_name="mod.func",
                        inputs=["in"],
                        outputs=["out"],
                    )
                },
                edges={
                    ("child", "in"): "nonexistent_input",  # doesn't exist
                },
            )
        self.assertIn("not a workflow input", str(ctx.exception))
        self.assertIn("nonexistent_input", str(ctx.exception))

    def test_invalid_workflow_output_name(self):
        """Edge references non-existent workflow output."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.WorkflowNode(
                inputs=["x"],
                outputs=["y"],
                nodes={
                    "child": model.AtomicNode(
                        fully_qualified_name="mod.func",
                        inputs=["in"],
                        outputs=["out"],
                    )
                },
                edges={
                    ("child", "in"): "x",
                    "nonexistent_output": ("child", "out"),  # doesn't exist
                },
            )
        self.assertIn("not a workflow output", str(ctx.exception))
        self.assertIn("nonexistent_output", str(ctx.exception))

    def test_multiple_ports_valid(self):
        """Valid edges with multiple input/output ports."""
        wf = model.WorkflowNode(
            inputs=["a", "b"],
            outputs=["x", "y"],
            nodes={
                "node1": model.AtomicNode(
                    fully_qualified_name="mod.func",
                    inputs=["in1", "in2"],
                    outputs=["out1", "out2"],
                )
            },
            edges={
                ("node1", "in1"): "a",
                ("node1", "in2"): "b",
                "x": ("node1", "out1"),
                "y": ("node1", "out2"),
            },
        )
        self.assertEqual(len(wf.edges), 4)

    def test_nested_workflow_port_validation(self):
        """Port validation works for nested workflows."""
        inner = model.WorkflowNode(
            inputs=["inner_in"],
            outputs=["inner_out"],
            nodes={
                "leaf": model.AtomicNode(
                    fully_qualified_name="m.f",
                    inputs=["x"],
                    outputs=["y"],
                )
            },
            edges={
                ("leaf", "x"): "inner_in",
                "inner_out": ("leaf", "y"),
            },
        )

        outer = model.WorkflowNode(
            inputs=["outer_in"],
            outputs=["outer_out"],
            nodes={"inner": inner},
            edges={
                ("inner", "inner_in"): "outer_in",
                "outer_out": ("inner", "inner_out"),
            },
        )
        self.assertEqual(len(outer.nodes), 1)

    def test_nested_workflow_invalid_port(self):
        """Port validation catches errors in nested workflow edges."""
        inner = model.WorkflowNode(
            inputs=["inner_in"],
            outputs=["inner_out"],
            nodes={
                "leaf": model.AtomicNode(
                    fully_qualified_name="m.f",
                    inputs=["x"],
                    outputs=["y"],
                )
            },
            edges={
                ("leaf", "x"): "inner_in",
                "inner_out": ("leaf", "y"),
            },
        )

        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.WorkflowNode(
                inputs=["outer_in"],
                outputs=["outer_out"],
                nodes={"inner": inner},
                edges={
                    ("inner", "wrong_port"): "outer_in",  # doesn't exist
                    "outer_out": ("inner", "inner_out"),
                },
            )
        self.assertIn("has no input port", str(ctx.exception))
        self.assertIn("wrong_port", str(ctx.exception))


class TestWorkflowNodeReservedNames(unittest.TestCase):
    """Tests for reserved node name validation."""

    def test_reserved_name_inputs(self):
        for reserved in model.WorkflowNode.reserved_node_names:
            with self.assertRaises(pydantic.ValidationError) as ctx:
                model.WorkflowNode(
                    inputs=["a"],
                    outputs=["b"],
                    nodes={
                        reserved: model.AtomicNode(
                            fully_qualified_name="m.f",
                            inputs=[],
                            outputs=[],
                        )
                    },
                    edges={},
                )
        self.assertIn("reserved names", str(ctx.exception))


class TestWorkflowNodeAcyclic(unittest.TestCase):
    """Tests for DAG validation."""

    def test_simple_cycle_rejected(self):
        """A -> B -> A should fail."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.WorkflowNode(
                inputs=["x"],
                outputs=["y"],
                nodes={
                    "a": model.AtomicNode(
                        fully_qualified_name="m.f",
                        inputs=["in", "feedback"],
                        outputs=["out"],
                    ),
                    "b": model.AtomicNode(
                        fully_qualified_name="m.g",
                        inputs=["in"],
                        outputs=["out", "out2"],
                    ),
                },
                edges={
                    ("a", "in"): "x",
                    ("b", "in"): ("a", "out"),
                    ("a", "feedback"): ("b", "out"),  # creates cycle
                    "y": ("b", "out2"),
                },
            )
        self.assertIn("cycle", str(ctx.exception).lower())

    def test_self_loop_rejected(self):
        """A -> A should fail."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.WorkflowNode(
                inputs=["x"],
                outputs=["y"],
                nodes={
                    "a": model.AtomicNode(
                        fully_qualified_name="m.f",
                        inputs=["in", "feedback"],
                        outputs=["out", "out2"],
                    )
                },
                edges={
                    ("a", "in"): "x",
                    ("a", "feedback"): ("a", "out"),  # self-loop
                    "y": ("a", "out2"),
                },
            )
        self.assertIn("cycle", str(ctx.exception).lower())

    def test_valid_dag(self):
        """Linear chain should pass."""
        wf = model.WorkflowNode(
            inputs=["x"],
            outputs=["y"],
            nodes={
                "a": model.AtomicNode(
                    fully_qualified_name="m.f",
                    inputs=["in"],
                    outputs=["out"],
                ),
                "b": model.AtomicNode(
                    fully_qualified_name="m.g",
                    inputs=["in"],
                    outputs=["out"],
                ),
                "c": model.AtomicNode(
                    fully_qualified_name="m.h",
                    inputs=["in"],
                    outputs=["out"],
                ),
            },
            edges={
                ("a", "in"): "x",
                ("b", "in"): ("a", "out"),
                ("c", "in"): ("b", "out"),
                "y": ("c", "out"),
            },
        )
        self.assertEqual(len(wf.nodes), 3)


class TestNestedWorkflow(unittest.TestCase):
    """Tests for nested workflow structures."""

    def test_nested_construction(self):
        """Nested workflows should validate recursively."""
        outer = model.WorkflowNode(
            inputs=["x", "y"],
            outputs=["z"],
            nodes={
                "inner": model.WorkflowNode(
                    inputs=["a"],
                    outputs=["b"],
                    nodes={
                        "leaf": model.AtomicNode(
                            fully_qualified_name="m.f",
                            inputs=["in"],
                            outputs=["out"],
                        )
                    },
                    edges={
                        ("leaf", "in"): "a",
                        "b": ("leaf", "out"),
                    },
                ),
            },
            edges={
                ("inner", "a"): "x",
                "z": ("inner", "b"),
            },
        )
        self.assertIsInstance(outer.nodes["inner"], model.WorkflowNode)

    def test_nested_invalid_fnc_bubbles_up(self):
        """Validation errors in nested nodes should propagate."""
        with self.assertRaises(pydantic.ValidationError):
            model.WorkflowNode(
                inputs=["x"],
                outputs=["y"],
                nodes={
                    "inner": model.WorkflowNode(
                        inputs=["a"],
                        outputs=["b"],
                        nodes={
                            "bad": model.AtomicNode(
                                fully_qualified_name="noDot",
                                inputs=[],
                                outputs=[],
                            )
                        },
                        edges={},
                    ),
                },
                edges={},
            )


class TestSerialization(unittest.TestCase):
    """Tests for JSON and Python mode serialization roundtrip."""

    def test_atomic_json_roundtrip(self):
        """JSON mode roundtrip for AtomicNode."""
        original = model.AtomicNode(
            fully_qualified_name="mod.func",
            inputs=["a"],
            outputs=["b"],
        )
        data = original.model_dump(mode="json")
        restored = model.AtomicNode.model_validate(data)
        self.assertEqual(original, restored)

    def test_atomic_python_roundtrip(self):
        """Python mode roundtrip for AtomicNode."""
        original = model.AtomicNode(
            fully_qualified_name="mod.func",
            inputs=["a"],
            outputs=["b"],
        )
        data = original.model_dump(mode="python")
        restored = model.AtomicNode.model_validate(data)
        self.assertEqual(original, restored)

    def test_workflow_json_roundtrip(self):
        """JSON mode roundtrip for WorkflowNode."""
        original = model.WorkflowNode(
            inputs=["x"],
            outputs=["y"],
            nodes={
                "n": model.AtomicNode(
                    fully_qualified_name="m.f",
                    inputs=["in"],
                    outputs=["out"],
                )
            },
            edges={("n", "in"): "x", "y": ("n", "out")},
        )
        data = original.model_dump(mode="json")
        restored = model.WorkflowNode.model_validate(data)
        self.assertEqual(original.inputs, restored.inputs)
        self.assertEqual(original.outputs, restored.outputs)
        self.assertEqual(len(original.nodes), len(restored.nodes))
        self.assertEqual(original.edges, restored.edges)

    def test_workflow_python_roundtrip(self):
        """Python mode roundtrip for WorkflowNode."""
        original = model.WorkflowNode(
            inputs=["x"],
            outputs=["y"],
            nodes={
                "n": model.AtomicNode(
                    fully_qualified_name="m.f",
                    inputs=["in"],
                    outputs=["out"],
                )
            },
            edges={("n", "in"): "x", "y": ("n", "out")},
        )
        data = original.model_dump(mode="python")
        restored = model.WorkflowNode.model_validate(data)
        self.assertEqual(original.inputs, restored.inputs)
        self.assertEqual(original.outputs, restored.outputs)
        self.assertEqual(len(original.nodes), len(restored.nodes))
        self.assertEqual(original.edges, restored.edges)

    def test_workflow_python_mode_preserves_dict_structure(self):
        """Python mode should preserve dict with tuple keys, not convert to list."""
        original = model.WorkflowNode(
            inputs=["x", "y"],
            outputs=["z", "w"],
            nodes={
                "a": model.AtomicNode(
                    fully_qualified_name="m.f",
                    inputs=["i1", "i2"],
                    outputs=["o1", "o2"],
                )
            },
            edges={
                ("a", "i1"): "x",
                ("a", "i2"): "y",
                "z": ("a", "o1"),
                "w": ("a", "o2"),
            },
        )
        data = original.model_dump(mode="python")

        # Verify edges is a dict, not a list
        self.assertIsInstance(data["edges"], dict)

        # Verify tuple keys are preserved
        self.assertIn(("a", "i1"), data["edges"])
        self.assertIn(("a", "i2"), data["edges"])

        # Verify values match
        self.assertEqual(data["edges"][("a", "i1")], "x")
        self.assertEqual(data["edges"]["z"], ("a", "o1"))

    def test_workflow_json_mode_uses_list_structure(self):
        """JSON mode should convert dict to list of pairs."""
        original = model.WorkflowNode(
            inputs=["x"],
            outputs=["y"],
            nodes={
                "n": model.AtomicNode(
                    fully_qualified_name="m.f",
                    inputs=["in"],
                    outputs=["out"],
                )
            },
            edges={("n", "in"): "x", "y": ("n", "out")},
        )
        data = original.model_dump(mode="json")

        # Verify edges is a list
        self.assertIsInstance(data["edges"], list)
        self.assertEqual(len(data["edges"]), 2)

        # Each element should be a [key, value] pair
        for item in data["edges"]:
            self.assertIsInstance(item, list)
            self.assertEqual(len(item), 2)

    def test_discriminated_union_roundtrip(self):
        """Ensure type discriminator works for polymorphic deserialization."""
        data = {
            "type": "atomic",
            "fully_qualified_name": "a.b",
            "inputs": ["x"],
            "outputs": ["y"],
        }
        node = pydantic.TypeAdapter(model.NodeType).validate_python(data)
        self.assertIsInstance(node, model.AtomicNode)

        data = {
            "type": "workflow",
            "inputs": [],
            "outputs": [],
            "nodes": {},
            "edges": {},
        }
        node = pydantic.TypeAdapter(model.NodeType).validate_python(data)
        self.assertIsInstance(node, model.WorkflowNode)


class TestEmptyWorkflow(unittest.TestCase):
    """Edge cases with empty collections."""

    def test_empty_nodes_and_edges(self):
        wf = model.WorkflowNode(inputs=[], outputs=[], nodes={}, edges={})
        self.assertEqual(wf.nodes, {})
        self.assertEqual(wf.edges, {})

    def test_empty_inputs_outputs(self):
        wf = model.WorkflowNode(
            inputs=[],
            outputs=[],
            nodes={
                "n": model.AtomicNode(
                    fully_qualified_name="m.f",
                    inputs=[],
                    outputs=[],
                )
            },
            edges={},
        )
        self.assertEqual(wf.inputs, [])


if __name__ == "__main__":
    unittest.main()
