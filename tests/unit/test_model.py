"""Unit tests for flowrep.model"""

import unittest

import pydantic

from flowrep import model


class TestNodeModel(unittest.TestCase):
    """Tests for input/output uniqueness validation on NodeModel base class."""

    def test_duplicate_inputs_rejected(self):
        """Any NodeModel subclass should reject duplicate inputs."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.AtomicNode(
                fully_qualified_name="mod.func",
                inputs=["x", "y", "x"],  # duplicate 'x'
                outputs=["z"],
            )
        self.assertIn("unique", str(ctx.exception).lower())
        self.assertIn("x", str(ctx.exception))

    def test_duplicate_outputs_rejected(self):
        """Any NodeModel subclass should reject duplicate outputs."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.WorkflowNode(
                inputs=["a"],
                outputs=["x", "y", "x"],  # duplicate 'x'
                nodes={},
                edges={},
            )
        self.assertIn("unique", str(ctx.exception).lower())
        self.assertIn("x", str(ctx.exception))

    def test_unique_inputs_outputs_preserved_order(self):
        """Unique inputs/outputs should preserve declaration order."""
        node = model.AtomicNode(
            fully_qualified_name="mod.func",
            inputs=["c", "a", "b"],
            outputs=["z", "x", "y"],
        )
        # Order must be preserved for function signature mapping
        self.assertEqual(node.inputs, ["c", "a", "b"])
        self.assertEqual(node.outputs, ["z", "x", "y"])

    def test_invalid_IO_labels(self):
        test_cases = [
            ("for", "Python keyword"),
            ("while", "Python keyword"),
            *[(reserved, "reserved name") for reserved in model.RESERVED_NAMES],
            ("1invalid", "not an identifier"),
            ("my-var", "not an identifier"),
            ("my var", "not an identifier"),
            ("", "not an identifier"),
        ]

        for io_type in ["inputs", "outputs"]:
            for invalid_label, reason in test_cases:
                with self.subTest(io_type=io_type, label=invalid_label, reason=reason):
                    with self.assertRaises(pydantic.ValidationError) as ctx:
                        kwargs = {
                            "fully_qualified_name": "mod.func",
                            "inputs": ["x"],
                            "outputs": ["y"],
                        }
                        kwargs[io_type] = [invalid_label, "valid"]
                        model.AtomicNode(**kwargs)

                    exc_str = str(ctx.exception)
                    self.assertIn(
                        "valid Python identifier",
                        exc_str,
                        f"{io_type} with {invalid_label} ({reason}) should fail",
                    )
                    if invalid_label:  # empty string won't appear in error
                        self.assertIn(invalid_label, exc_str)

    def test_valid_IO_labels(self):
        """Valid identifiers should pass."""
        node = model.AtomicNode(
            fully_qualified_name="mod.func",
            inputs=["x", "y_1", "_private", "camelCase"],
            outputs=["result", "status_code"],
        )
        self.assertEqual(len(node.inputs), 4)
        self.assertEqual(len(node.outputs), 2)


class TestAtomicNode(unittest.TestCase):
    """Tests for AtomicNode validation."""

    def test_valid_fqn(self):
        node = model.AtomicNode(
            fully_qualified_name="module.func",
            inputs=[],
            outputs=[],
        )
        self.assertEqual(node.fully_qualified_name, "module.func")
        self.assertEqual(node.type, "atomic")

    def test_valid_fqn_deep(self):
        node = model.AtomicNode(
            fully_qualified_name="a.b.c.d",
            inputs=[],
            outputs=[],
        )
        self.assertEqual(node.fully_qualified_name, "a.b.c.d")

    def test_fqn_no_period(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.AtomicNode(
                fully_qualified_name="noDot",
                inputs=[],
                outputs=[],
            )
        self.assertIn("at least one period", str(ctx.exception))

    def test_fqn_empty_string(self):
        with self.assertRaises(pydantic.ValidationError):
            model.AtomicNode(
                fully_qualified_name="",
                inputs=[],
                outputs=[],
            )

    def test_fqn_empty_part(self):
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
                    inputs=["inp"],
                    outputs=["out"],
                )
            },
            edges={
                model.TargetHandle(node=k[0], port=k[1]): model.SourceHandle(
                    node=v[0], port=v[1]
                )
                for k, v in edges.items()
            },
        )

    def test_valid_edges(self):
        wf = self._make_workflow(
            {
                ("child", "inp"): (None, "x"),
                (None, "y"): ("child", "out"),
            }
        )
        self.assertEqual(len(wf.edges), 2)

    def test_edge_both_strings_rejected(self):
        """
        dict[str, str] should fail - it would represent pass-through data.
        Don't do that, just make an edge going around this node!
        """
        with self.assertRaises(pydantic.ValidationError) as ctx:
            self._make_workflow({(None, "y"): (None, "x")})
        self.assertIn(
            "Invalid edge: No pass-through data -- if a workflow declares IO "
            "it should use it",
            str(ctx.exception),
        )

    def test_edge_reference_nonexistent_node(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            self._make_workflow({("nonexistent", "port"): (None, "x")})
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
                        inputs=["inp"],
                        outputs=["out"],
                    )
                },
                edges={
                    model.TargetHandle(
                        node="child", port="wrong_port"
                    ): model.SourceHandle(node=None, port="x"),
                    # 'wrong_port' doesn't exist
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
                        inputs=["inp"],
                        outputs=["out"],
                    )
                },
                edges={
                    model.TargetHandle(node="child", port="inp"): model.SourceHandle(
                        node=None, port="x"
                    ),
                    model.TargetHandle(node=None, port="y"): model.SourceHandle(
                        node="child", port="wrong_port"
                    ),
                    # 'wrong_port' doesn't exist
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
                        inputs=["inp"],
                        outputs=["out"],
                    )
                },
                edges={
                    model.TargetHandle(node="child", port="inp"): model.SourceHandle(
                        node=None, port="nonexistent_input"
                    ),
                    # doesn't exist
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
                        inputs=["inp"],
                        outputs=["out"],
                    )
                },
                edges={
                    model.TargetHandle(node="child", port="inp"): model.SourceHandle(
                        node=None, port="x"
                    ),
                    model.TargetHandle(
                        node=None, port="nonexistent_output"
                    ): model.SourceHandle(
                        node="child", port="out"
                    ),  # doesn't exist
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
                model.TargetHandle(node="node1", port="in1"): model.SourceHandle(
                    node=None, port="a"
                ),
                model.TargetHandle(node="node1", port="in2"): model.SourceHandle(
                    node=None, port="b"
                ),
                model.TargetHandle(node=None, port="x"): model.SourceHandle(
                    node="node1", port="out1"
                ),
                model.TargetHandle(node=None, port="y"): model.SourceHandle(
                    node="node1", port="out2"
                ),
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
                model.TargetHandle(node="leaf", port="x"): model.SourceHandle(
                    node=None, port="inner_in"
                ),
                model.TargetHandle(node=None, port="inner_out"): model.SourceHandle(
                    node="leaf", port="y"
                ),
            },
        )

        outer = model.WorkflowNode(
            inputs=["outer_in"],
            outputs=["outer_out"],
            nodes={"inner": inner},
            edges={
                model.TargetHandle(node="inner", port="inner_in"): model.SourceHandle(
                    node=None, port="outer_in"
                ),
                model.TargetHandle(node=None, port="outer_out"): model.SourceHandle(
                    node="inner", port="inner_out"
                ),
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
                model.TargetHandle(node="leaf", port="x"): model.SourceHandle(
                    node=None, port="inner_in"
                ),
                model.TargetHandle(node=None, port="inner_out"): model.SourceHandle(
                    node="leaf", port="y"
                ),
            },
        )

        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.WorkflowNode(
                inputs=["outer_in"],
                outputs=["outer_out"],
                nodes={"inner": inner},
                edges={
                    model.TargetHandle(
                        node="inner", port="wrong_port"
                    ): model.SourceHandle(
                        node=None, port="outer_in"
                    ),  # doesn't exist
                    model.TargetHandle(node=None, port="outer_out"): model.SourceHandle(
                        node="inner", port="inner_out"
                    ),
                },
            )
        self.assertIn("has no input port", str(ctx.exception))
        self.assertIn("wrong_port", str(ctx.exception))


class TestWorkflowNodeReservedNames(unittest.TestCase):
    """Tests for reserved node name validation."""

    def test_reserved_node_names(self):
        """Node labels cannot use reserved names."""
        test_cases = [
            ("for", "Python keyword"),
            ("while", "Python keyword"),
            *[(reserved, "reserved name") for reserved in model.RESERVED_NAMES],
            ("1invalid", "not an identifier"),
            ("my-var", "not an identifier"),
            ("my var", "not an identifier"),
            ("", "not an identifier"),
        ]

        for invalid_label, reason in test_cases:
            with self.subTest(label=invalid_label, reason=reason):
                with self.assertRaises(pydantic.ValidationError) as ctx:
                    model.WorkflowNode(
                        inputs=["a"],
                        outputs=["b"],
                        nodes={
                            invalid_label: model.AtomicNode(
                                fully_qualified_name="m.f",
                                inputs=[],
                                outputs=[],
                            )
                        },
                        edges={},
                    )
                exc_str = str(ctx.exception)
                self.assertIn(invalid_label, exc_str)


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
                        inputs=["inp", "feedback"],
                        outputs=["out"],
                    ),
                    "b": model.AtomicNode(
                        fully_qualified_name="m.g",
                        inputs=["inp"],
                        outputs=["out", "out2"],
                    ),
                },
                edges={
                    model.TargetHandle(node="a", port="inp"): model.SourceHandle(
                        node=None, port="x"
                    ),
                    model.TargetHandle(node="b", port="inp"): model.SourceHandle(
                        node="a", port="out"
                    ),
                    model.TargetHandle(node="a", port="feedback"): model.SourceHandle(
                        node="b", port="out"
                    ),  # creates cycle
                    model.TargetHandle(node=None, port="y"): model.SourceHandle(
                        node="b", port="out2"
                    ),
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
                        inputs=["inp", "feedback"],
                        outputs=["out", "out2"],
                    )
                },
                edges={
                    model.TargetHandle(node="a", port="inp"): model.SourceHandle(
                        node=None, port="x"
                    ),
                    model.TargetHandle(node="a", port="feedback"): model.SourceHandle(
                        node="a", port="out"
                    ),  # self-loop
                    model.TargetHandle(node=None, port="y"): model.SourceHandle(
                        node="a", port="out2"
                    ),
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
                    inputs=["inp"],
                    outputs=["out"],
                ),
                "b": model.AtomicNode(
                    fully_qualified_name="m.g",
                    inputs=["inp"],
                    outputs=["out"],
                ),
                "c": model.AtomicNode(
                    fully_qualified_name="m.h",
                    inputs=["inp"],
                    outputs=["out"],
                ),
            },
            edges={
                model.TargetHandle(node="a", port="inp"): model.SourceHandle(
                    node=None, port="x"
                ),
                model.TargetHandle(node="b", port="inp"): model.SourceHandle(
                    node="a", port="out"
                ),
                model.TargetHandle(node="c", port="inp"): model.SourceHandle(
                    node="b", port="out"
                ),
                model.TargetHandle(node=None, port="y"): model.SourceHandle(
                    node="c", port="out"
                ),
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
                            inputs=["inp"],
                            outputs=["out"],
                        )
                    },
                    edges={
                        model.TargetHandle(node="leaf", port="inp"): model.SourceHandle(
                            node=None, port="a"
                        ),
                        model.TargetHandle(node=None, port="b"): model.SourceHandle(
                            node="leaf", port="out"
                        ),
                    },
                ),
            },
            edges={
                model.TargetHandle(node="inner", port="a"): model.SourceHandle(
                    node=None, port="x"
                ),
                model.TargetHandle(node=None, port="z"): model.SourceHandle(
                    node="inner", port="b"
                ),
            },
        )
        self.assertIsInstance(outer.nodes["inner"], model.WorkflowNode)

    def test_nested_invalid_fqn_bubbles_up(self):
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
                    inputs=["inp"],
                    outputs=["out"],
                )
            },
            edges={
                model.TargetHandle(node="n", port="inp"): model.SourceHandle(
                    node=None, port="x"
                ),
                model.TargetHandle(node=None, port="y"): model.SourceHandle(
                    node="n", port="out"
                ),
            },
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
                    inputs=["inp"],
                    outputs=["out"],
                )
            },
            edges={
                model.TargetHandle(node="n", port="inp"): model.SourceHandle(
                    node=None, port="x"
                ),
                model.TargetHandle(node=None, port="y"): model.SourceHandle(
                    node="n", port="out"
                ),
            },
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
                model.TargetHandle(node="a", port="i1"): model.SourceHandle(
                    node=None, port="x"
                ),
                model.TargetHandle(node="a", port="i2"): model.SourceHandle(
                    node=None, port="y"
                ),
                model.TargetHandle(node=None, port="z"): model.SourceHandle(
                    node="a", port="o1"
                ),
                model.TargetHandle(node=None, port="w"): model.SourceHandle(
                    node="a", port="o2"
                ),
            },
        )
        data = original.model_dump(mode="python")

        # Verify edges is a dict, not a list
        self.assertIsInstance(data["edges"], dict)

        # Verify tuple keys are stringified
        self.assertIn("a.i1", data["edges"])
        self.assertIn("x", data["edges"].values())

        # Verify values match
        self.assertEqual(data["edges"]["a.i1"], "x")
        self.assertEqual(data["edges"]["z"], "a.o1")

    def test_workflow_json_mode_uses_list_structure(self):
        """JSON mode should convert handles to simple strings."""
        original = model.WorkflowNode(
            inputs=["x"],
            outputs=["y"],
            nodes={
                "n": model.AtomicNode(
                    fully_qualified_name="m.f",
                    inputs=["inp"],
                    outputs=["out"],
                )
            },
            edges={
                model.TargetHandle(node="n", port="inp"): model.SourceHandle(
                    node=None, port="x"
                ),
                model.TargetHandle(node=None, port="y"): model.SourceHandle(
                    node="n", port="out"
                ),
            },
        )
        data = original.model_dump(mode="json")

        # Verify edges is a dict
        self.assertIsInstance(data["edges"], dict)
        self.assertEqual(len(data["edges"]), 2)
        # Each element should be a [key, value] pair
        for k, v in data["edges"].items():
            self.assertIsInstance(k, str)
            self.assertIsInstance(v, str)

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
