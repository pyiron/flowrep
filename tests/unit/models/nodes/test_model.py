"""Unit tests for flowrep.model"""

import unittest
from typing import Literal

import pydantic

from flowrep.models import edge_models
from flowrep.models.nodes import atomic_model, base_models, union, workflow_model


class TestNodeModel(unittest.TestCase):
    """Tests for input/output uniqueness validation on NodeModel base class."""

    def test_duplicate_inputs_rejected(self):
        """Any NodeModel subclass should reject duplicate inputs."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            atomic_model.AtomicNode(
                fully_qualified_name="mod.func",
                inputs=["x", "y", "x"],  # duplicate 'x'
                outputs=["z"],
            )
        self.assertIn("unique", str(ctx.exception).lower())
        self.assertIn("x", str(ctx.exception))

    def test_duplicate_outputs_rejected(self):
        """Any NodeModel subclass should reject duplicate outputs."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            workflow_model.WorkflowNode(
                inputs=["a"],
                outputs=["x", "y", "x"],  # duplicate 'x'
                nodes={},
                input_edges={},
                edges={},
                output_edges={},
            )
        self.assertIn("unique", str(ctx.exception).lower())
        self.assertIn("x", str(ctx.exception))

    def test_unique_inputs_outputs_preserved_order(self):
        """Unique inputs/outputs should preserve declaration order."""
        node = atomic_model.AtomicNode(
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
            *[(reserved, "reserved name") for reserved in base_models.RESERVED_NAMES],
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
                        atomic_model.AtomicNode(**kwargs)

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
        node = atomic_model.AtomicNode(
            fully_qualified_name="mod.func",
            inputs=["x", "y_1", "_private", "camelCase"],
            outputs=["result", "status_code"],
        )
        self.assertEqual(len(node.inputs), 4)
        self.assertEqual(len(node.outputs), 2)


class TestNodeTypeImmutability(unittest.TestCase):
    """Tests that the 'type' field is immutable on NodeModel subclasses."""

    def test_type_field_cannot_be_overridden_at_construction(self):
        """AtomicNode should reject type override during instantiation."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            atomic_model.AtomicNode(
                type=base_models.RecipeElementType.WORKFLOW,  # Wrong type
                fully_qualified_name="mod.func",
                inputs=["x"],
                outputs=["y"],
            )
        exc_str = str(ctx.exception)
        self.assertIn("Input should be", exc_str)
        self.assertIn(base_models.RecipeElementType.ATOMIC.value, exc_str)

    def test_type_field_cannot_be_mutated_after_construction(self):
        """AtomicNode should reject mutation of type field."""
        node = atomic_model.AtomicNode(
            fully_qualified_name="mod.func",
            inputs=["x"],
            outputs=["y"],
        )
        with self.assertRaises(pydantic.ValidationError) as ctx:
            node.type = base_models.RecipeElementType.WORKFLOW
        exc_str = str(ctx.exception)
        self.assertIn("frozen", exc_str.lower())

    def test_subclass_must_provide_type_default(self):
        """NodeModel subclasses must provide a default value for 'type'."""
        with self.assertRaises(TypeError) as ctx:

            class BadNode(base_models.NodeModel):
                type: Literal[base_models.RecipeElementType.ATOMIC]  # No default
                inputs: list[str]
                outputs: list[str]

        exc_str = str(ctx.exception)
        self.assertIn("BadNode", exc_str)
        self.assertIn("default value for 'type'", exc_str)

    def test_subclass_must_freeze_type_field(self):
        """NodeModel subclasses must mark 'type' as frozen."""
        with self.assertRaises(TypeError) as ctx:

            class BadNode(base_models.NodeModel):
                type: Literal[base_models.RecipeElementType.ATOMIC] = (
                    base_models.RecipeElementType.ATOMIC
                )  # Not frozen
                inputs: list[str]
                outputs: list[str]

        exc_str = str(ctx.exception)
        self.assertIn("BadNode", exc_str)
        self.assertIn("frozen", exc_str)

    def test_subclass_must_redefine_type_field(self):
        """NodeModel subclasses must redefine 'type' field, not inherit base definition."""
        with self.assertRaises(TypeError) as ctx:

            class BadNode(base_models.NodeModel):
                # Doesn't mention 'type' at all
                inputs: list[str]
                outputs: list[str]

        exc_str = str(ctx.exception)
        self.assertIn("BadNode", exc_str)
        self.assertIn("default value for 'type'", exc_str)


class TestAtomicNode(unittest.TestCase):
    """Tests for AtomicNode validation."""

    def test_valid_fqn(self):
        node = atomic_model.AtomicNode(
            fully_qualified_name="module.func",
            inputs=[],
            outputs=[],
        )
        self.assertEqual(node.fully_qualified_name, "module.func")
        self.assertEqual(node.type, base_models.RecipeElementType.ATOMIC)

    def test_valid_fqn_deep(self):
        node = atomic_model.AtomicNode(
            fully_qualified_name="a.b.c.d",
            inputs=[],
            outputs=[],
        )
        self.assertEqual(node.fully_qualified_name, "a.b.c.d")

    def test_fqn_no_period(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            atomic_model.AtomicNode(
                fully_qualified_name="noDot",
                inputs=[],
                outputs=[],
            )
        self.assertIn("at least one period", str(ctx.exception))

    def test_fqn_empty_string(self):
        with self.assertRaises(pydantic.ValidationError):
            atomic_model.AtomicNode(
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
                atomic_model.AtomicNode(
                    fully_qualified_name=bad,
                    inputs=[],
                    outputs=[],
                )


class TestAtomicNodeUnpacking(unittest.TestCase):
    """Tests for unpack_mode validation."""

    def test_default_unpack_mode(self):
        """Default unpack_mode should be 'tuple'."""
        node = atomic_model.AtomicNode(
            fully_qualified_name="mod.func",
            inputs=["x"],
            outputs=["a", "b"],
        )
        self.assertEqual(node.unpack_mode, atomic_model.UnpackMode.TUPLE)

    def test_tuple_mode_multiple_outputs(self):
        """Multiple outputs allowed with unpack_mode='tuple'."""
        node = atomic_model.AtomicNode(
            fully_qualified_name="mod.func",
            inputs=[],
            outputs=["a", "b", "c"],
            unpack_mode=atomic_model.UnpackMode.TUPLE,
        )
        self.assertEqual(len(node.outputs), 3)
        self.assertEqual(node.unpack_mode, atomic_model.UnpackMode.TUPLE)

    def test_dataclass_mode_multiple_outputs(self):
        """Multiple outputs allowed with unpack_mode='dataclass'."""
        node = atomic_model.AtomicNode(
            fully_qualified_name="mod.func",
            inputs=[],
            outputs=["a", "b", "c"],
            unpack_mode=atomic_model.UnpackMode.DATACLASS,
        )
        self.assertEqual(len(node.outputs), 3)
        self.assertEqual(node.unpack_mode, atomic_model.UnpackMode.DATACLASS)

    def test_none_mode_multiple_outputs_rejected(self):
        """Multiple outputs rejected when unpack_mode='none'."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            atomic_model.AtomicNode(
                fully_qualified_name="mod.func",
                inputs=[],
                outputs=["a", "b"],
                unpack_mode=atomic_model.UnpackMode.NONE,
            )
        self.assertIn("exactly one element", str(ctx.exception))
        self.assertIn(
            f"unpack_mode={atomic_model.UnpackMode.NONE.value}", str(ctx.exception)
        )

    def test_none_mode_single_output_valid(self):
        """Single output valid with unpack_mode='none'."""
        node = atomic_model.AtomicNode(
            fully_qualified_name="mod.func",
            inputs=["x"],
            outputs=["result"],
            unpack_mode=atomic_model.UnpackMode.NONE,
        )
        self.assertEqual(node.outputs, ["result"])
        self.assertEqual(node.unpack_mode, atomic_model.UnpackMode.NONE)

    def test_none_mode_zero_outputs_valid(self):
        """Zero outputs valid with unpack_mode='none'."""
        node = atomic_model.AtomicNode(
            fully_qualified_name="mod.func",
            inputs=[],
            outputs=[],
            unpack_mode=atomic_model.UnpackMode.NONE,
        )
        self.assertEqual(len(node.outputs), 0)
        self.assertEqual(node.unpack_mode, atomic_model.UnpackMode.NONE)

    def test_tuple_mode_zero_outputs_valid(self):
        """Zero outputs valid with unpack_mode='tuple'."""
        node = atomic_model.AtomicNode(
            fully_qualified_name="mod.func",
            inputs=["x"],
            outputs=[],
            unpack_mode=atomic_model.UnpackMode.TUPLE,
        )
        self.assertEqual(len(node.outputs), 0)

    def test_all_unpack_modes_valid_literal(self):
        """All three unpack modes should be valid."""
        for mode in ["none", "tuple", "dataclass"]:
            node = atomic_model.AtomicNode(
                fully_qualified_name="mod.func",
                inputs=[],
                outputs=["out"],
                unpack_mode=mode,
            )
            self.assertEqual(node.unpack_mode, mode)


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
        self.assertIn("not a workflow input", str(ctx.exception))

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
        self.assertIn("not a child node", str(ctx.exception))

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
        self.assertIn("has no input port", str(ctx.exception))
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
        self.assertIn("not a workflow output", str(ctx.exception))

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
        self.assertIn("not a child node", str(ctx.exception))

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
        self.assertIn("has no output port", str(ctx.exception))
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
                outputs=["y"],
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
        self.assertIn("not a child node", str(ctx.exception))

    def test_internal_edge_invalid_target_node(self):
        """Internal edge to nonexistent target node."""
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
                edges={
                    edge_models.TargetHandle(
                        node="nonexistent", port="inp"
                    ): edge_models.SourceHandle(node="child", port="out"),
                },
                output_edges={},
            )
        self.assertIn("not a child node", str(ctx.exception))

    def test_internal_edge_invalid_source_port(self):
        """Internal edge from nonexistent source port."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            workflow_model.WorkflowNode(
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
                input_edges={},
                edges={
                    edge_models.TargetHandle(
                        node="b", port="inp"
                    ): edge_models.SourceHandle(node="a", port="wrong"),
                },
                output_edges={},
            )
        self.assertIn("has no output port", str(ctx.exception))

    def test_internal_edge_invalid_target_port(self):
        """Internal edge to nonexistent target port."""
        with self.assertRaises(pydantic.ValidationError) as ctx:
            workflow_model.WorkflowNode(
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
                input_edges={},
                edges={
                    edge_models.TargetHandle(
                        node="b", port="wrong"
                    ): edge_models.SourceHandle(node="a", port="out"),
                },
                output_edges={},
            )
        self.assertIn("has no input port", str(ctx.exception))


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
        self.assertIn("has no input port", str(ctx.exception))
        self.assertIn("wrong_port", str(ctx.exception))


class TestSerialization(unittest.TestCase):
    """Tests for JSON and Python mode serialization roundtrip."""

    def test_atomic_json_roundtrip(self):
        original = atomic_model.AtomicNode(
            fully_qualified_name="mod.func",
            inputs=["a"],
            outputs=["b"],
        )
        data = original.model_dump(mode="json")
        restored = atomic_model.AtomicNode.model_validate(data)
        self.assertEqual(original, restored)

    def test_atomic_python_roundtrip(self):
        original = atomic_model.AtomicNode(
            fully_qualified_name="mod.func",
            inputs=["a"],
            outputs=["b"],
        )
        data = original.model_dump(mode="python")
        restored = atomic_model.AtomicNode.model_validate(data)
        self.assertEqual(original, restored)

    def test_workflow_json_roundtrip(self):
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
        data = original.model_dump(mode="json")
        restored = workflow_model.WorkflowNode.model_validate(data)
        self.assertEqual(original.inputs, restored.inputs)
        self.assertEqual(original.outputs, restored.outputs)
        self.assertEqual(len(original.nodes), len(restored.nodes))
        self.assertEqual(original.input_edges, restored.input_edges)
        self.assertEqual(original.edges, restored.edges)
        self.assertEqual(original.output_edges, restored.output_edges)

    def test_workflow_python_roundtrip(self):
        """Python mode roundtrip for WorkflowNode."""
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
        data = original.model_dump(mode="python")
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

    def test_discriminated_union_roundtrip(self):
        """Ensure type discriminator works for polymorphic deserialization."""
        data = {
            "type": base_models.RecipeElementType.ATOMIC,
            "fully_qualified_name": "a.b",
            "inputs": ["x"],
            "outputs": ["y"],
        }
        node = pydantic.TypeAdapter(union.NodeType).validate_python(data)
        self.assertIsInstance(node, atomic_model.AtomicNode)

        data = {
            "type": base_models.RecipeElementType.WORKFLOW,
            "inputs": [],
            "outputs": [],
            "nodes": {},
            "input_edges": {},
            "edges": {},
            "output_edges": {},
        }
        node = pydantic.TypeAdapter(union.NodeType).validate_python(data)
        self.assertIsInstance(node, workflow_model.WorkflowNode)


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


if __name__ == "__main__":
    unittest.main()
