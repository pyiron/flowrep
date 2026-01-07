"""Unit tests for flowrep.model"""

import unittest

import pydantic

from flowrep import model


class TestAtomicNode(unittest.TestCase):
    """Tests for AtomicNode validation."""

    def test_valid_fnc(self):
        node = model.AtomicNode(fully_qualified_name="module.func")
        self.assertEqual(node.fully_qualified_name, "module.func")
        self.assertEqual(node.type, "atomic")

    def test_valid_fnc_deep(self):
        node = model.AtomicNode(fully_qualified_name="a.b.c.d")
        self.assertEqual(node.fully_qualified_name, "a.b.c.d")

    def test_fnc_no_period(self):
        with self.assertRaises(pydantic.ValidationError) as ctx:
            model.AtomicNode(fully_qualified_name="noDot")
        self.assertIn("at least one period", str(ctx.exception))

    def test_fnc_empty_string(self):
        with self.assertRaises(pydantic.ValidationError):
            model.AtomicNode(fully_qualified_name="")

    def test_fnc_empty_part(self):
        """e.g., 'module.' or '.func' or 'a..b'"""
        for bad in ["module.", ".func", "a..b"]:
            with self.assertRaises(
                pydantic.ValidationError, msg=f"Should reject {bad!r}"
            ):
                model.AtomicNode(fully_qualified_name=bad)


class TestWorkflowNodeEdgeValidation(unittest.TestCase):
    """Tests for WorkflowNode edge validation."""

    def _make_workflow(self, edges):
        """Helper to create a workflow with given edges and one atomic child."""
        return model.WorkflowNode(
            inputs=["x"],
            outputs=["y"],
            nodes={"child": model.AtomicNode(fully_qualified_name="mod.func")},
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


class TestWorkflowNodeReservedNames(unittest.TestCase):
    """Tests for reserved node name validation."""

    def test_reserved_name_inputs(self):
        for reserved in model.RESERVED_NAMES:
            with self.assertRaises(pydantic.ValidationError) as ctx:
                model.WorkflowNode(
                    inputs=["a"],
                    outputs=["b"],
                    nodes={reserved: model.AtomicNode(fully_qualified_name="m.f")},
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
                    "a": model.AtomicNode(fully_qualified_name="m.f"),
                    "b": model.AtomicNode(fully_qualified_name="m.g"),
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
                nodes={"a": model.AtomicNode(fully_qualified_name="m.f")},
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
                "a": model.AtomicNode(fully_qualified_name="m.f"),
                "b": model.AtomicNode(fully_qualified_name="m.g"),
                "c": model.AtomicNode(fully_qualified_name="m.h"),
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
                    nodes={"leaf": model.AtomicNode(fully_qualified_name="m.f")},
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
                        nodes={"bad": model.AtomicNode(fully_qualified_name="noDot")},
                        edges={},
                    ),
                },
                edges={},
            )


class TestSerialization(unittest.TestCase):
    """Tests for JSON serialization roundtrip."""

    def test_atomic_roundtrip(self):
        original = model.AtomicNode(fully_qualified_name="mod.func")
        data = original.model_dump(mode="json")
        restored = model.AtomicNode.model_validate(data)
        self.assertEqual(original, restored)

    def test_workflow_roundtrip(self):
        original = model.WorkflowNode(
            inputs=["x"],
            outputs=["y"],
            nodes={"n": model.AtomicNode(fully_qualified_name="m.f")},
            edges={("n", "in"): "x", "y": ("n", "out")},
        )
        data = original.model_dump(mode="json")
        restored = model.WorkflowNode.model_validate(data)
        self.assertEqual(original.inputs, restored.inputs)
        self.assertEqual(original.outputs, restored.outputs)
        self.assertEqual(len(original.nodes), len(restored.nodes))

    def test_discriminated_union_roundtrip(self):
        """Ensure type discriminator works for polymorphic deserialization."""
        data = {"type": "atomic", "fully_qualified_name": "a.b"}
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
            nodes={"n": model.AtomicNode(fully_qualified_name="m.f")},
            edges={},
        )
        self.assertEqual(wf.inputs, [])


if __name__ == "__main__":
    unittest.main()
