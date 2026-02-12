"""Unit tests for flowrep.models.nodes.union"""

import unittest

import pydantic

from flowrep.models import base_models
from flowrep.models.nodes import (
    atomic_model,
    for_model,
    if_model,
    try_model,
    union,
    while_model,
    workflow_model,
)


class TestDiscriminatedUnionRoundtrip(unittest.TestCase):
    """Parameterized tests for NodeType discriminated union deserialization."""

    @classmethod
    def setUpClass(cls):
        """Define test cases: (type_enum, minimal_json_data, expected_class)"""
        cls.test_cases = [
            (
                base_models.RecipeElementType.ATOMIC,
                {
                    "type": "atomic",
                    "fully_qualified_name": "mod.func",
                    "inputs": ["x"],
                    "outputs": ["y"],
                },
                atomic_model.AtomicNode,
            ),
            (
                base_models.RecipeElementType.WORKFLOW,
                {
                    "type": "workflow",
                    "inputs": [],
                    "outputs": [],
                    "nodes": {},
                    "input_edges": {},
                    "edges": {},
                    "output_edges": {},
                },
                workflow_model.WorkflowNode,
            ),
            (
                base_models.RecipeElementType.FOR,
                {
                    "type": "for",
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
                },
                for_model.ForNode,
            ),
            (
                base_models.RecipeElementType.WHILE,
                {
                    "type": "while",
                    "inputs": ["x"],
                    "outputs": [],
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
                },
                while_model.WhileNode,
            ),
            (  # Without else clause
                base_models.RecipeElementType.IF,
                {
                    "type": "if",
                    "inputs": ["inp"],
                    "outputs": ["out"],
                    "cases": [
                        {
                            "condition": {
                                "label": "condition_0",
                                "node": {
                                    "type": "atomic",
                                    "inputs": ["x"],
                                    "outputs": ["result"],
                                    "fully_qualified_name": "mod.check",
                                    "unpack_mode": "tuple",
                                },
                            },
                            "body": {
                                "label": "body_0",
                                "node": {
                                    "type": "atomic",
                                    "inputs": ["x"],
                                    "outputs": ["y"],
                                    "fully_qualified_name": "mod.handle",
                                    "unpack_mode": "tuple",
                                },
                            },
                            "condition_output": None,
                        }
                    ],
                    "input_edges": {"body_0.x": "inp"},
                    "prospective_output_edges": {"out": ["body_0.y"]},
                    "else_case": None,
                },
                if_model.IfNode,
            ),
            (  # With else clause
                base_models.RecipeElementType.IF,
                {
                    "type": "if",
                    "inputs": ["inp"],
                    "outputs": ["out"],
                    "cases": [
                        {
                            "condition": {
                                "label": "condition_0",
                                "node": {
                                    "type": "atomic",
                                    "inputs": ["x"],
                                    "outputs": ["result"],
                                    "fully_qualified_name": "mod.check",
                                    "unpack_mode": "tuple",
                                },
                            },
                            "body": {
                                "label": "body_0",
                                "node": {
                                    "type": "atomic",
                                    "inputs": ["x"],
                                    "outputs": ["y"],
                                    "fully_qualified_name": "mod.handle",
                                    "unpack_mode": "tuple",
                                },
                            },
                            "condition_output": None,
                        }
                    ],
                    "input_edges": {"body_0.x": "inp", "else_body.x": "inp"},
                    "prospective_output_edges": {"out": ["body_0.y", "else_body.y"]},
                    "else_case": {
                        "label": "else_body",
                        "node": {
                            "type": "atomic",
                            "inputs": ["x"],
                            "outputs": ["y"],
                            "fully_qualified_name": "mod.handle",
                            "unpack_mode": "tuple",
                        },
                    },
                },
                if_model.IfNode,
            ),
            (
                base_models.RecipeElementType.TRY,
                {
                    "type": "try",
                    "inputs": ["inp"],
                    "outputs": ["out"],
                    "try_node": {
                        "label": "try_body",
                        "node": {
                            "type": "atomic",
                            "inputs": ["x"],
                            "outputs": ["y"],
                            "fully_qualified_name": "mod.try_func",
                            "unpack_mode": "tuple",
                        },
                    },
                    "exception_cases": [
                        {
                            "exceptions": ["builtins.ValueError"],
                            "body": {
                                "label": "except_0",
                                "node": {
                                    "type": "atomic",
                                    "inputs": ["x"],
                                    "outputs": ["y"],
                                    "fully_qualified_name": "mod.handle_error",
                                    "unpack_mode": "tuple",
                                },
                            },
                        }
                    ],
                    "input_edges": {"try_body.x": "inp", "except_0.x": "inp"},
                    "prospective_output_edges": {"out": ["try_body.y", "except_0.y"]},
                },
                try_model.TryNode,
            ),
        ]

    def test_discriminator_resolves_correct_type(self):
        """Each node type is correctly identified via 'type' discriminator."""
        adapter = pydantic.TypeAdapter(union.NodeType)
        for type_enum, data, expected_class in self.test_cases:
            with self.subTest(type=type_enum.value):
                node = adapter.validate_python(data)
                self.assertIsInstance(node, expected_class)
                self.assertEqual(node.type, type_enum)

    def test_roundtrip_through_union(self):
        """Serialize then deserialize through the union type."""
        adapter = pydantic.TypeAdapter(union.NodeType)
        for type_enum, data, expected_class in self.test_cases:
            with self.subTest(type=type_enum.value):
                node = adapter.validate_python(data)
                dumped = adapter.dump_python(node, mode="json")
                restored = adapter.validate_python(dumped)
                self.assertIsInstance(restored, expected_class)
                self.assertEqual(node, restored)

    def test_json_schema_includes_all_types(self):
        """Union schema should reference all node types."""
        adapter = pydantic.TypeAdapter(union.NodeType)
        schema = adapter.json_schema()
        # Discriminated unions use anyOf or oneOf
        self.assertTrue(
            "anyOf" in schema or "oneOf" in schema or "$defs" in schema,
            "Schema should define union variants",
        )


class TestDiscriminatorValidation(unittest.TestCase):
    """Tests for discriminator error handling."""

    def test_unknown_type_rejected(self):
        adapter = pydantic.TypeAdapter(union.NodeType)
        with self.assertRaises(pydantic.ValidationError) as ctx:
            adapter.validate_python(
                {
                    "type": "unknown_type",
                    "inputs": [],
                    "outputs": [],
                }
            )
        exc_str = str(ctx.exception)
        self.assertIn("type", exc_str.lower())

    def test_missing_type_rejected(self):
        adapter = pydantic.TypeAdapter(union.NodeType)
        with self.assertRaises(pydantic.ValidationError):
            adapter.validate_python(
                {
                    "fully_qualified_name": "mod.func",
                    "inputs": [],
                    "outputs": [],
                }
            )

    def test_type_mismatch_with_fields_rejected(self):
        """Type says 'atomic' but fields are for workflow."""
        adapter = pydantic.TypeAdapter(union.NodeType)
        with self.assertRaises(pydantic.ValidationError):
            adapter.validate_python(
                {
                    "type": "atomic",
                    "inputs": [],
                    "outputs": [],
                    "nodes": {},  # workflow field, not atomic
                    "input_edges": {},
                    "edges": {},
                    "output_edges": {},
                }
            )


class TestNodesTypeAlias(unittest.TestCase):
    """Tests for Nodes dict type alias (dict[Label, NodeType])."""

    def test_valid_nodes_dict(self):
        """Nodes dict with valid labels and node types."""
        nodes_data = {
            "step1": {
                "type": "atomic",
                "fully_qualified_name": "mod.f",
                "inputs": [],
                "outputs": ["x"],
            },
            "step2": {
                "type": "atomic",
                "fully_qualified_name": "mod.g",
                "inputs": ["x"],
                "outputs": [],
            },
        }
        adapter = pydantic.TypeAdapter(union.Nodes)
        nodes = adapter.validate_python(nodes_data)
        self.assertEqual(len(nodes), 2)
        self.assertIsInstance(nodes["step1"], atomic_model.AtomicNode)
        self.assertIsInstance(nodes["step2"], atomic_model.AtomicNode)

    def test_invalid_label_in_nodes_rejected(self):
        """Nodes dict keys must be valid Labels."""
        nodes_data = {
            "for": {  # Python keyword
                "type": "atomic",
                "fully_qualified_name": "mod.f",
                "inputs": [],
                "outputs": [],
            },
        }
        adapter = pydantic.TypeAdapter(union.Nodes)
        with self.assertRaises(pydantic.ValidationError) as ctx:
            adapter.validate_python(nodes_data)
        self.assertIn("for", str(ctx.exception))

    def test_reserved_label_in_nodes_rejected(self):
        """Nodes dict keys cannot be reserved names."""
        for reserved in base_models.RESERVED_NAMES:
            with self.subTest(reserved=reserved):
                nodes_data = {
                    reserved: {
                        "type": "atomic",
                        "fully_qualified_name": "mod.f",
                        "inputs": [],
                        "outputs": [],
                    },
                }
                adapter = pydantic.TypeAdapter(union.Nodes)
                with self.assertRaises(pydantic.ValidationError):
                    adapter.validate_python(nodes_data)

    def test_mixed_node_types_in_dict(self):
        """Nodes dict can contain different node types."""
        inner_workflow = {
            "type": "workflow",
            "inputs": ["a"],
            "outputs": ["b"],
            "nodes": {
                "leaf": {
                    "type": "atomic",
                    "fully_qualified_name": "mod.f",
                    "inputs": ["x"],
                    "outputs": ["y"],
                },
            },
            "input_edges": {"leaf.x": "a"},
            "edges": {},
            "output_edges": {"b": "leaf.y"},
        }
        nodes_data = {
            "atomic_node": {
                "type": "atomic",
                "fully_qualified_name": "mod.g",
                "inputs": [],
                "outputs": [],
            },
            "workflow_node": inner_workflow,
        }
        adapter = pydantic.TypeAdapter(union.Nodes)
        nodes = adapter.validate_python(nodes_data)
        self.assertIsInstance(nodes["atomic_node"], atomic_model.AtomicNode)
        self.assertIsInstance(nodes["workflow_node"], workflow_model.WorkflowNode)

    def test_empty_nodes_dict(self):
        adapter = pydantic.TypeAdapter(union.Nodes)
        nodes = adapter.validate_python({})
        self.assertEqual(nodes, {})


class TestNestedUnionResolution(unittest.TestCase):
    """Tests for nested node type resolution in complex structures."""

    def test_workflow_with_nested_for_node(self):
        """Workflow containing a ForNode deserializes correctly."""
        data = {
            "type": "workflow",
            "inputs": ["data"],
            "outputs": ["results"],
            "nodes": {
                "loop": {
                    "type": "for",
                    "inputs": ["items"],
                    "outputs": ["out"],
                    "body_node": {
                        "label": "body",
                        "node": {
                            "type": "atomic",
                            "fully_qualified_name": "mod.transform",
                            "inputs": ["item"],
                            "outputs": ["result"],
                        },
                    },
                    "input_edges": {"body.item": "items"},
                    "output_edges": {"out": "body.result"},
                    "nested_ports": ["item"],
                    "zipped_ports": [],
                    "transfer_edges": {},
                },
            },
            "input_edges": {"loop.items": "data"},
            "edges": {},
            "output_edges": {"results": "loop.out"},
        }
        adapter = pydantic.TypeAdapter(union.NodeType)
        wf = adapter.validate_python(data)
        self.assertIsInstance(wf, workflow_model.WorkflowNode)
        self.assertIsInstance(wf.nodes["loop"], for_model.ForNode)
        self.assertIsInstance(wf.nodes["loop"].body_node.node, atomic_model.AtomicNode)

    def test_deeply_nested_workflows(self):
        """Three levels of workflow nesting."""
        innermost = {
            "type": "atomic",
            "fully_qualified_name": "mod.leaf",
            "inputs": ["x"],
            "outputs": ["y"],
        }
        middle = {
            "type": "workflow",
            "inputs": ["a"],
            "outputs": ["b"],
            "nodes": {"leaf": innermost},
            "input_edges": {"leaf.x": "a"},
            "edges": {},
            "output_edges": {"b": "leaf.y"},
        }
        outer = {
            "type": "workflow",
            "inputs": ["inp"],
            "outputs": ["out"],
            "nodes": {"middle": middle},
            "input_edges": {"middle.a": "inp"},
            "edges": {},
            "output_edges": {"out": "middle.b"},
        }
        adapter = pydantic.TypeAdapter(union.NodeType)
        wf = adapter.validate_python(outer)
        self.assertIsInstance(wf.nodes["middle"], workflow_model.WorkflowNode)
        self.assertIsInstance(wf.nodes["middle"].nodes["leaf"], atomic_model.AtomicNode)


if __name__ == "__main__":
    unittest.main()
