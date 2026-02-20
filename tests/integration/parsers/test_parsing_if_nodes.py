import unittest

from flowrep.models.nodes import workflow_model
from flowrep.models.parsers import atomic_parser, workflow_parser


# Reusable atomics (same as other test files)
@atomic_parser.atomic
def my_cond(m, n):
    return m < n


@atomic_parser.atomic
def my_add(a, b):
    return a + b


@atomic_parser.atomic
def my_mul(a, b):
    return a * b


@atomic_parser.atomic
def my_identity(x):
    return x


# 1. Simple if/else — single symbol assigned in both branches
def simple_if_else(x, y):
    if my_cond(x, y):  # noqa: SIM108
        z = my_add(x, y)
    else:
        z = my_mul(x, y)
    return z


simple_node = workflow_model.WorkflowNode.model_validate(
    {
        "type": "workflow",
        "inputs": ["x", "y"],
        "outputs": ["z"],
        "nodes": {
            "if_0": {
                "type": "if",
                "inputs": ["x", "y"],
                "outputs": ["z"],
                "cases": [
                    {
                        "condition": {
                            "label": "condition_0",
                            "node": my_cond.flowrep_recipe,
                        },
                        "body": {
                            "label": "body_0",
                            "node": {
                                "type": "workflow",
                                "inputs": ["x", "y"],
                                "outputs": ["z"],
                                "nodes": {
                                    "my_add_0": my_add.flowrep_recipe,
                                },
                                "input_edges": {"my_add_0.a": "x", "my_add_0.b": "y"},
                                "edges": {},
                                "output_edges": {"z": "my_add_0.output_0"},
                            },
                        },
                        "condition_output": None,
                    },
                ],
                "else_case": {
                    "label": "else_body",
                    "node": {
                        "type": "workflow",
                        "inputs": ["x", "y"],
                        "outputs": ["z"],
                        "nodes": {
                            "my_mul_0": my_mul.flowrep_recipe,
                        },
                        "input_edges": {"my_mul_0.a": "x", "my_mul_0.b": "y"},
                        "edges": {},
                        "output_edges": {"z": "my_mul_0.output_0"},
                    },
                },
                "input_edges": {
                    "condition_0.m": "x",
                    "condition_0.n": "y",
                    "body_0.x": "x",
                    "body_0.y": "y",
                    "else_body.x": "x",
                    "else_body.y": "y",
                },
                "prospective_output_edges": {"z": ["body_0.z", "else_body.z"]},
            }
        },
        "input_edges": {"if_0.x": "x", "if_0.y": "y"},
        "edges": {},
        "output_edges": {"z": "if_0.z"},
        "fully_qualified_name": "integration.parsers.test_parsing_if_nodes.simple_if_else",
    }
)


# 2. if/elif/else chain
def if_elif_else(x, y, flag):
    if my_cond(x, flag):
        z = my_add(x, y)
    elif my_cond(y, flag):
        z = my_mul(x, y)
    else:
        z = my_add(y, flag)
    return z


elif_node = workflow_model.WorkflowNode.model_validate(
    {
        "type": "workflow",
        "inputs": ["x", "y", "flag"],
        "outputs": ["z"],
        "nodes": {
            "if_0": {
                "type": "if",
                "inputs": ["x", "flag", "y"],
                "outputs": ["z"],
                "cases": [
                    {
                        "condition": {
                            "label": "condition_0",
                            "node": my_cond.flowrep_recipe,
                        },
                        "body": {
                            "label": "body_0",
                            "node": {
                                "type": "workflow",
                                "inputs": ["x", "y"],
                                "outputs": ["z"],
                                "nodes": {
                                    "my_add_0": my_add.flowrep_recipe,
                                },
                                "input_edges": {"my_add_0.a": "x", "my_add_0.b": "y"},
                                "edges": {},
                                "output_edges": {"z": "my_add_0.output_0"},
                            },
                        },
                        "condition_output": None,
                    },
                    {
                        "condition": {
                            "label": "condition_1",
                            "node": my_cond.flowrep_recipe,
                        },
                        "body": {
                            "label": "body_1",
                            "node": {
                                "type": "workflow",
                                "inputs": ["x", "y"],
                                "outputs": ["z"],
                                "nodes": {
                                    "my_mul_0": my_mul.flowrep_recipe,
                                },
                                "input_edges": {"my_mul_0.a": "x", "my_mul_0.b": "y"},
                                "edges": {},
                                "output_edges": {"z": "my_mul_0.output_0"},
                            },
                        },
                        "condition_output": None,
                    },
                ],
                "else_case": {
                    "label": "else_body",
                    "node": {
                        "type": "workflow",
                        "inputs": ["y", "flag"],
                        "outputs": ["z"],
                        "nodes": {
                            "my_add_0": my_add.flowrep_recipe,
                        },
                        "input_edges": {"my_add_0.a": "y", "my_add_0.b": "flag"},
                        "edges": {},
                        "output_edges": {"z": "my_add_0.output_0"},
                    },
                },
                "input_edges": {
                    "condition_0.m": "x",
                    "condition_0.n": "flag",
                    "condition_1.m": "y",
                    "condition_1.n": "flag",
                    "body_0.x": "x",
                    "body_0.y": "y",
                    "body_1.x": "x",
                    "body_1.y": "y",
                    "else_body.y": "y",
                    "else_body.flag": "flag",
                },
                "prospective_output_edges": {
                    "z": ["body_0.z", "body_1.z", "else_body.z"]
                },
            }
        },
        "input_edges": {"if_0.x": "x", "if_0.flag": "flag", "if_0.y": "y"},
        "edges": {},
        "output_edges": {"z": "if_0.z"},
        "fully_qualified_name": "integration.parsers.test_parsing_if_nodes.if_elif_else",
    }
)


# 3. if-node embedded between upstream and downstream siblings
def if_with_context(a, b):
    x = my_add(a, b)
    if my_cond(x, b):  # noqa: SIM108
        y = my_mul(x, b)
    else:
        y = my_add(x, b)
    z = my_identity(y)
    return z


context_node = workflow_model.WorkflowNode.model_validate(
    {
        "type": "workflow",
        "inputs": ["a", "b"],
        "outputs": ["z"],
        "nodes": {
            "my_add_0": my_add.flowrep_recipe,
            "if_0": {
                "type": "if",
                "inputs": ["x", "b"],
                "outputs": ["y"],
                "cases": [
                    {
                        "condition": {
                            "label": "condition_0",
                            "node": my_cond.flowrep_recipe,
                        },
                        "body": {
                            "label": "body_0",
                            "node": {
                                "type": "workflow",
                                "inputs": ["x", "b"],
                                "outputs": ["y"],
                                "nodes": {
                                    "my_mul_0": my_mul.flowrep_recipe,
                                },
                                "input_edges": {"my_mul_0.a": "x", "my_mul_0.b": "b"},
                                "edges": {},
                                "output_edges": {"y": "my_mul_0.output_0"},
                            },
                        },
                        "condition_output": None,
                    }
                ],
                "else_case": {
                    "label": "else_body",
                    "node": {
                        "type": "workflow",
                        "inputs": ["x", "b"],
                        "outputs": ["y"],
                        "nodes": {
                            "my_add_0": my_add.flowrep_recipe,
                        },
                        "input_edges": {"my_add_0.a": "x", "my_add_0.b": "b"},
                        "edges": {},
                        "output_edges": {"y": "my_add_0.output_0"},
                    },
                },
                "input_edges": {
                    "condition_0.m": "x",
                    "condition_0.n": "b",
                    "body_0.x": "x",
                    "body_0.b": "b",
                    "else_body.x": "x",
                    "else_body.b": "b",
                },
                "prospective_output_edges": {"y": ["body_0.y", "else_body.y"]},
            },
            "my_identity_0": my_identity.flowrep_recipe,
        },
        "input_edges": {"my_add_0.a": "a", "my_add_0.b": "b", "if_0.b": "b"},
        "edges": {"if_0.x": "my_add_0.output_0", "my_identity_0.x": "if_0.y"},
        "output_edges": {"z": "my_identity_0.x"},
        "fully_qualified_name": "integration.parsers.test_parsing_if_nodes.if_with_context",
    }
)


# 4. Multiple outputs from if branches
def multi_output_if(x, y):
    if my_cond(x, y):
        a = my_add(x, y)
        b = my_mul(x, y)
    else:
        a = my_mul(x, y)
        b = my_add(x, y)
    return a, b


multi_output_node = workflow_model.WorkflowNode.model_validate(
    {
        "type": "workflow",
        "inputs": ["x", "y"],
        "outputs": ["a", "b"],
        "nodes": {
            "if_0": {
                "type": "if",
                "inputs": ["x", "y"],
                "outputs": ["a", "b"],
                "cases": [
                    {
                        "condition": {
                            "label": "condition_0",
                            "node": my_cond.flowrep_recipe,
                        },
                        "body": {
                            "label": "body_0",
                            "node": {
                                "type": "workflow",
                                "inputs": ["x", "y"],
                                "outputs": ["a", "b"],
                                "nodes": {
                                    "my_add_0": my_add.flowrep_recipe,
                                    "my_mul_0": my_mul.flowrep_recipe,
                                },
                                "input_edges": {
                                    "my_add_0.a": "x",
                                    "my_add_0.b": "y",
                                    "my_mul_0.a": "x",
                                    "my_mul_0.b": "y",
                                },
                                "edges": {},
                                "output_edges": {
                                    "a": "my_add_0.output_0",
                                    "b": "my_mul_0.output_0",
                                },
                            },
                        },
                        "condition_output": None,
                    }
                ],
                "else_case": {
                    "label": "else_body",
                    "node": {
                        "type": "workflow",
                        "inputs": ["x", "y"],
                        "outputs": ["a", "b"],
                        "nodes": {
                            "my_mul_0": my_mul.flowrep_recipe,
                            "my_add_0": my_add.flowrep_recipe,
                        },
                        "input_edges": {
                            "my_mul_0.a": "x",
                            "my_mul_0.b": "y",
                            "my_add_0.a": "x",
                            "my_add_0.b": "y",
                        },
                        "edges": {},
                        "output_edges": {
                            "a": "my_mul_0.output_0",
                            "b": "my_add_0.output_0",
                        },
                    },
                },
                "input_edges": {
                    "condition_0.m": "x",
                    "condition_0.n": "y",
                    "body_0.x": "x",
                    "body_0.y": "y",
                    "else_body.x": "x",
                    "else_body.y": "y",
                },
                "prospective_output_edges": {
                    "a": ["body_0.a", "else_body.a"],
                    "b": ["body_0.b", "else_body.b"],
                },
            }
        },
        "input_edges": {"if_0.x": "x", "if_0.y": "y"},
        "edges": {},
        "output_edges": {"a": "if_0.a", "b": "if_0.b"},
        "fully_qualified_name": "integration.parsers.test_parsing_if_nodes.multi_output_if",
    }
)


# 5. if without else — output may be non-data;
#    symbol must pre-exist so the WfMS has a fallback
def if_no_else(x, y):
    z = my_identity(x)
    if my_cond(x, y):
        z = my_add(x, y)
    return z


no_else_node = workflow_model.WorkflowNode.model_validate(
    {
        "type": "workflow",
        "inputs": ["x", "y"],
        "outputs": ["z"],
        "nodes": {
            "my_identity_0": my_identity.flowrep_recipe,
            "if_0": {
                "type": "if",
                "inputs": ["x", "y"],
                "outputs": ["z"],
                "cases": [
                    {
                        "condition": {
                            "label": "condition_0",
                            "node": my_cond.flowrep_recipe,
                        },
                        "body": {
                            "label": "body_0",
                            "node": {
                                "type": "workflow",
                                "inputs": ["x", "y"],
                                "outputs": ["z"],
                                "nodes": {
                                    "my_add_0": my_add.flowrep_recipe,
                                },
                                "input_edges": {"my_add_0.a": "x", "my_add_0.b": "y"},
                                "edges": {},
                                "output_edges": {"z": "my_add_0.output_0"},
                            },
                        },
                        "condition_output": None,
                    }
                ],
                "else_case": None,
                "input_edges": {
                    "condition_0.m": "x",
                    "condition_0.n": "y",
                    "body_0.x": "x",
                    "body_0.y": "y",
                },
                "prospective_output_edges": {"z": ["body_0.z"]},
            },
        },
        "input_edges": {"my_identity_0.x": "x", "if_0.x": "x", "if_0.y": "y"},
        "edges": {},
        "output_edges": {"z": "if_0.z"},
        "fully_qualified_name": "integration.parsers.test_parsing_if_nodes.if_no_else",
    }
)


def _field_differences(
    reference: workflow_model.WorkflowNode, actual: workflow_model.WorkflowNode
) -> dict:
    dict1 = reference.model_dump(mode="json")
    dict2 = actual.model_dump(mode="json")
    return {
        k: (dict1.get(k), dict2.get(k))
        for k in dict1.keys() | dict2.keys()
        if dict1.get(k) != dict2.get(k)
    }


class TestParsingForLoops(unittest.TestCase):
    """
    These are pretty brittle to source code changes, as the reference is using the
    fully-stringified nested dictionary representation.
    """

    def test_against_static_recipes(self):
        for function, reference in (
            (simple_if_else, simple_node),
            (if_elif_else, elif_node),
            (if_with_context, context_node),
            (multi_output_if, multi_output_node),
            (if_no_else, no_else_node),
        ):
            with self.subTest(function=function.__name__):
                parsed_node = workflow_parser.parse_workflow(function)
                self.assertEqual(
                    parsed_node,
                    reference,
                    msg=f"Differences: {_field_differences(reference, parsed_node)}",
                )


if __name__ == "__main__":
    unittest.main()
