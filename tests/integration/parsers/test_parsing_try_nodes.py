import unittest

from pyiron_snippets import versions

from flowrep.models.nodes import workflow_model
from flowrep.models.parsers import atomic_parser, parser_helpers, workflow_parser

_VALUE_ERROR_INFO = versions.VersionInfo.of(ValueError)
_TYPE_ERROR_INFO = versions.VersionInfo.of(TypeError)


# Reusable atomics (same as other test files)
@atomic_parser.atomic
def my_add(a, b):
    return a + b


@atomic_parser.atomic
def my_mul(a, b):
    return a * b


@atomic_parser.atomic
def my_identity(x):
    return x


# 1. Simple try/except — single symbol assigned in both branches
def simple_try_except(x, y):
    try:
        z = my_add(x, y)
    except ValueError:
        z = my_mul(x, y)
    return z


simple_node = workflow_model.WorkflowNode.model_validate(
    {
        "type": "workflow",
        "inputs": ["x", "y"],
        "outputs": ["z"],
        "nodes": {
            "try_0": {
                "type": "try",
                "inputs": ["x", "y"],
                "outputs": ["z"],
                "try_node": {
                    "label": "try_body",
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
                "exception_cases": [
                    {
                        "exceptions": [_VALUE_ERROR_INFO],
                        "body": {
                            "label": "except_body_0",
                            "node": {
                                "type": "workflow",
                                "inputs": ["x", "y"],
                                "outputs": ["z"],
                                "nodes": {
                                    "my_mul_0": my_mul.flowrep_recipe,
                                },
                                "input_edges": {
                                    "my_mul_0.a": "x",
                                    "my_mul_0.b": "y",
                                },
                                "edges": {},
                                "output_edges": {"z": "my_mul_0.output_0"},
                            },
                        },
                    }
                ],
                "input_edges": {
                    "try_body.x": "x",
                    "try_body.y": "y",
                    "except_body_0.x": "x",
                    "except_body_0.y": "y",
                },
                "prospective_output_edges": {
                    "z": ["try_body.z", "except_body_0.z"],
                },
            }
        },
        "input_edges": {"try_0.x": "x", "try_0.y": "y"},
        "edges": {},
        "output_edges": {"z": "try_0.z"},
        "reference": {
            "info": {
                "module": "integration.parsers.test_parsing_try_nodes",
                "qualname": "simple_try_except",
                "version": None,
            },
            "inputs_with_defaults": [],
        },
        "source_code": parser_helpers.get_available_source_code(simple_try_except),
    }
)


# 2. try with multiple except handlers
def try_multi_except(x, y):
    try:
        z = my_add(x, y)
    except ValueError:
        z = my_mul(x, y)
    except TypeError:
        z = my_identity(x)
    return z


multi_except_node = workflow_model.WorkflowNode.model_validate(
    {
        "type": "workflow",
        "inputs": ["x", "y"],
        "outputs": ["z"],
        "nodes": {
            "try_0": {
                "type": "try",
                "inputs": ["x", "y"],
                "outputs": ["z"],
                "try_node": {
                    "label": "try_body",
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
                "exception_cases": [
                    {
                        "exceptions": [_VALUE_ERROR_INFO],
                        "body": {
                            "label": "except_body_0",
                            "node": {
                                "type": "workflow",
                                "inputs": ["x", "y"],
                                "outputs": ["z"],
                                "nodes": {
                                    "my_mul_0": my_mul.flowrep_recipe,
                                },
                                "input_edges": {
                                    "my_mul_0.a": "x",
                                    "my_mul_0.b": "y",
                                },
                                "edges": {},
                                "output_edges": {"z": "my_mul_0.output_0"},
                            },
                        },
                    },
                    {
                        "exceptions": [_TYPE_ERROR_INFO],
                        "body": {
                            "label": "except_body_1",
                            "node": {
                                "type": "workflow",
                                "inputs": ["x"],
                                "outputs": ["z"],
                                "nodes": {
                                    "my_identity_0": my_identity.flowrep_recipe,
                                },
                                "input_edges": {"my_identity_0.x": "x"},
                                "edges": {},
                                "output_edges": {"z": "my_identity_0.x"},
                            },
                        },
                    },
                ],
                "input_edges": {
                    "try_body.x": "x",
                    "try_body.y": "y",
                    "except_body_0.x": "x",
                    "except_body_0.y": "y",
                    "except_body_1.x": "x",
                },
                "prospective_output_edges": {
                    "z": [
                        "try_body.z",
                        "except_body_0.z",
                        "except_body_1.z",
                    ],
                },
            }
        },
        "input_edges": {"try_0.x": "x", "try_0.y": "y"},
        "edges": {},
        "output_edges": {"z": "try_0.z"},
        "reference": {
            "info": {
                "module": "integration.parsers.test_parsing_try_nodes",
                "qualname": "try_multi_except",
                "version": None,
            },
            "inputs_with_defaults": [],
        },
        "source_code": parser_helpers.get_available_source_code(try_multi_except),
    }
)


# 3. try-node embedded between upstream and downstream siblings
def try_with_context(a, b):
    x = my_add(a, b)
    try:
        y = my_mul(x, b)
    except ValueError:
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
            "try_0": {
                "type": "try",
                "inputs": ["x", "b"],
                "outputs": ["y"],
                "try_node": {
                    "label": "try_body",
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
                "exception_cases": [
                    {
                        "exceptions": [_VALUE_ERROR_INFO],
                        "body": {
                            "label": "except_body_0",
                            "node": {
                                "type": "workflow",
                                "inputs": ["x", "b"],
                                "outputs": ["y"],
                                "nodes": {
                                    "my_add_0": my_add.flowrep_recipe,
                                },
                                "input_edges": {
                                    "my_add_0.a": "x",
                                    "my_add_0.b": "b",
                                },
                                "edges": {},
                                "output_edges": {"y": "my_add_0.output_0"},
                            },
                        },
                    }
                ],
                "input_edges": {
                    "try_body.x": "x",
                    "try_body.b": "b",
                    "except_body_0.x": "x",
                    "except_body_0.b": "b",
                },
                "prospective_output_edges": {
                    "y": ["try_body.y", "except_body_0.y"],
                },
            },
            "my_identity_0": my_identity.flowrep_recipe,
        },
        "input_edges": {"my_add_0.a": "a", "my_add_0.b": "b", "try_0.b": "b"},
        "edges": {
            "try_0.x": "my_add_0.output_0",
            "my_identity_0.x": "try_0.y",
        },
        "output_edges": {"z": "my_identity_0.x"},
        "reference": {
            "info": {
                "module": "integration.parsers.test_parsing_try_nodes",
                "qualname": "try_with_context",
                "version": None,
            },
            "inputs_with_defaults": [],
        },
        "source_code": parser_helpers.get_available_source_code(try_with_context),
    }
)


# 4. Multiple outputs from try/except branches
def multi_output_try(x, y):
    try:
        a = my_add(x, y)
        b = my_mul(x, y)
    except ValueError:
        a = my_mul(x, y)
        b = my_add(x, y)
    return a, b


multi_output_node = workflow_model.WorkflowNode.model_validate(
    {
        "type": "workflow",
        "inputs": ["x", "y"],
        "outputs": ["a", "b"],
        "nodes": {
            "try_0": {
                "type": "try",
                "inputs": ["x", "y"],
                "outputs": ["a", "b"],
                "try_node": {
                    "label": "try_body",
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
                "exception_cases": [
                    {
                        "exceptions": [_VALUE_ERROR_INFO],
                        "body": {
                            "label": "except_body_0",
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
                    }
                ],
                "input_edges": {
                    "try_body.x": "x",
                    "try_body.y": "y",
                    "except_body_0.x": "x",
                    "except_body_0.y": "y",
                },
                "prospective_output_edges": {
                    "a": ["try_body.a", "except_body_0.a"],
                    "b": ["try_body.b", "except_body_0.b"],
                },
            }
        },
        "input_edges": {"try_0.x": "x", "try_0.y": "y"},
        "edges": {},
        "output_edges": {"a": "try_0.a", "b": "try_0.b"},
        "reference": {
            "info": {
                "module": "integration.parsers.test_parsing_try_nodes",
                "qualname": "multi_output_try",
                "version": None,
            },
            "inputs_with_defaults": [],
        },
        "source_code": parser_helpers.get_available_source_code(multi_output_try),
    }
)


# 5. Tuple exception types in a single handler
def try_tuple_exceptions(x, y):
    try:
        z = my_add(x, y)
    except (ValueError, TypeError):
        z = my_mul(x, y)
    return z


tuple_exc_node = workflow_model.WorkflowNode.model_validate(
    {
        "type": "workflow",
        "inputs": ["x", "y"],
        "outputs": ["z"],
        "nodes": {
            "try_0": {
                "type": "try",
                "inputs": ["x", "y"],
                "outputs": ["z"],
                "try_node": {
                    "label": "try_body",
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
                "exception_cases": [
                    {
                        "exceptions": [
                            _VALUE_ERROR_INFO,
                            _TYPE_ERROR_INFO,
                        ],
                        "body": {
                            "label": "except_body_0",
                            "node": {
                                "type": "workflow",
                                "inputs": ["x", "y"],
                                "outputs": ["z"],
                                "nodes": {
                                    "my_mul_0": my_mul.flowrep_recipe,
                                },
                                "input_edges": {
                                    "my_mul_0.a": "x",
                                    "my_mul_0.b": "y",
                                },
                                "edges": {},
                                "output_edges": {"z": "my_mul_0.output_0"},
                            },
                        },
                    }
                ],
                "input_edges": {
                    "try_body.x": "x",
                    "try_body.y": "y",
                    "except_body_0.x": "x",
                    "except_body_0.y": "y",
                },
                "prospective_output_edges": {
                    "z": ["try_body.z", "except_body_0.z"],
                },
            }
        },
        "input_edges": {"try_0.x": "x", "try_0.y": "y"},
        "edges": {},
        "output_edges": {"z": "try_0.z"},
        "reference": {
            "info": {
                "module": "integration.parsers.test_parsing_try_nodes",
                "qualname": "try_tuple_exceptions",
                "version": None,
            },
            "inputs_with_defaults": [],
        },
        "source_code": parser_helpers.get_available_source_code(try_tuple_exceptions),
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


class TestParsingTryNodes(unittest.TestCase):
    """
    These are pretty brittle to source code changes, as the reference is using the
    fully-stringified nested dictionary representation.
    """

    def test_against_static_recipes(self):
        for function, reference in (
            (simple_try_except, simple_node),
            (try_multi_except, multi_except_node),
            (try_with_context, context_node),
            (multi_output_try, multi_output_node),
            (try_tuple_exceptions, tuple_exc_node),
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
