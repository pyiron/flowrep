import unittest

from flowrep.models.nodes import while_model, workflow_model
from flowrep.models.parsers import atomic_parser, workflow_parser


@atomic_parser.atomic
def my_unity(x):
    return x


@atomic_parser.atomic
def my_add(a, b):
    return a + b


@atomic_parser.atomic
def my_condition(m, n):
    return m < n


def simple_while(a=10, b=20, c=40):
    x = my_unity(a)
    while my_condition(x, c):
        x = my_add(x, b)
    y = my_unity(x)
    return y


# We always wrap the flow control bodies in a workflow, even if they're just a single
# node. This is just a tradeoff of more verbosity for less parsing complexity
simple_while_while_node_body = workflow_model.WorkflowNode.model_validate(
    {
        "type": "workflow",
        "inputs": ["x", "b"],
        "outputs": ["x"],
        "nodes": {"my_add_0": my_add.flowrep_recipe},
        "input_edges": {"my_add_0.a": "x", "my_add_0.b": "b"},
        "edges": {},
        "output_edges": {"x": "my_add_0.output_0"},
    }
)

simple_while_while_node = while_model.WhileNode.model_validate(
    {
        "type": "while",
        "inputs": ["x", "c", "b"],
        "outputs": ["x"],
        "case": {
            "condition": {
                "label": "condition",
                "node": my_condition.flowrep_recipe,
            },
            "body": {
                "label": "body",
                "node": simple_while_while_node_body,
            },
        },
        "input_edges": {
            "condition.m": "x",
            "condition.n": "c",
            "body.x": "x",
            "body.b": "b",
        },
        "output_edges": {"x": "body.x"},
    }
)

simple_while_node = workflow_model.WorkflowNode.model_validate(
    {
        "type": "workflow",
        "inputs": ["a", "b", "c"],
        "outputs": ["y"],
        "nodes": {
            "my_unity_0": my_unity.flowrep_recipe,
            "while_0": simple_while_while_node,
            "my_unity_1": my_unity.flowrep_recipe,
        },
        "input_edges": {
            "my_unity_0.x": "a",
            "while_0.c": "c",  # Re-ordering is sensible because the while node
            "while_0.b": "b",  # has inputs in the order of appearance
        },
        "edges": {
            "while_0.x": "my_unity_0.x",
            "my_unity_1.x": "while_0.x",
        },
        "output_edges": {"y": "my_unity_1.x"},
    }
)


def nested_while(x, m, n, a):
    y = my_unity(x)  # Initial value for y must be available
    # since we return it from a while node
    while my_condition(x, m):
        x = my_add(x, a)
        while my_condition(y, n):
            y = my_add(y, x)
    return x, y


nest_while_node = workflow_model.WorkflowNode.model_validate(
    {
        "type": "workflow",
        "inputs": ["x", "m", "n", "a"],
        "outputs": ["x", "y"],
        "nodes": {
            "my_unity_0": my_unity.flowrep_recipe,
            "while_0": {
                "type": "while",
                "inputs": ["x", "m", "a", "y", "n"],
                "outputs": ["x", "y"],
                "case": {
                    "condition": {
                        "label": "condition",
                        "node": my_condition.flowrep_recipe,
                    },
                    "body": {
                        "label": "body",
                        "node": {
                            "type": "workflow",
                            "inputs": ["x", "a", "y", "n"],
                            "outputs": ["x", "y"],
                            "nodes": {
                                "my_add_0": my_add.flowrep_recipe,
                                "while_0": {
                                    "type": "while",
                                    "inputs": ["y", "n", "x"],
                                    "outputs": ["y"],
                                    "case": {
                                        "condition": {
                                            "label": "condition",
                                            "node": my_condition.flowrep_recipe,
                                        },
                                        "body": {
                                            "label": "body",
                                            "node": {
                                                "type": "workflow",
                                                "inputs": ["y", "x"],
                                                "outputs": ["y"],
                                                "nodes": {
                                                    "my_add_0": my_add.flowrep_recipe,
                                                },
                                                "input_edges": {
                                                    "my_add_0.a": "y",
                                                    "my_add_0.b": "x",
                                                },
                                                "edges": {},
                                                "output_edges": {
                                                    "y": "my_add_0.output_0"
                                                },
                                            },
                                        },
                                    },
                                    "input_edges": {
                                        "condition.m": "y",
                                        "condition.n": "n",
                                        "body.y": "y",
                                        "body.x": "x",
                                    },
                                    "output_edges": {"y": "body.y"},
                                },
                            },
                            "input_edges": {
                                "my_add_0.a": "x",
                                "my_add_0.b": "a",
                                "while_0.y": "y",
                                "while_0.n": "n",
                            },
                            "edges": {"while_0.x": "my_add_0.output_0"},
                            "output_edges": {
                                "x": "my_add_0.output_0",
                                "y": "while_0.y",
                            },
                        },
                    },
                },
                "input_edges": {
                    "condition.m": "x",
                    "condition.n": "m",
                    "body.x": "x",
                    "body.a": "a",
                    "body.y": "y",
                    "body.n": "n",
                },
                "output_edges": {"x": "body.x", "y": "body.y"},
            },
        },
        "input_edges": {
            "my_unity_0.x": "x",
            "while_0.x": "x",
            "while_0.m": "m",
            "while_0.a": "a",
            "while_0.n": "n",
        },
        "edges": {"while_0.y": "my_unity_0.x"},
        "output_edges": {"x": "while_0.x", "y": "while_0.y"},
    }
)


def multi_reassign(x, y, bound):
    while my_condition(x, bound):
        x = my_add(x, y)
        y = my_add(y, x)  # Gets internal edge to first my_add
    return x, y


multi_reassign_body = workflow_model.WorkflowNode.model_validate(
    {
        "type": "workflow",
        "inputs": ["x", "y"],
        "outputs": ["x", "y"],
        "nodes": {
            "my_add_0": my_add.flowrep_recipe,
            "my_add_1": my_add.flowrep_recipe,
        },
        "input_edges": {
            "my_add_0.a": "x",
            "my_add_0.b": "y",
            "my_add_1.a": "y",
        },
        "edges": {
            "my_add_1.b": "my_add_0.output_0",
        },
        "output_edges": {
            "x": "my_add_0.output_0",
            "y": "my_add_1.output_0",
        },
    }
)

multi_reassign_node = workflow_model.WorkflowNode.model_validate(
    {
        "type": "workflow",
        "inputs": ["x", "y", "bound"],
        "outputs": ["x", "y"],
        "nodes": {
            "while_0": {
                "type": "while",
                "inputs": ["x", "bound", "y"],
                "outputs": ["x", "y"],
                "case": {
                    "condition": {
                        "label": "condition",
                        "node": my_condition.flowrep_recipe,
                    },
                    "body": {
                        "label": "body",
                        "node": multi_reassign_body,
                    },
                },
                "input_edges": {
                    "condition.m": "x",
                    "condition.n": "bound",
                    "body.x": "x",
                    "body.y": "y",
                },
                "output_edges": {
                    "x": "body.x",
                    "y": "body.y",
                },
            },
        },
        "input_edges": {
            "while_0.x": "x",
            "while_0.bound": "bound",
            "while_0.y": "y",
        },
        "edges": {},
        "output_edges": {
            "x": "while_0.x",
            "y": "while_0.y",
        },
    }
)


def sequential_whiles(x, y, m, n):
    while my_condition(x, m):
        x = my_add(x, y)
    while my_condition(x, n):  # Fully sequential -- gets a sibling edge to first while
        x = my_add(x, y)
    return x


seq_while_body = workflow_model.WorkflowNode.model_validate(
    {
        "type": "workflow",
        "inputs": ["x", "y"],
        "outputs": ["x"],
        "nodes": {"my_add_0": my_add.flowrep_recipe},
        "input_edges": {"my_add_0.a": "x", "my_add_0.b": "y"},
        "edges": {},
        "output_edges": {"x": "my_add_0.output_0"},
    }
)

sequential_whiles_node = workflow_model.WorkflowNode.model_validate(
    {
        "type": "workflow",
        "inputs": ["x", "y", "m", "n"],
        "outputs": ["x"],
        "nodes": {
            "while_0": {
                "type": "while",
                "inputs": ["x", "m", "y"],
                "outputs": ["x"],
                "case": {
                    "condition": {
                        "label": "condition",
                        "node": my_condition.flowrep_recipe,
                    },
                    "body": {"label": "body", "node": seq_while_body},
                },
                "input_edges": {
                    "condition.m": "x",
                    "condition.n": "m",
                    "body.x": "x",
                    "body.y": "y",
                },
                "output_edges": {"x": "body.x"},
            },
            "while_1": {
                "type": "while",
                "inputs": ["x", "n", "y"],
                "outputs": ["x"],
                "case": {
                    "condition": {
                        "label": "condition",
                        "node": my_condition.flowrep_recipe,
                    },
                    "body": {"label": "body", "node": seq_while_body},
                },
                "input_edges": {
                    "condition.m": "x",
                    "condition.n": "n",
                    "body.x": "x",
                    "body.y": "y",
                },
                "output_edges": {"x": "body.x"},
            },
        },
        "input_edges": {
            "while_0.x": "x",
            "while_0.m": "m",
            "while_0.y": "y",
            "while_1.n": "n",
            "while_1.y": "y",
        },
        "edges": {
            "while_1.x": "while_0.x",
        },
        "output_edges": {
            "x": "while_1.x",
        },
    }
)


def chained_body(x, a, b, bound):
    while my_condition(x, bound):
        tmp = my_add(x, a)  # Internally created variable creates internal edge
        x = my_add(tmp, b)
    return x


chained_body_node = workflow_model.WorkflowNode.model_validate(
    {
        "type": "workflow",
        "inputs": ["x", "a", "b", "bound"],
        "outputs": ["x"],
        "nodes": {
            "while_0": {
                "type": "while",
                "inputs": ["x", "bound", "a", "b"],
                "outputs": ["x"],
                "case": {
                    "condition": {
                        "label": "condition",
                        "node": my_condition.flowrep_recipe,
                    },
                    "body": {
                        "label": "body",
                        "node": {
                            "type": "workflow",
                            "inputs": ["x", "a", "b"],
                            "outputs": ["x"],
                            "nodes": {
                                "my_add_0": my_add.flowrep_recipe,
                                "my_add_1": my_add.flowrep_recipe,
                            },
                            "input_edges": {
                                "my_add_0.a": "x",
                                "my_add_0.b": "a",
                                "my_add_1.b": "b",
                            },
                            "edges": {
                                "my_add_1.a": "my_add_0.output_0",
                            },
                            "output_edges": {
                                "x": "my_add_1.output_0",
                            },
                        },
                    },
                },
                "input_edges": {
                    "condition.m": "x",
                    "condition.n": "bound",
                    "body.x": "x",
                    "body.a": "a",
                    "body.b": "b",
                },
                "output_edges": {"x": "body.x"},
            },
        },
        "input_edges": {
            "while_0.x": "x",
            "while_0.bound": "bound",
            "while_0.a": "a",
            "while_0.b": "b",
        },
        "edges": {},
        "output_edges": {"x": "while_0.x"},
    }
)


class TestParsingWhileLoops(unittest.TestCase):
    """
    These are pretty brittle to source code changes, as the reference is using the
    fully-stringified nested dictionary representation.
    """

    def test_against_static_recipes(self):
        for function, reference in (
            (simple_while, simple_while_node),
            (nested_while, nest_while_node),
            (multi_reassign, multi_reassign_node),
            (sequential_whiles, sequential_whiles_node),
            (chained_body, chained_body_node),
        ):
            with self.subTest(function=function.__name__):
                parsed_node = workflow_parser.parse_workflow(function)
                self.assertEqual(parsed_node, reference)
