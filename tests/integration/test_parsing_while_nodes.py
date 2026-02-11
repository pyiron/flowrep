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


class TestParsingWhileLoops(unittest.TestCase):
    """
    These are pretty brittle to source code changes, as the reference is using the
    fully-stringified nested dictionary representation.
    """

    def test_against_static_recipes(self):
        for function, reference in ((simple_while, simple_while_node),):
            with self.subTest(function=function.__name__):
                parsed_node = workflow_parser.parse_workflow(function)
                self.assertEqual(parsed_node, reference)
