import unittest

from flowrep.models.nodes import workflow_model
from flowrep.models.parsers import atomic_parser, workflow_parser


@atomic_parser.atomic
def my_add(a, b):
    return a + b


@atomic_parser.atomic
def my_condition(m, n):
    return m < n


@atomic_parser.atomic
def my_range(n: int) -> list[int]:
    return list(range(n))


@atomic_parser.atomic
def my_sum(lst: list[int]) -> int:
    total = sum(lst)
    return total


def composite(x, m, n, a):
    while my_condition(x, m):
        x = my_add(x, a)
        vec = my_range(n)
        acc = []
        for i in vec:
            more = my_add(i, x)
            acc.append(more)
    return x


composite_node = workflow_model.WorkflowNode.model_validate(
    {
        "type": "workflow",
        "inputs": ["x", "m", "n", "a"],
        "outputs": ["x"],
        "nodes": {
            "while_0": {
                "type": "while",
                "inputs": ["x", "m", "a", "n"],
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
                            "inputs": ["x", "a", "n"],
                            "outputs": ["x"],
                            "nodes": {
                                "my_add_0": my_add.flowrep_recipe,
                                "my_range_0": my_range.flowrep_recipe,
                                "for_0": {
                                    "type": "for",
                                    "inputs": ["x", "vec"],
                                    "outputs": ["acc"],
                                    "body_node": {
                                        "label": "body",
                                        "node": {
                                            "type": "workflow",
                                            "inputs": ["i", "x"],
                                            "outputs": ["more"],
                                            "nodes": {
                                                "my_add_0": my_add.flowrep_recipe,
                                            },
                                            "input_edges": {
                                                "my_add_0.a": "i",
                                                "my_add_0.b": "x",
                                            },
                                            "edges": {},
                                            "output_edges": {
                                                "more": "my_add_0.output_0"
                                            },
                                        },
                                    },
                                    "input_edges": {"body.x": "x", "body.i": "vec"},
                                    "output_edges": {"acc": "body.more"},
                                    "nested_ports": ["i"],
                                    "zipped_ports": [],
                                },
                            },
                            "input_edges": {
                                "my_add_0.a": "x",
                                "my_add_0.b": "a",
                                "my_range_0.n": "n",
                            },
                            "edges": {
                                "for_0.x": "my_add_0.output_0",
                                "for_0.vec": "my_range_0.output_0",
                            },
                            "output_edges": {"x": "my_add_0.output_0"},
                        },
                    },
                    "condition_output": None,
                },
                "input_edges": {
                    "condition.m": "x",
                    "condition.n": "m",
                    "body.x": "x",
                    "body.a": "a",
                    "body.n": "n",
                },
                "output_edges": {"x": "body.x"},
            }
        },
        "input_edges": {
            "while_0.x": "x",
            "while_0.m": "m",
            "while_0.a": "a",
            "while_0.n": "n",
        },
        "edges": {},
        "output_edges": {"x": "while_0.x"},
    }
)


class TestParsingCompositeWorkflows(unittest.TestCase):
    """
    These are pretty brittle to source code changes, as the reference is using the
    fully-stringified nested dictionary representation.
    """

    def test_against_static_recipes(self):
        for function, reference in ((composite, composite_node),):
            with self.subTest(function=function.__name__):
                parsed_node = workflow_parser.parse_workflow(function)
                self.assertEqual(parsed_node, reference)


if __name__ == "__main__":
    unittest.main()
