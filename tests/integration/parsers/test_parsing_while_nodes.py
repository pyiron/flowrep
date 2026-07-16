import unittest

from flowrep import std
from flowrep.parsers import workflow_parser
from flowrep.prospective import while_recipe, workflow_recipe

from flowrep_static import library


def simple_while(a=10, b=20, c=40):
    x = std.identity(a)
    while library.my_condition(x, c):
        x = std.add(x, b)
    y = std.identity(x)
    return y


# We always wrap the flow control bodies in a workflow, even if they're just a single
# node. This is just a tradeoff of more verbosity for less parsing complexity
simple_while_while_node_body = workflow_recipe.WorkflowRecipe.model_validate(
    {
        "type": "workflow",
        "inputs": ["x", "b"],
        "outputs": ["x"],
        "nodes": {"add_0": std.add.flowrep_recipe},
        "input_edges": {"add_0.a": "x", "add_0.b": "b"},
        "edges": {},
        "output_edges": {"x": "add_0.added"},
    }
)

simple_while_while_node = while_recipe.WhileRecipe.model_validate(
    {
        "type": "while",
        "inputs": ["x", "c", "b"],
        "outputs": ["x"],
        "case": {
            "condition": {
                "label": "condition",
                "recipe": library.my_condition.flowrep_recipe,
            },
            "body": {
                "label": "body",
                "recipe": simple_while_while_node_body,
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

simple_while_node = workflow_recipe.WorkflowRecipe.model_validate(
    {
        "type": "workflow",
        "inputs": ["a", "b", "c"],
        "outputs": ["y"],
        "nodes": {
            "identity_0": std.identity.flowrep_recipe,
            "while_0": simple_while_while_node,
            "identity_1": std.identity.flowrep_recipe,
        },
        "input_edges": {
            "identity_0.x": "a",
            "while_0.c": "c",  # Re-ordering is sensible because the while node
            "while_0.b": "b",  # has inputs in the order of appearance
        },
        "edges": {
            "while_0.x": "identity_0.x",
            "identity_1.x": "while_0.x",
        },
        "output_edges": {"y": "identity_1.x"},
        "reference": {
            "info": {
                "module": "integration.parsers.test_parsing_while_nodes",
                "qualname": "simple_while",
                "version": None,
            },
            "inputs_with_defaults": ["a", "b", "c"],
        },
    }
)


def nested_while(x, m, n, a):
    y = std.identity(x)  # Initial value for y must be available
    # since we return it from a while node
    while library.my_condition(x, m):
        x = std.add(x, a)
        while library.my_condition(y, n):
            y = std.add(y, x)
    return x, y


nest_while_node = workflow_recipe.WorkflowRecipe.model_validate(
    {
        "type": "workflow",
        "inputs": ["x", "m", "n", "a"],
        "outputs": ["x", "y"],
        "nodes": {
            "identity_0": std.identity.flowrep_recipe,
            "while_0": {
                "type": "while",
                "inputs": ["x", "m", "a", "y", "n"],
                "outputs": ["x", "y"],
                "case": {
                    "condition": {
                        "label": "condition",
                        "recipe": library.my_condition.flowrep_recipe,
                    },
                    "body": {
                        "label": "body",
                        "recipe": {
                            "type": "workflow",
                            "inputs": ["x", "a", "y", "n"],
                            "outputs": ["x", "y"],
                            "nodes": {
                                "add_0": std.add.flowrep_recipe,
                                "while_0": {
                                    "type": "while",
                                    "inputs": ["y", "n", "x"],
                                    "outputs": ["y"],
                                    "case": {
                                        "condition": {
                                            "label": "condition",
                                            "recipe": library.my_condition.flowrep_recipe,
                                        },
                                        "body": {
                                            "label": "body",
                                            "recipe": {
                                                "type": "workflow",
                                                "inputs": ["y", "x"],
                                                "outputs": ["y"],
                                                "nodes": {
                                                    "add_0": std.add.flowrep_recipe,
                                                },
                                                "input_edges": {
                                                    "add_0.a": "y",
                                                    "add_0.b": "x",
                                                },
                                                "edges": {},
                                                "output_edges": {"y": "add_0.added"},
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
                                "add_0.a": "x",
                                "add_0.b": "a",
                                "while_0.y": "y",
                                "while_0.n": "n",
                            },
                            "edges": {"while_0.x": "add_0.added"},
                            "output_edges": {
                                "x": "add_0.added",
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
            "identity_0.x": "x",
            "while_0.x": "x",
            "while_0.m": "m",
            "while_0.a": "a",
            "while_0.n": "n",
        },
        "edges": {"while_0.y": "identity_0.x"},
        "output_edges": {"x": "while_0.x", "y": "while_0.y"},
        "reference": {
            "info": {
                "module": "integration.parsers.test_parsing_while_nodes",
                "qualname": "nested_while",
                "version": None,
            },
            "inputs_with_defaults": [],
        },
    }
)


def multi_reassign(x, y, bound):
    while library.my_condition(x, bound):
        x = std.add(x, y)
        y = std.add(y, x)  # Gets internal edge to first std.add
    return x, y


multi_reassign_body = workflow_recipe.WorkflowRecipe.model_validate(
    {
        "type": "workflow",
        "inputs": ["x", "y"],
        "outputs": ["x", "y"],
        "nodes": {
            "add_0": std.add.flowrep_recipe,
            "add_1": std.add.flowrep_recipe,
        },
        "input_edges": {
            "add_0.a": "x",
            "add_0.b": "y",
            "add_1.a": "y",
        },
        "edges": {
            "add_1.b": "add_0.added",
        },
        "output_edges": {
            "x": "add_0.added",
            "y": "add_1.added",
        },
    }
)

multi_reassign_node = workflow_recipe.WorkflowRecipe.model_validate(
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
                        "recipe": library.my_condition.flowrep_recipe,
                    },
                    "body": {
                        "label": "body",
                        "recipe": multi_reassign_body,
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
        "reference": {
            "info": {
                "module": "integration.parsers.test_parsing_while_nodes",
                "qualname": "multi_reassign",
                "version": None,
            },
            "inputs_with_defaults": [],
        },
    }
)


def sequential_whiles(x, y, m, n):
    while library.my_condition(x, m):
        x = std.add(x, y)
    while library.my_condition(
        x, n
    ):  # Fully sequential -- gets a sibling edge to first while
        x = std.add(x, y)
    return x


seq_while_body = workflow_recipe.WorkflowRecipe.model_validate(
    {
        "type": "workflow",
        "inputs": ["x", "y"],
        "outputs": ["x"],
        "nodes": {"add_0": std.add.flowrep_recipe},
        "input_edges": {"add_0.a": "x", "add_0.b": "y"},
        "edges": {},
        "output_edges": {"x": "add_0.added"},
    }
)

sequential_whiles_node = workflow_recipe.WorkflowRecipe.model_validate(
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
                        "recipe": library.my_condition.flowrep_recipe,
                    },
                    "body": {"label": "body", "recipe": seq_while_body},
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
                        "recipe": library.my_condition.flowrep_recipe,
                    },
                    "body": {"label": "body", "recipe": seq_while_body},
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
        "reference": {
            "info": {
                "module": "integration.parsers.test_parsing_while_nodes",
                "qualname": "sequential_whiles",
                "version": None,
            },
            "inputs_with_defaults": [],
        },
    }
)


def chained_body(x, a, b, bound):
    while library.my_condition(x, bound):
        tmp = std.add(x, a)  # Internally created variable creates internal edge
        x = std.add(tmp, b)
    return x


chained_body_node = workflow_recipe.WorkflowRecipe.model_validate(
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
                        "recipe": library.my_condition.flowrep_recipe,
                    },
                    "body": {
                        "label": "body",
                        "recipe": {
                            "type": "workflow",
                            "inputs": ["x", "a", "b"],
                            "outputs": ["x"],
                            "nodes": {
                                "add_0": std.add.flowrep_recipe,
                                "add_1": std.add.flowrep_recipe,
                            },
                            "input_edges": {
                                "add_0.a": "x",
                                "add_0.b": "a",
                                "add_1.b": "b",
                            },
                            "edges": {
                                "add_1.a": "add_0.added",
                            },
                            "output_edges": {
                                "x": "add_1.added",
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
        "reference": {
            "info": {
                "module": "integration.parsers.test_parsing_while_nodes",
                "qualname": "chained_body",
                "version": None,
            },
            "inputs_with_defaults": [],
        },
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


if __name__ == "__main__":
    unittest.main()
