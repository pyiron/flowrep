import unittest

from flowrep.models.nodes import workflow_model
from flowrep.models.parsers import atomic_parser, parser_helpers, workflow_parser


# Reusable atomics
@atomic_parser.atomic
def my_add(a, b):
    return a + b


@atomic_parser.atomic
def my_mul(a, b):
    return a * b


@atomic_parser.atomic
def my_condition(m, n):
    return m < n


@atomic_parser.atomic
def my_identity(x):
    return x


@atomic_parser.atomic
def my_range(n: int) -> list[int]:
    return list(range(n))


@atomic_parser.atomic
def my_sum(lst: list[int]) -> int:
    total = sum(lst)
    return total


def full_composite(x, y, bound):
    """
    Nests all four flow controls with sibling context at every level:
      top-level workflow, siblings around try
        try / except, siblings around while
          while, siblings around for
            for, if/else inside body
    """
    a = my_add(x, y)

    # --- try (level 1) ---
    try:
        b = my_mul(a, y)

        # --- while (level 2) ---
        while my_condition(b, bound):
            c = my_add(b, y)
            rs = my_range(c)

            # --- for (level 3) ---
            acc = []
            for r in rs:
                # --- if/else (level 4) ---
                if my_condition(r, y):  # noqa: SIM108
                    v = my_add(r, c)
                else:
                    v = my_mul(r, c)
                acc.append(v)

            b = my_sum(acc)

        z = my_identity(b)
    except ValueError:
        z = my_identity(a)

    # --- sibling after try ---
    result = my_identity(z)
    return result


# =====================================================================
# Reference nodes (bottom-up)
# =====================================================================

# --- Level 4: if/else true & else bodies ---

_if_true_body = {
    "type": "workflow",
    "inputs": ["r", "c"],
    "outputs": ["v"],
    "nodes": {"my_add_0": my_add.flowrep_recipe},
    "input_edges": {"my_add_0.a": "r", "my_add_0.b": "c"},
    "edges": {},
    "output_edges": {"v": "my_add_0.output_0"},
}

_if_else_body = {
    "type": "workflow",
    "inputs": ["r", "c"],
    "outputs": ["v"],
    "nodes": {"my_mul_0": my_mul.flowrep_recipe},
    "input_edges": {"my_mul_0.a": "r", "my_mul_0.b": "c"},
    "edges": {},
    "output_edges": {"v": "my_mul_0.output_0"},
}

_if_node = {
    "type": "if",
    "inputs": ["r", "y", "c"],
    "outputs": ["v"],
    "cases": [
        {
            "condition": {
                "label": "condition_0",
                "node": my_condition.flowrep_recipe,
            },
            "body": {"label": "body_0", "node": _if_true_body},
            "condition_output": None,
        },
    ],
    "else_case": {"label": "else_body", "node": _if_else_body},
    "input_edges": {
        "condition_0.m": "r",
        "condition_0.n": "y",
        "body_0.r": "r",
        "body_0.c": "c",
        "else_body.r": "r",
        "else_body.c": "c",
    },
    "prospective_output_edges": {"v": ["body_0.v", "else_body.v"]},
}

# --- Level 3: for body & for node ---

_for_body = {
    "type": "workflow",
    "inputs": ["r", "y", "c"],
    "outputs": ["v"],
    "nodes": {"if_0": _if_node},
    "input_edges": {"if_0.r": "r", "if_0.y": "y", "if_0.c": "c"},
    "edges": {},
    "output_edges": {"v": "if_0.v"},
}

_for_node = {
    "type": "for",
    "inputs": ["y", "c", "rs"],
    "outputs": ["acc"],
    "body_node": {"label": "body", "node": _for_body},
    "input_edges": {"body.y": "y", "body.c": "c", "body.r": "rs"},
    "output_edges": {"acc": "body.v"},
    "nested_ports": ["r"],
    "zipped_ports": [],
}

# --- Level 2: while body & while node ---

_while_body = {
    "type": "workflow",
    "inputs": ["b", "y"],
    "outputs": ["b"],
    "nodes": {
        "my_add_0": my_add.flowrep_recipe,
        "my_range_0": my_range.flowrep_recipe,
        "for_0": _for_node,
        "my_sum_0": my_sum.flowrep_recipe,
    },
    "input_edges": {
        "my_add_0.a": "b",
        "my_add_0.b": "y",
        "for_0.y": "y",
    },
    "edges": {
        "my_range_0.n": "my_add_0.output_0",
        "for_0.c": "my_add_0.output_0",
        "for_0.rs": "my_range_0.output_0",
        "my_sum_0.lst": "for_0.acc",
    },
    "output_edges": {"b": "my_sum_0.total"},
}

_while_node = {
    "type": "while",
    "inputs": ["b", "bound", "y"],
    "outputs": ["b"],
    "case": {
        "condition": {
            "label": "condition",
            "node": my_condition.flowrep_recipe,
        },
        "body": {"label": "body", "node": _while_body},
        "condition_output": None,
    },
    "input_edges": {
        "condition.m": "b",
        "condition.n": "bound",
        "body.b": "b",
        "body.y": "y",
    },
    "output_edges": {"b": "body.b"},
}

# --- Level 1: try body, except body, try node ---

_try_body = {
    "type": "workflow",
    "inputs": ["a", "y", "bound"],
    "outputs": ["b", "z"],
    "nodes": {
        "my_mul_0": my_mul.flowrep_recipe,
        "while_0": _while_node,
        "my_identity_0": my_identity.flowrep_recipe,
    },
    "input_edges": {
        "my_mul_0.a": "a",
        "my_mul_0.b": "y",
        "while_0.bound": "bound",
        "while_0.y": "y",
    },
    "edges": {
        "while_0.b": "my_mul_0.output_0",
        "my_identity_0.x": "while_0.b",
    },
    "output_edges": {
        "b": "while_0.b",
        "z": "my_identity_0.x",
    },
}

_except_body = {
    "type": "workflow",
    "inputs": ["a"],
    "outputs": ["z"],
    "nodes": {"my_identity_0": my_identity.flowrep_recipe},
    "input_edges": {"my_identity_0.x": "a"},
    "edges": {},
    "output_edges": {"z": "my_identity_0.x"},
}

_try_node = {
    "type": "try",
    "inputs": ["a", "y", "bound"],
    "outputs": ["b", "z"],
    "try_node": {"label": "try_body", "node": _try_body},
    "exception_cases": [
        {
            "exceptions": ["builtins.ValueError"],
            "body": {"label": "except_body_0", "node": _except_body},
        },
    ],
    "input_edges": {
        "try_body.a": "a",
        "try_body.y": "y",
        "try_body.bound": "bound",
        "except_body_0.a": "a",
    },
    "prospective_output_edges": {
        "b": ["try_body.b"],
        "z": ["try_body.z", "except_body_0.z"],
    },
}

# --- Top-level workflow ---

full_composite_node = workflow_model.WorkflowNode.model_validate(
    {
        "type": "workflow",
        "inputs": ["x", "y", "bound"],
        "outputs": ["result"],
        "nodes": {
            "my_add_0": my_add.flowrep_recipe,
            "try_0": _try_node,
            "my_identity_0": my_identity.flowrep_recipe,
        },
        "input_edges": {
            "my_add_0.a": "x",
            "my_add_0.b": "y",
            "try_0.y": "y",
            "try_0.bound": "bound",
        },
        "edges": {
            "try_0.a": "my_add_0.output_0",
            "my_identity_0.x": "try_0.z",
        },
        "output_edges": {"result": "my_identity_0.x"},
        "source": {
            "module": "integration.parsers.test_parsing_composite_workflow",
            "qualname": "full_composite",
            "version": None,
        },
        "source_code": parser_helpers.get_available_source_code(full_composite),
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


class TestParsingFullComposite(unittest.TestCase):
    """
    Integration test nesting all four flow-control constructs
    (try, while, for, if) with sibling context at every level.
    """

    def test_against_static_recipe(self):
        parsed_node = workflow_parser.parse_workflow(full_composite)
        self.assertEqual(
            parsed_node,
            full_composite_node,
            msg=f"Differences: {_field_differences(full_composite_node, parsed_node)}",
        )


if __name__ == "__main__":
    unittest.main()
