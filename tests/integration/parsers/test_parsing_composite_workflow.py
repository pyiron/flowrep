import dataclasses
import inspect
import unittest

from pyiron_snippets import versions

from flowrep import std, wfms
from flowrep.compiler import source
from flowrep.parsers import atomic_parser, workflow_parser
from flowrep.prospective import (
    for_recipe,
    if_recipe,
    try_recipe,
    while_recipe,
    workflow_recipe,
)

from flowrep_static import library, makers


@atomic_parser.atomic
def my_sum(lst: list[int]) -> int:
    total = sum(lst)
    return total


def full_composite(x, /, y, *, bound):
    """
    Nests all four flow controls with sibling context at every level:
      top-level workflow, siblings around try
        try / except, siblings around while
          while, siblings around for
            for, if/else inside body
    """
    a = std.add(x, y)

    # --- try (level 1) ---
    try:
        b = std.mul(a, y)

        # --- while (level 2) ---
        while library.my_condition(b, bound):
            c = std.add(b, y)
            rs = library.my_range(c)

            # --- for (level 3) ---
            acc = []
            for r in rs:
                # --- if/else (level 4) ---
                if library.my_condition(r, y):  # noqa: SIM108
                    v = std.add(r, c)
                else:
                    v = std.mul(r, c)
                acc.append(v)

            b = my_sum(acc)

        z = std.identity(b)
    except ValueError:
        z = std.identity(a)

    # --- sibling after try ---
    result = std.identity(z)
    return result


# =====================================================================
# Reference nodes (bottom-up)
# =====================================================================

# --- Level 4: if/else true & else bodies ---

_if_true_body = {
    "type": "workflow",
    "inputs": ["r", "c"],
    "outputs": ["v"],
    "nodes": {"add_0": std.add.flowrep_recipe},
    "input_edges": {"add_0.a": "r", "add_0.b": "c"},
    "edges": {},
    "output_edges": {"v": "add_0.added"},
}

_if_else_body = {
    "type": "workflow",
    "inputs": ["r", "c"],
    "outputs": ["v"],
    "nodes": {"mul_0": std.mul.flowrep_recipe},
    "input_edges": {"mul_0.a": "r", "mul_0.b": "c"},
    "edges": {},
    "output_edges": {"v": "mul_0.product"},
}

_if_node = {
    "type": "if",
    "inputs": ["r", "y", "c"],
    "outputs": ["v"],
    "cases": [
        {
            "condition": {
                "label": "condition_0",
                "recipe": library.my_condition.flowrep_recipe,
            },
            "body": {"label": "body_0", "recipe": _if_true_body},
            "condition_output": None,
        },
    ],
    "else_case": {"label": "else_body", "recipe": _if_else_body},
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
    "type": "for_each",
    "inputs": ["y", "c", "rs"],
    "outputs": ["acc"],
    "body_node": {"label": "body", "recipe": _for_body},
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
        "add_0": std.add.flowrep_recipe,
        "my_range_0": library.my_range.flowrep_recipe,
        "for_each_0": _for_node,
        "my_sum_0": my_sum.flowrep_recipe,
    },
    "input_edges": {
        "add_0.a": "b",
        "add_0.b": "y",
        "for_each_0.y": "y",
    },
    "edges": {
        "my_range_0.n": "add_0.added",
        "for_each_0.c": "add_0.added",
        "for_each_0.rs": "my_range_0.output_0",
        "my_sum_0.lst": "for_each_0.acc",
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
            "recipe": library.my_condition.flowrep_recipe,
        },
        "body": {"label": "body", "recipe": _while_body},
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
        "mul_0": std.mul.flowrep_recipe,
        "while_0": _while_node,
        "identity_0": std.identity.flowrep_recipe,
    },
    "input_edges": {
        "mul_0.a": "a",
        "mul_0.b": "y",
        "while_0.bound": "bound",
        "while_0.y": "y",
    },
    "edges": {
        "while_0.b": "mul_0.product",
        "identity_0.x": "while_0.b",
    },
    "output_edges": {
        "b": "while_0.b",
        "z": "identity_0.x",
    },
}

_except_body = {
    "type": "workflow",
    "inputs": ["a"],
    "outputs": ["z"],
    "nodes": {"identity_0": std.identity.flowrep_recipe},
    "input_edges": {"identity_0.x": "a"},
    "edges": {},
    "output_edges": {"z": "identity_0.x"},
}

_try_node = {
    "type": "try",
    "inputs": ["a", "y", "bound"],
    "outputs": ["b", "z"],
    "try_node": {"label": "try_body", "recipe": _try_body},
    "exception_cases": [
        {
            "exceptions": [versions.VersionInfo.of(ValueError)],
            "body": {"label": "except_body_0", "recipe": _except_body},
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

full_composite_node = workflow_recipe.WorkflowRecipe.model_validate(
    {
        "type": "workflow",
        "inputs": ["x", "y", "bound"],
        "outputs": ["result"],
        "description": inspect.getdoc(full_composite),
        "nodes": {
            "add_0": std.add.flowrep_recipe,
            "try_0": _try_node,
            "identity_0": std.identity.flowrep_recipe,
        },
        "input_edges": {
            "add_0.a": "x",
            "add_0.b": "y",
            "try_0.y": "y",
            "try_0.bound": "bound",
        },
        "edges": {
            "try_0.a": "add_0.added",
            "identity_0.x": "try_0.z",
        },
        "output_edges": {"result": "identity_0.x"},
        "reference": {
            "info": dataclasses.asdict(versions.VersionInfo.of(full_composite)),
            "inputs_with_defaults": [],
            "restricted_input_kinds": {
                "x": "POSITIONAL_ONLY",
                "bound": "KEYWORD_ONLY",
            },
        },
    }
)


def _field_differences(
    reference: workflow_recipe.WorkflowRecipe, actual: workflow_recipe.WorkflowRecipe
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

    def test_roundtrip_back_to_python(self):
        free = makers.reference_free(full_composite)
        rendered = source._workflow2python(free)
        fn = rendered.build()
        for x, y, bound in [(1, 2, 10), (3, 1, 8)]:
            self.assertEqual(fn(x, y, bound=bound), full_composite(x, y, bound=bound))
        self.assertEqual(
            makers.dump_no_refs(fn.flowrep_recipe), makers.dump_no_refs(free)
        )


# =====================================================================
# Calling the recipes instead of writing the python
# =====================================================================

# The static recipe above already spells out one recipe of every flow-control type,
# nested inside each other. Pull them back out and call them directly.

_try_flow = full_composite_node.nodes["try_0"]
_while_flow = _try_flow.try_node.recipe.nodes["while_0"]
_for_flow = _while_flow.case.body.recipe.nodes["for_each_0"]
_for_body = _for_flow.body_node.recipe  # A reference-free workflow
_if_flow = _for_body.nodes["if_0"]


@workflow_parser.workflow
def try_by_recipe_call(x, /, y, *, bound):
    """:func:`full_composite` with its ``try`` block replaced by a recipe call."""
    a = std.add(x, y)
    b, z = _try_flow(a, y, bound)
    result = std.identity(z)
    return result


@workflow_parser.workflow
def all_flows_by_recipe_call(x, /, y, *, bound):
    """One call to every recipe type, chained so that each feeds the next.

    Not equivalent to any of the functions above -- the point is only that a single
    workflow exercises all five ``__call__`` implementations at once.
    """
    a = std.add(x, y)
    b, z = _try_flow(a, y, bound)
    n = _while_flow(b, bound, y)
    rs = library.my_range(y)
    acc = _for_flow(y, n, rs)
    s = my_sum(acc)
    v = _if_flow(s, y, n)
    w = _for_body(v, y, n)
    result = std.add(z, w)
    return result


class TestCallingCompositeRecipes(unittest.TestCase):
    """
    Integration test that recipes are callable: a workflow whose nodes are invoked as
    recipe objects must give the same answer run as plain python and run by the WfMS.
    """

    _CASES = [(1, 2, 10), (3, 1, 8)]

    def _via_wfms(self, func, x, y, bound):
        data = wfms.run_recipe(func.flowrep_recipe, x=x, y=y, bound=bound)
        return data.output_ports["result"].value

    def test_try_call_matches_python_syntax(self):
        """Calling the try-recipe stands in for writing the ``try`` block by hand."""
        for x, y, bound in self._CASES:
            with self.subTest(x=x, y=y, bound=bound):
                self.assertEqual(
                    try_by_recipe_call(x, y, bound=bound),
                    full_composite(x, y, bound=bound),
                )

    def test_try_call_matches_wfms(self):
        for x, y, bound in self._CASES:
            with self.subTest(x=x, y=y, bound=bound):
                self.assertEqual(
                    try_by_recipe_call(x, y, bound=bound),
                    self._via_wfms(try_by_recipe_call, x, y, bound),
                )

    def test_all_flows_call_matches_wfms(self):
        for x, y, bound in self._CASES:
            with self.subTest(x=x, y=y, bound=bound):
                self.assertEqual(
                    all_flows_by_recipe_call(x, y, bound=bound),
                    self._via_wfms(all_flows_by_recipe_call, x, y, bound),
                )

    def test_underfilled_call_is_not_swallowed_by_the_handler(self):
        """``bound`` is consumed by the while-condition, three levels inside a try
        that handles ValueError. A missing input must not be mistaken for a domain
        error and quietly answered with the except branch."""
        with self.assertRaises(TypeError) as ctx:
            _try_flow(3, 2)
        self.assertEqual(
            str(ctx.exception),
            "One of your TryRecipe() calls is missing 1 required input: ['bound']",
        )

    def test_every_flow_control_type_is_called(self):
        """Guard the premise of :func:`all_flows_by_recipe_call`: if a refactor of the
        static recipe above changes what gets pulled out, the test above could quietly
        stop covering a recipe type."""
        self.assertIsInstance(_try_flow, try_recipe.TryRecipe)
        self.assertIsInstance(_while_flow, while_recipe.WhileRecipe)
        self.assertIsInstance(_for_flow, for_recipe.ForEachRecipe)
        self.assertIsInstance(_if_flow, if_recipe.IfRecipe)
        self.assertIsInstance(_for_body, workflow_recipe.WorkflowRecipe)
        self.assertIsNone(
            _for_body.reference,
            msg="A referenced workflow would defer to its python function instead of "
            "running its own graph",
        )


if __name__ == "__main__":
    unittest.main()
