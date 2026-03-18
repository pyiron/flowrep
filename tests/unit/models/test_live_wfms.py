"""Tests for the live data model and the minimal WfMS."""

import dataclasses
import pickle
import unittest
from typing import get_origin

from pyiron_snippets import versions

from flowrep.models import edge_models, live, wfms
from flowrep.models.live import NOT_DATA
from flowrep.models.nodes import (
    atomic_model,
    for_model,
    helper_models,
    if_model,
    try_model,
    while_model,
    workflow_model,
)
from flowrep.models.parsers import atomic_parser, workflow_parser

from flowrep_static import library

# ═══════════════════════════════════════════════════════════════════════════
# Helper factories for manual recipe construction
# ═══════════════════════════════════════════════════════════════════════════


def _single_node_workflow(
    inputs: list[str],
    outputs: list[str],
    child_label: str,
    child_recipe: atomic_model.AtomicNode,
    input_map: dict[str, str],
    output_map: dict[str, str],
) -> workflow_model.WorkflowNode:
    """
    Convenience wrapper: one-atomic-child workflow.

    *input_map*: `{child_port: wf_input}`.
    *output_map*: `{wf_output: child_port}`.
    """
    return workflow_model.WorkflowNode(
        inputs=inputs,
        outputs=outputs,
        nodes={child_label: child_recipe},
        input_edges={
            edge_models.TargetHandle(
                node=child_label, port=cp
            ): edge_models.InputSource(port=wp)
            for cp, wp in input_map.items()
        },
        edges={},
        output_edges={
            edge_models.OutputTarget(port=wp): edge_models.SourceHandle(
                node=child_label, port=cp
            )
            for wp, cp in output_map.items()
        },
    )


def _linear_workflow() -> workflow_model.WorkflowNode:
    """`add(x, y) -> mul(result, z) -> output`."""
    return workflow_model.WorkflowNode(
        inputs=["x", "y", "z"],
        outputs=["result"],
        nodes={
            "add_0": library.my_add.flowrep_recipe,
            "mul_0": library.my_mul.flowrep_recipe,
        },
        input_edges={
            edge_models.TargetHandle(node="add_0", port="a"): edge_models.InputSource(
                port="x"
            ),
            edge_models.TargetHandle(node="add_0", port="b"): edge_models.InputSource(
                port="y"
            ),
            edge_models.TargetHandle(node="mul_0", port="b"): edge_models.InputSource(
                port="z"
            ),
        },
        edges={
            edge_models.TargetHandle(node="mul_0", port="a"): edge_models.SourceHandle(
                node="add_0", port="output_0"
            ),
        },
        output_edges={
            edge_models.OutputTarget(port="result"): edge_models.SourceHandle(
                node="mul_0", port="output_0"
            ),
        },
    )


@workflow_parser.workflow
def _passthrough_workflow(x: int = 42) -> int:
    return x


@workflow_parser.workflow
def _diamond_workflow(a: int, b: int = 1) -> int:
    s = library.my_add(a, b)
    n = library.negate(a)
    result = library.my_mul(s, n)
    return result


def _for_negate() -> for_model.ForNode:
    """For each `x` in `xs`, negate it; collect into `ys`."""
    return for_model.ForNode(
        inputs=["xs"],
        outputs=["ys"],
        body_node=helper_models.LabeledNode(
            label="body", node=library.negate.flowrep_recipe
        ),
        input_edges={
            edge_models.TargetHandle(node="body", port="x"): edge_models.InputSource(
                port="xs"
            )
        },
        output_edges={
            edge_models.OutputTarget(port="ys"): edge_models.SourceHandle(
                node="body", port="output_0"
            )
        },
        nested_ports=["x"],
    )


def _for_add_broadcast() -> for_model.ForNode:
    """
    For each `x` in `xs`, compute `add(x, offset)` (offset is broadcast).
    Also transfers scattered `xs` elements to output `inputs_used`.
    """
    return for_model.ForNode(
        inputs=["xs", "offset"],
        outputs=["ys", "inputs_used"],
        body_node=helper_models.LabeledNode(
            label="body", node=library.my_add.flowrep_recipe
        ),
        input_edges={
            edge_models.TargetHandle(node="body", port="a"): edge_models.InputSource(
                port="xs"
            ),
            edge_models.TargetHandle(node="body", port="b"): edge_models.InputSource(
                port="offset"
            ),
        },
        output_edges={
            edge_models.OutputTarget(port="ys"): edge_models.SourceHandle(
                node="body", port="output_0"
            ),
            edge_models.OutputTarget(port="inputs_used"): edge_models.InputSource(
                port="xs"
            ),
        },
        nested_ports=["a"],
    )


def _for_add_zipped() -> for_model.ForNode:
    """Zip `xs` and `ys` element-wise, compute `add(a, b)` for each pair."""
    return for_model.ForNode(
        inputs=["xs", "ys"],
        outputs=["sums"],
        body_node=helper_models.LabeledNode(
            label="body", node=library.my_add.flowrep_recipe
        ),
        input_edges={
            edge_models.TargetHandle(node="body", port="a"): edge_models.InputSource(
                port="xs"
            ),
            edge_models.TargetHandle(node="body", port="b"): edge_models.InputSource(
                port="ys"
            ),
        },
        output_edges={
            edge_models.OutputTarget(port="sums"): edge_models.SourceHandle(
                node="body", port="output_0"
            ),
        },
        zipped_ports=["a", "b"],
    )


def _decrement_body_workflow() -> workflow_model.WorkflowNode:
    """`n -> decrement(n) -> n` — single-step body for while loops."""
    return _single_node_workflow(
        inputs=["n"],
        outputs=["n"],
        child_label="decrement_0",
        child_recipe=library.decrement.flowrep_recipe,
        input_map={"x": "n"},
        output_map={"n": "output_0"},
    )


def _while_countdown() -> while_model.WhileNode:
    """Decrement `n` while `is_positive(n)`."""
    return while_model.WhileNode(
        inputs=["n"],
        outputs=["n"],
        case=helper_models.ConditionalCase(
            condition=helper_models.LabeledNode(
                label="condition", node=library.is_positive.flowrep_recipe
            ),
            body=helper_models.LabeledNode(
                label="body", node=_decrement_body_workflow()
            ),
        ),
        input_edges={
            edge_models.TargetHandle(
                node="condition", port="n"
            ): edge_models.InputSource(port="n"),
            edge_models.TargetHandle(node="body", port="n"): edge_models.InputSource(
                port="n"
            ),
        },
        output_edges={
            edge_models.OutputTarget(port="n"): edge_models.SourceHandle(
                node="body", port="n"
            )
        },
    )


def _identity_body_workflow() -> workflow_model.WorkflowNode:
    return _single_node_workflow(
        inputs=["x"],
        outputs=["y"],
        child_label="identity_0",
        child_recipe=library.identity.flowrep_recipe,
        input_map={"x": "x"},
        output_map={"y": "x"},  # identity output port is named "x"
    )


def _negate_body_workflow() -> workflow_model.WorkflowNode:
    return _single_node_workflow(
        inputs=["x"],
        outputs=["y"],
        child_label="negate_0",
        child_recipe=library.negate.flowrep_recipe,
        input_map={"x": "x"},
        output_map={"y": "output_0"},
    )


def _if_abs() -> if_model.IfNode:
    """If `is_positive(x)` return `identity(x)` else `negate(x)`."""
    return if_model.IfNode(
        inputs=["x"],
        outputs=["y"],
        cases=[
            helper_models.ConditionalCase(
                condition=helper_models.LabeledNode(
                    label="condition_0",
                    node=library.is_positive.flowrep_recipe,
                ),
                body=helper_models.LabeledNode(
                    label="body_0",
                    node=_identity_body_workflow(),
                ),
            )
        ],
        else_case=helper_models.LabeledNode(
            label="else_body",
            node=_negate_body_workflow(),
        ),
        input_edges={
            edge_models.TargetHandle(
                node="condition_0", port="n"
            ): edge_models.InputSource(port="x"),
            edge_models.TargetHandle(node="body_0", port="x"): edge_models.InputSource(
                port="x"
            ),
            edge_models.TargetHandle(
                node="else_body", port="x"
            ): edge_models.InputSource(port="x"),
        },
        prospective_output_edges={
            edge_models.OutputTarget(port="y"): [
                edge_models.SourceHandle(node="body_0", port="y"),
                edge_models.SourceHandle(node="else_body", port="y"),
            ]
        },
    )


@atomic_parser.atomic("value", "flag")
def _value_and_flag(x: int) -> tuple[int, bool]:
    """Returns ``(x, x > 0)`` — multi-output condition for if-node tests."""
    return x, x > 0


def _if_abs_multi_output_condition() -> if_model.IfNode:
    """
    Like :func:`_if_abs` but the condition node returns two outputs (``value``,
    ``flag``) and the case explicitly selects ``flag`` via ``condition_output``.
    """
    return if_model.IfNode(
        inputs=["x"],
        outputs=["y"],
        cases=[
            helper_models.ConditionalCase(
                condition=helper_models.LabeledNode(
                    label="condition_0",
                    node=_value_and_flag.flowrep_recipe,
                ),
                body=helper_models.LabeledNode(
                    label="body_0",
                    node=_identity_body_workflow(),
                ),
                condition_output="flag",
            )
        ],
        else_case=helper_models.LabeledNode(
            label="else_body",
            node=_negate_body_workflow(),
        ),
        input_edges={
            edge_models.TargetHandle(
                node="condition_0", port="x"
            ): edge_models.InputSource(port="x"),
            edge_models.TargetHandle(node="body_0", port="x"): edge_models.InputSource(
                port="x"
            ),
            edge_models.TargetHandle(
                node="else_body", port="x"
            ): edge_models.InputSource(port="x"),
        },
        prospective_output_edges={
            edge_models.OutputTarget(port="y"): [
                edge_models.SourceHandle(node="body_0", port="y"),
                edge_models.SourceHandle(node="else_body", port="y"),
            ]
        },
    )


def _divide_body_workflow() -> workflow_model.WorkflowNode:
    return _single_node_workflow(
        inputs=["a", "b"],
        outputs=["result"],
        child_label="divide_0",
        child_recipe=library.divide.flowrep_recipe,
        input_map={"a": "a", "b": "b"},
        output_map={"result": "output_0"},
    )


def _fallback_body_workflow() -> workflow_model.WorkflowNode:
    """Return `a` unchanged as `result`."""
    return _single_node_workflow(
        inputs=["a"],
        outputs=["result"],
        child_label="identity_0",
        child_recipe=library.identity.flowrep_recipe,
        input_map={"x": "a"},
        output_map={"result": "x"},
    )


def _try_safe_divide() -> try_model.TryNode:
    """Try `divide(a, b)`; on `ZeroDivisionError` return `identity(a)`."""
    return try_model.TryNode(
        inputs=["a", "b"],
        outputs=["result"],
        try_node=helper_models.LabeledNode(
            label="try_body", node=_divide_body_workflow()
        ),
        exception_cases=[
            helper_models.ExceptionCase(
                exceptions=[
                    versions.VersionInfo.of(ZeroDivisionError),
                ],
                body=helper_models.LabeledNode(
                    label="except_body_0", node=_fallback_body_workflow()
                ),
            )
        ],
        input_edges={
            edge_models.TargetHandle(
                node="try_body", port="a"
            ): edge_models.InputSource(port="a"),
            edge_models.TargetHandle(
                node="try_body", port="b"
            ): edge_models.InputSource(port="b"),
            edge_models.TargetHandle(
                node="except_body_0", port="a"
            ): edge_models.InputSource(port="a"),
        },
        prospective_output_edges={
            edge_models.OutputTarget(port="result"): [
                edge_models.SourceHandle(node="try_body", port="result"),
                edge_models.SourceHandle(node="except_body_0", port="result"),
            ]
        },
    )


@atomic_parser.atomic
def _raises_value(fail: bool):
    if fail:
        raise ValueError("That's what I do")
    else:
        return fail


@workflow_parser.workflow
def _failing_try(x):
    try:
        y = _raises_value(x)
    except TypeError:
        y = library.identity(x)
    return y


@dataclasses.dataclass
class SumClass:
    x: int
    y: int

    def sum(self) -> int:
        return self.x + self.y


def takes_positional_only(x, /, y) -> SumClass:
    return SumClass(x=x, y=y)


# ═══════════════════════════════════════════════════════════════════════════
# live.py tests
# ═══════════════════════════════════════════════════════════════════════════


class TestNotData(unittest.TestCase):
    def test_singleton(self):
        self.assertIs(live.NotData(), live.NOT_DATA)

    def test_falsy(self):
        self.assertFalse(live.NOT_DATA)

    def test_repr(self):
        self.assertEqual(repr(live.NotData()), "NOT_DATA")

    def test_pickle_roundtrip(self):
        loaded = pickle.loads(pickle.dumps(live.NOT_DATA))
        self.assertIs(loaded, live.NOT_DATA)


class TestPorts(unittest.TestCase):
    def test_input_port_defaults(self):
        port = live.InputPort()
        self.assertIsInstance(port.value, live.NotData)
        self.assertIsNone(port.annotation)
        self.assertIsInstance(port.default, live.NotData)

    def test_input_port_get_data(self):
        self.assertIs(live.InputPort().get_data(), live.NOT_DATA)
        self.assertIs(live.InputPort(default=0).get_data(), 0)
        self.assertIs(live.InputPort(value=1).get_data(), 1)
        self.assertIs(live.InputPort(value=1, default=0).get_data(), 1)

    def test_output_port_defaults(self):
        port = live.OutputPort()
        self.assertIsInstance(port.value, live.NotData)
        self.assertIsNone(port.annotation)

    def test_input_port_with_values(self):
        port = live.InputPort(value=42, annotation=int, default=0)
        self.assertEqual(port.value, 42)
        self.assertIs(port.annotation, int)
        self.assertEqual(port.default, 0)


class TestAtomicFromRecipe(unittest.TestCase):
    def test_simple(self):
        node = live.Atomic.from_recipe(library.identity.flowrep_recipe)
        self.assertTrue(callable(node.function))
        self.assertIn("x", node.input_ports)
        self.assertIn("x", node.output_ports)
        self.assertEqual(len(node.output_ports), 1)

    def test_with_defaults(self):
        node = live.Atomic.from_recipe(library.increment.flowrep_recipe)
        self.assertEqual(node.input_ports["step"].default, 1)
        self.assertIsInstance(node.input_ports["x"].default, live.NotData)

    def test_function_resolves_correctly(self):
        node = live.Atomic.from_recipe(library.my_add.flowrep_recipe)
        self.assertEqual(node.function(3, 4), 7)

    def test_multi_output(self):
        node = live.Atomic.from_recipe(library.divmod_func.flowrep_recipe)
        self.assertIn("quotient", node.output_ports)
        self.assertIn("remainder", node.output_ports)
        self.assertIs(node.output_ports["quotient"].annotation, float)

    def test_not_unpacking(self):
        recipe = atomic_model.AtomicNode(
            inputs=["a", "b"],
            outputs=["result"],
            reference=library.divmod_func.flowrep_recipe.reference,
            unpack_mode=atomic_model.UnpackMode.NONE,
        )
        node = live.Atomic.from_recipe(recipe)
        origin = get_origin(node.output_ports["result"].annotation)
        self.assertIs(origin, tuple)

    def test_unpacking_dataclass(self):
        recipe = atomic_parser.parse_atomic(
            takes_positional_only,
            unpack_mode=atomic_model.UnpackMode.DATACLASS,
        )
        node = live.Atomic.from_recipe(recipe)
        self.assertIs(node.output_ports["x"].annotation, int)
        self.assertIs(node.output_ports["y"].annotation, int)

    def test_unpacking_dataclass_as_none(self):
        recipe = atomic_parser.parse_atomic(
            takes_positional_only,
            "dc",
            unpack_mode=atomic_model.UnpackMode.NONE,
        )
        node = live.Atomic.from_recipe(recipe)
        self.assertIs(node.output_ports["dc"].annotation, SumClass)

    def test_input_mismatch_raises(self):
        with self.assertRaises(ValueError) as ctx:
            live.Atomic.from_recipe(
                atomic_model.AtomicNode(
                    inputs=["these", "are_not", "correct"],
                    outputs=["x"],
                    reference=library.identity.flowrep_recipe.reference,
                )
            )
        self.assertIn("not found in signature", str(ctx.exception))

    def test_output_not_splittable_raises(self):
        with self.assertRaises(ValueError) as ctx:
            live.Atomic.from_recipe(
                atomic_model.AtomicNode(
                    inputs=["x"],
                    outputs=["x", "y"],
                    reference=library.decrement.flowrep_recipe.reference,
                )
            )
        self.assertIn("is not splittable", str(ctx.exception))

    def test_output_length_mismatch_raises(self):
        with self.assertRaises(ValueError) as ctx:
            live.Atomic.from_recipe(
                atomic_model.AtomicNode(
                    inputs=["a", "b"],
                    outputs=["quotient", "remainder", "extra"],
                    reference=library.divmod_func.flowrep_recipe.reference,
                )
            )
        self.assertIn(
            "(n=3) do not match return annotation args (n=2)", str(ctx.exception)
        )


class TestWorkflowFromRecipe(unittest.TestCase):
    def test_no_reference(self):
        recipe = _linear_workflow()
        wf = live.Workflow.from_recipe(recipe)
        self.assertIs(wf.recipe, recipe)
        self.assertEqual(set(wf.input_ports), {"x", "y", "z"})
        self.assertEqual(set(wf.output_ports), {"result"})
        self.assertIn("add_0", wf.nodes)
        self.assertIsInstance(wf.nodes["add_0"], live.Atomic)

    def test_with_reference(self):
        """
        Ensure that annotations and defaults are parsed from the reference
        """
        recipe = _diamond_workflow.flowrep_recipe
        wf = live.Workflow.from_recipe(recipe)
        self.assertIs(wf.recipe, recipe)
        self.assertEqual(set(wf.input_ports), {"a", "b"})
        self.assertIs(wf.input_ports["a"].annotation, int)
        self.assertIs(wf.input_ports["a"].default, NOT_DATA)
        self.assertIs(wf.input_ports["b"].annotation, int)
        self.assertEqual(wf.input_ports["b"].default, 1)
        self.assertEqual(set(wf.output_ports), {"result"})
        self.assertIs(wf.output_ports["result"].annotation, int)
        self.assertIn("my_add_0", wf.nodes)
        self.assertIsInstance(wf.nodes["my_add_0"], live.Atomic)

    def test_edges_are_carried_over(self):
        recipe = _linear_workflow()
        wf = live.Workflow.from_recipe(recipe)
        self.assertDictEqual(wf.input_edges, recipe.input_edges)
        self.assertDictEqual(wf.edges, recipe.edges)
        self.assertDictEqual(wf.output_edges, recipe.output_edges)


class TestFlowControlFromRecipe(unittest.TestCase):
    def test_starts_empty(self):
        for recipe in (
            _for_negate(),
            _if_abs(),
            _try_safe_divide(),
            _while_countdown(),
        ):
            with self.subTest(recipe=recipe):
                fc = live.FlowControl.from_recipe(recipe)
                self.assertEqual(len(fc.nodes), 0)
                self.assertEqual(len(fc.edges), 0)
                self.assertSetEqual(set(fc.input_ports), set(recipe.inputs))
                self.assertSetEqual(set(fc.output_ports), set(recipe.outputs))


class TestRecipe2Live(unittest.TestCase):
    def test_conversion_types(self):
        for recipe, type_ in (
            (library.identity.flowrep_recipe, live.Atomic),
            (_linear_workflow(), live.Workflow),
            (_for_negate(), live.FlowControl),
            (_if_abs(), live.FlowControl),
            (_try_safe_divide(), live.FlowControl),
            (_while_countdown(), live.FlowControl),
        ):
            with self.subTest(recipe=recipe, type=type_):
                self.assertIsInstance(live.recipe2live(recipe), type_)


# ═══════════════════════════════════════════════════════════════════════════
# wfms.py tests — atomic
# ═══════════════════════════════════════════════════════════════════════════


class TestRunAtomic(unittest.TestCase):
    def test_simple(self):
        node = wfms.run_recipe(library.my_add.flowrep_recipe, a=3, b=4)
        self.assertIsInstance(node, live.Atomic)
        self.assertEqual(node.output_ports["output_0"].value, 7)

    def test_identity_preserves_value(self):
        node = wfms.run_recipe(library.identity.flowrep_recipe, x=42)
        self.assertEqual(node.output_ports["x"].value, 42)

    def test_default_used_when_input_omitted(self):
        node = wfms.run_recipe(library.increment.flowrep_recipe, x=10)
        self.assertEqual(node.output_ports["output_0"].value, 11)

    def test_default_overridden(self):
        node = wfms.run_recipe(library.increment.flowrep_recipe, x=10, step=5)
        self.assertEqual(node.output_ports["output_0"].value, 15)

    def test_multi_output(self):
        node = wfms.run_recipe(library.divmod_func.flowrep_recipe, a=17, b=5)
        self.assertAlmostEqual(node.output_ports["quotient"].value, 3.0)
        self.assertAlmostEqual(node.output_ports["remainder"].value, 2.0)

    def test_missing_input_raises(self):
        with self.assertRaisesRegex(ValueError, "no value and no default"):
            wfms.run_recipe(library.my_add.flowrep_recipe, a=3)

    def test_input_ports_populated(self):
        node = wfms.run_recipe(library.my_add.flowrep_recipe, a=3, b=4)
        self.assertEqual(node.input_ports["a"].value, 3)
        self.assertEqual(node.input_ports["b"].value, 4)

    def test_not_unpacking(self):
        recipe = atomic_model.AtomicNode(
            inputs=["a", "b"],
            outputs=["result"],
            reference=library.divmod_func.flowrep_recipe.reference,
            unpack_mode=atomic_model.UnpackMode.NONE,
        )
        node = wfms.run_recipe(recipe, a=1, b=2)
        self.assertEqual(node.output_ports["result"].value, (0, 1))

    def test_unpacking_dataclass(self):
        recipe = atomic_parser.parse_atomic(
            takes_positional_only,
            "dc_x_field",
            "dc_y_field",
            unpack_mode=atomic_model.UnpackMode.DATACLASS,
        )
        node = wfms.run_recipe(recipe, x=1, y=2)
        self.assertEqual(node.output_ports["dc_x_field"].value, 1)
        self.assertEqual(node.output_ports["dc_y_field"].value, 2)

    def test_unrecognized_input_raises(self):
        with self.assertRaises(ValueError) as ctx:
            wfms.run_recipe(library.my_add.flowrep_recipe, a=3, not_an_input=4)
        self.assertIn("not_an_input", str(ctx.exception))
        self.assertIn("not found", str(ctx.exception))
        self.assertIn(str(library.my_add.flowrep_recipe.inputs), str(ctx.exception))

    def test_positional_only_arguments(self):
        recipe = atomic_parser.parse_atomic(takes_positional_only, "result")
        node = wfms.run_recipe(recipe, x=1, y=2)
        self.assertEqual(node.output_ports["result"].value.sum(), 1 + 2)


# ═══════════════════════════════════════════════════════════════════════════
# wfms.py tests — workflow
# ═══════════════════════════════════════════════════════════════════════════


class TestRunWorkflow(unittest.TestCase):
    def test_linear(self):
        wf = wfms.run_recipe(_linear_workflow(), x=1, y=2, z=3)
        self.assertIsInstance(wf, live.Workflow)
        self.assertEqual(wf.output_ports["result"].value, (1 + 2) * 3)

    def test_diamond(self):
        wf = wfms.run_recipe(_diamond_workflow.flowrep_recipe, a=3, b=7)
        self.assertEqual(wf.output_ports["result"].value, (3 + 7) * (-3))

    def test_passthrough(self):
        wf = wfms.run_recipe(_passthrough_workflow.flowrep_recipe, x=1)
        self.assertEqual(wf.output_ports["x"].value, 1)

    def test_child_nodes_populated(self):
        wf = wfms.run_recipe(_linear_workflow(), x=1, y=2, z=3)
        self.assertIsInstance(wf.nodes["add_0"], live.Atomic)
        self.assertEqual(wf.nodes["add_0"].output_ports["output_0"].value, 3)

    def test_child_defaults(self):
        """Atomic child with a default not wired by any edge still works."""
        recipe = workflow_model.WorkflowNode(
            inputs=["x"],
            outputs=["result"],
            nodes={"inc_0": library.increment.flowrep_recipe},
            input_edges={
                edge_models.TargetHandle(
                    node="inc_0", port="x"
                ): edge_models.InputSource(port="x")
            },
            edges={},
            output_edges={
                edge_models.OutputTarget(port="result"): edge_models.SourceHandle(
                    node="inc_0", port="output_0"
                )
            },
        )
        wf = wfms.run_recipe(recipe, x=10)  # step defaults to 1
        self.assertEqual(wf.output_ports["result"].value, 10 + 1)

    def test_defaults_passthrough(self):
        """Workflow defaults should be used"""
        wf = wfms.run_recipe(_passthrough_workflow.flowrep_recipe)
        self.assertEqual(wf.output_ports["x"].value, 42)


class TestTopoSort(unittest.TestCase):
    def test_linear_order(self):
        recipe = _linear_workflow()
        order = wfms._topo_sort_children(recipe)
        self.assertLess(order.index("add_0"), order.index("mul_0"))

    def test_diamond_order(self):
        order = wfms._topo_sort_children(_diamond_workflow.flowrep_recipe)
        self.assertLess(order.index("my_add_0"), order.index("my_mul_0"))
        self.assertLess(order.index("negate_0"), order.index("my_mul_0"))

    def test_independent_nodes_sorted_alphabetically(self):
        order = wfms._topo_sort_children(_diamond_workflow.flowrep_recipe)
        # add_0 and negate_0 are independent, should be alphabetically ordered
        self.assertLess(order.index("my_add_0"), order.index("negate_0"))


# ═══════════════════════════════════════════════════════════════════════════
# wfms.py tests — for
# ═══════════════════════════════════════════════════════════════════════════


class TestRunFor(unittest.TestCase):
    def test_simple_map(self):
        node = wfms.run_recipe(_for_negate(), xs=[1, 2, 3])
        self.assertEqual(node.output_ports["ys"].value, [-1, -2, -3])

    def test_empty_input(self):
        node = wfms.run_recipe(_for_negate(), xs=[])
        self.assertEqual(node.output_ports["ys"].value, [])

    def test_broadcast_and_transfer(self):
        node = wfms.run_recipe(_for_add_broadcast(), xs=[10, 20, 30], offset=5)
        self.assertEqual(node.output_ports["ys"].value, [15, 25, 35])
        self.assertEqual(node.output_ports["inputs_used"].value, [10, 20, 30])

    def test_body_instances_stored(self):
        node = wfms.run_recipe(_for_negate(), xs=[1, 2])
        self.assertEqual(len(node.nodes), 2)

    def test_zipped_ports(self):
        node = wfms.run_recipe(_for_add_zipped(), xs=[1, 2, 3], ys=[10, 20, 30])
        self.assertEqual(node.output_ports["sums"].value, [11, 22, 33])

    def test_zipped_unequal_lengths_raises(self):
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            wfms.run_recipe(_for_add_zipped(), xs=[1, 2], ys=[10, 20, 30])


# ═══════════════════════════════════════════════════════════════════════════
# wfms.py tests — while
# ═══════════════════════════════════════════════════════════════════════════


class TestRunWhile(unittest.TestCase):
    def test_countdown(self):
        node = wfms.run_recipe(_while_countdown(), n=3)
        self.assertEqual(node.output_ports["n"].value, 0)

    def test_immediate_false(self):
        node = wfms.run_recipe(_while_countdown(), n=0)
        self.assertEqual(node.output_ports["n"].value, 0)

    def test_single_iteration(self):
        node = wfms.run_recipe(_while_countdown(), n=1)
        self.assertEqual(node.output_ports["n"].value, 0)

    def test_body_instances_counted(self):
        node = wfms.run_recipe(_while_countdown(), n=3)
        # 3 body iterations + 4 condition evaluations
        self.assertEqual(len(node.nodes), 3 + 4)


# ═══════════════════════════════════════════════════════════════════════════
# wfms.py tests — if
# ═══════════════════════════════════════════════════════════════════════════


class TestRunIf(unittest.TestCase):
    def test_positive_branch(self):
        node = wfms.run_recipe(_if_abs(), x=5)
        self.assertEqual(node.output_ports["y"].value, 5)

    def test_else_branch(self):
        node = wfms.run_recipe(_if_abs(), x=-3)
        self.assertEqual(node.output_ports["y"].value, 3)

    def test_zero_takes_else(self):
        node = wfms.run_recipe(_if_abs(), x=0)
        self.assertEqual(node.output_ports["y"].value, 0)  # negate(0) == 0

    def test_condition_node_stored(self):
        node = wfms.run_recipe(_if_abs(), x=5)
        self.assertIn("condition_0", node.nodes)

    def test_only_matched_body_stored(self):
        node = wfms.run_recipe(_if_abs(), x=5)
        self.assertIn("body_0", node.nodes)
        self.assertNotIn("else_body", node.nodes)

    def test_multi_output_condition_positive(self):
        node = wfms.run_recipe(_if_abs_multi_output_condition(), x=5)
        self.assertEqual(node.output_ports["y"].value, 5)

    def test_multi_output_condition_negative(self):
        node = wfms.run_recipe(_if_abs_multi_output_condition(), x=-3)
        self.assertEqual(node.output_ports["y"].value, 3)


# ═══════════════════════════════════════════════════════════════════════════
# wfms.py tests — try
# ═══════════════════════════════════════════════════════════════════════════


class TestRunTry(unittest.TestCase):
    def test_no_exception(self):
        node = wfms.run_recipe(_try_safe_divide(), a=10, b=2)
        self.assertAlmostEqual(node.output_ports["result"].value, 5.0)

    def test_caught_exception(self):
        node = wfms.run_recipe(_try_safe_divide(), a=7, b=0)
        self.assertEqual(node.output_ports["result"].value, 7)

    def test_try_body_stored_on_success(self):
        node = wfms.run_recipe(_try_safe_divide(), a=10, b=2)
        self.assertIn("try_body", node.nodes)
        self.assertNotIn("except_body_0", node.nodes)

    def test_handler_body_stored_on_exception(self):
        node = wfms.run_recipe(_try_safe_divide(), a=7, b=0)
        self.assertIn("except_body_0", node.nodes)

    def test_unhandled_exception_propagates(self):
        with self.assertRaisesRegex(ValueError, "That's what I do"):
            wfms.run_recipe(_failing_try.flowrep_recipe, x=42)


# ═══════════════════════════════════════════════════════════════════════════
# wfms.py tests — edge case
# ═══════════════════════════════════════════════════════════════════════════


class TestUnrecognizedRecipe(unittest.TestCase):
    def test_unrecognized_input(self):
        not_a_recipe = "not at all"
        with self.assertRaises(TypeError) as ctx:
            wfms.run_recipe(not_a_recipe)
        self.assertIn("Unsupported recipe type", str(ctx.exception))


# ═══════════════════════════════════════════════════════════════════════════
# wfms.py tests — provenance walk (integration)
# ═══════════════════════════════════════════════════════════════════════════


class TestProvenanceWalk(unittest.TestCase):
    def test_child_io_accessible(self):
        """Every child's input and output port values are available after execution."""
        wf = wfms.run_recipe(_linear_workflow(), x=5, y=3, z=2)
        self.assertIsInstance(wf, live.Workflow)
        add_node = wf.nodes["add_0"]
        self.assertEqual(add_node.input_ports["a"].value, 5)
        self.assertEqual(add_node.input_ports["b"].value, 3)
        self.assertEqual(add_node.output_ports["output_0"].value, 8)
        mul_node = wf.nodes["mul_0"]
        self.assertEqual(mul_node.input_ports["a"].value, 8)
        self.assertEqual(mul_node.input_ports["b"].value, 2)
        self.assertEqual(mul_node.output_ports["output_0"].value, 16)


if __name__ == "__main__":
    unittest.main()
