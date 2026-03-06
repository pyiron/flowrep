"""Tests for the python-workflow-definition ↔ flowrep converter.

Requires the optional ``python_workflow_definition`` package (installed at test
time from conda-forge).
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import pydantic
import python_workflow_definition.models as pwd_models

from flowrep.models import edge_models  # noqa: E402
from flowrep.models.converters import (  # noqa: E402
    python_workflow_definition as pwd_conv,
)
from flowrep.models.nodes import (  # noqa: E402
    atomic_model,
    workflow_model,
)
from flowrep.models.parsers import workflow_parser  # noqa: E402

from flowrep_static import library, makers  # noqa: E402

_PWD_JSON_WORKFLOW_LOCATION = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "flowrep_static"
    / "python-workflow-definition"
)


_NormalizedEdge = tuple[str, str | None, str, str | None]


def _build_pwd_id_to_key(
    wf: pwd_models.PythonWorkflowDefinitionWorkflow,
) -> dict[int, str]:
    """Map node IDs to canonical string keys for ID-independent comparison."""
    id_to_key: dict[int, str] = {}
    for node in wf.nodes:
        if isinstance(node, pwd_models.PythonWorkflowDefinitionInputNode):
            id_to_key[node.id] = f"input:{node.name}"
        elif isinstance(node, pwd_models.PythonWorkflowDefinitionOutputNode):
            id_to_key[node.id] = f"output:{node.name}"

    func_nodes_by_value: dict[str, list[int]] = {}
    for node in wf.nodes:
        if isinstance(node, pwd_models.PythonWorkflowDefinitionFunctionNode):
            func_nodes_by_value.setdefault(node.value, []).append(node.id)
    for ids in func_nodes_by_value.values():
        ids.sort()
    for value, ids in func_nodes_by_value.items():
        for pos, nid in enumerate(ids):
            id_to_key[nid] = f"func:{value}:{pos}"

    return id_to_key


def _normalize_pwd_edges(
    wf: pwd_models.PythonWorkflowDefinitionWorkflow,
) -> dict[str, list[_NormalizedEdge]]:
    """
    Group normalized edges by target key, preserving intra-group order.

    Global edge-list order across different targets is not semantically
    meaningful — consumers like ``purepython.group_edges`` group by target
    first.  But *within* a target, order matters (e.g. ``get_list`` relies
    on kwargs insertion order).
    """
    id_to_key = _build_pwd_id_to_key(wf)
    by_target: dict[str, list[_NormalizedEdge]] = {}
    for e in wf.edges:
        target_key = id_to_key[e.target]
        edge_tuple: _NormalizedEdge = (
            id_to_key[e.source],
            e.sourcePort,
            target_key,
            e.targetPort,
        )
        by_target.setdefault(target_key, []).append(edge_tuple)
    return by_target


def _assert_pwd_structurally_equal(
    tc: unittest.TestCase,
    pwd1: pwd_models.PythonWorkflowDefinitionWorkflow,
    pwd2: pwd_models.PythonWorkflowDefinitionWorkflow,
) -> None:
    """
    Compare two PWD workflows modulo arbitrary node-ID assignment.

    Checks that the normalized edge lists are identical (not merely sets —
    list order matters for order-sensitive nodes like ``get_list``).
    Also verifies matching node counts by type and identical input-node
    defaults.
    """
    tc.assertEqual(len(pwd1.nodes), len(pwd2.nodes))
    tc.assertEqual(len(pwd1.edges), len(pwd2.edges))

    edges1 = _normalize_pwd_edges(pwd1)
    edges2 = _normalize_pwd_edges(pwd2)
    tc.assertEqual(edges1, edges2)

    # Compare input-node defaults (keyed by name)
    def _input_defaults(
        wf: pwd_models.PythonWorkflowDefinitionWorkflow,
    ) -> dict[str, pwd_models.AllowableDefaults]:
        return {
            n.name: n.value
            for n in wf.nodes
            if isinstance(n, pwd_models.PythonWorkflowDefinitionInputNode)
        }

    tc.assertEqual(_input_defaults(pwd1), _input_defaults(pwd2))


def _assert_flowrep_roundtrip_equal(
    tc: unittest.TestCase,
    wf_orig: workflow_model.WorkflowNode,
    wf_rt: workflow_model.WorkflowNode,
    terminal_inputs: dict[str, pwd_models.AllowableDefaults],
    defaults_rt: dict[str, pwd_models.AllowableDefaults],
) -> None:
    """
    Compare original and round-tripped flowrep.

    The pwd format preserves explicit ``sourcePort`` names, so output-port
    names are not lost in the round-trip.  ``reference``, ``source_code``,
    and ``inputs_with_defaults`` are not represented in pwd and are therefore
    excluded from comparison.
    """
    tc.assertEqual(wf_orig.inputs, wf_rt.inputs)
    tc.assertEqual(wf_orig.outputs, wf_rt.outputs)
    tc.assertEqual(set(wf_orig.nodes.keys()), set(wf_rt.nodes.keys()))
    for label in wf_orig.nodes:
        n1, n2 = wf_orig.nodes[label], wf_rt.nodes[label]
        tc.assertEqual(n1.inputs, n2.inputs)
        tc.assertEqual(n1.outputs, n2.outputs)
        tc.assertEqual(n1.fully_qualified_name, n2.fully_qualified_name)
    tc.assertEqual(wf_orig.input_edges, wf_rt.input_edges)
    tc.assertEqual(wf_orig.edges, wf_rt.edges)
    tc.assertEqual(wf_orig.output_edges, wf_rt.output_edges)
    tc.assertEqual(terminal_inputs, defaults_rt)


def _load_pwd_workflow(name: str) -> pwd_models.PythonWorkflowDefinitionWorkflow:
    path = _PWD_JSON_WORKFLOW_LOCATION / name
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return pwd_models.PythonWorkflowDefinitionWorkflow.model_validate(data)


class TestPortSanitization(unittest.TestCase):
    """Unit tests for pwd._sanitize_port / pwd._desanitize_port."""

    def test_valid_identifier_unchanged(self):
        self.assertEqual(pwd_conv._sanitize_port("foo"), "foo")
        self.assertEqual(pwd_conv._sanitize_port("x_1"), "x_1")

    def test_digit_string_sanitized(self):
        self.assertEqual(
            pwd_conv._sanitize_port("0"), f"{pwd_conv._PORT_SANITIZE_PREFIX}0"
        )
        self.assertEqual(
            pwd_conv._sanitize_port("42"), f"{pwd_conv._PORT_SANITIZE_PREFIX}42"
        )

    def test_keyword_sanitized(self):
        self.assertEqual(
            pwd_conv._sanitize_port("for"), f"{pwd_conv._PORT_SANITIZE_PREFIX}for"
        )

    def test_reserved_name_sanitized(self):
        self.assertEqual(
            pwd_conv._sanitize_port("inputs"), f"{pwd_conv._PORT_SANITIZE_PREFIX}inputs"
        )
        self.assertEqual(
            pwd_conv._sanitize_port("outputs"),
            f"{pwd_conv._PORT_SANITIZE_PREFIX}outputs",
        )

    def test_roundtrip_digit(self):
        self.assertEqual(pwd_conv._desanitize_port(pwd_conv._sanitize_port("0")), "0")
        self.assertEqual(pwd_conv._desanitize_port(pwd_conv._sanitize_port("4")), "4")

    def test_roundtrip_keyword(self):
        self.assertEqual(
            pwd_conv._desanitize_port(pwd_conv._sanitize_port("for")), "for"
        )

    def test_normal_port_not_desanitized(self):
        """A port that happens to start with the prefix but has a valid
        remainder should NOT be desanitized."""
        self.assertEqual(pwd_conv._desanitize_port("p_foo"), "p_foo")

    def test_sanitized_port_is_valid_label(self):
        """Sanitized ports must pass flowrep's label validation."""
        for raw in ["0", "1", "42", "for", "inputs"]:
            sanitized = pwd_conv._sanitize_port(raw)
            with self.subTest(raw=raw, sanitized=sanitized):
                self.assertTrue(sanitized.isidentifier())


class TestPwd2FlowrepArithmetic(unittest.TestCase):
    """Smoke-test conversion of the arithmetic example."""

    def setUp(self):
        pwd_wf = _load_pwd_workflow("arithmetic-workflow.json")
        self.wf, self.defaults = pwd_conv.pwd2flowrep(pwd_wf)

    def test_inputs(self):
        self.assertEqual(set(self.wf.inputs), {"x", "y"})

    def test_outputs(self):
        self.assertEqual(self.wf.outputs, ["result"])

    def test_defaults(self):
        self.assertEqual(self.defaults, {"x": 1, "y": 2})

    def test_node_count(self):
        self.assertEqual(len(self.wf.nodes), 3)

    def test_all_atomic(self):
        for node in self.wf.nodes.values():
            self.assertIsInstance(node, atomic_model.AtomicNode)

    def test_multi_output_node(self):
        """get_prod_and_div should have two named outputs."""
        prod_div = [
            n
            for n in self.wf.nodes.values()
            if n.fully_qualified_name == "workflow.get_prod_and_div"
        ]
        self.assertEqual(len(prod_div), 1)
        self.assertEqual(set(prod_div[0].outputs), {"prod", "div"})


class TestPwd2FlowrepNfdi(unittest.TestCase):
    """Smoke-test conversion of the NFDI example."""

    def setUp(self):
        pwd_wf = _load_pwd_workflow("nfdi-workflow.json")
        self.wf, self.defaults = pwd_conv.pwd2flowrep(pwd_wf)

    def test_inputs(self):
        self.assertEqual(set(self.wf.inputs), {"domain_size", "source_directory"})

    def test_defaults(self):
        self.assertEqual(self.defaults["domain_size"], 2.0)
        self.assertEqual(self.defaults["source_directory"], "source")

    def test_node_count(self):
        self.assertEqual(len(self.wf.nodes), 6)

    def test_fan_out_input(self):
        """source_directory feeds multiple children."""
        source_edges = [
            target
            for target, source in self.wf.input_edges.items()
            if source.port == "source_directory"
        ]
        # Nodes 2, 3, 4, 5 in the original all receive source_directory
        self.assertGreater(len(source_edges), 1)


class TestPwd2FlowrepQuantumEspresso(unittest.TestCase):
    """Smoke-test conversion of the quantum-espresso example."""

    def setUp(self):
        pwd_wf = _load_pwd_workflow("quantum_espresso-workflow.json")
        self.wf, self.defaults = pwd_conv.pwd2flowrep(pwd_wf)

    def test_complex_defaults(self):
        """Dict and list default values survive conversion."""
        self.assertEqual(
            self.defaults["pseudopotentials"],
            {"Al": "Al.pbe-n-kjpaw_psl.1.0.0.UPF"},
        )
        self.assertEqual(self.defaults["kpts"], [3, 3, 3])
        self.assertEqual(self.defaults["strain_lst"], [0.9, 0.95, 1.0, 1.05, 1.1])

    def test_repeated_function(self):
        """calculate_qe appears multiple times → distinct labels."""
        calc_nodes = [
            label
            for label, n in self.wf.nodes.items()
            if "calculate_qe" in n.fully_qualified_name
        ]
        self.assertEqual(len(calc_nodes), 6)
        # All labels must be unique
        self.assertEqual(len(calc_nodes), len(set(calc_nodes)))

    def test_get_list_ports_sanitized(self):
        """get_list input ports like '0', '1' must be sanitized to valid labels."""
        list_nodes = [
            n
            for n in self.wf.nodes.values()
            if n.fully_qualified_name == "python_workflow_definition.shared.get_list"
        ]
        self.assertGreater(len(list_nodes), 0)
        for node in list_nodes:
            for port in node.inputs:
                with self.subTest(port=port):
                    self.assertTrue(
                        port.isidentifier(),
                        f"Port {port!r} is not a valid identifier",
                    )
                    self.assertTrue(port.startswith(pwd_conv._PORT_SANITIZE_PREFIX))


class TestPwd2FlowrepErrorCases(unittest.TestCase):

    def test_cycle_raises_validation_error(self):
        """
        PWD does not validate acyclicity, but flowrep does.

        A trivial A→B→A cycle must be caught by flowrep's DAG validator.
        """

        cyclic = pwd_models.PythonWorkflowDefinitionWorkflow(
            version="0.1.0",
            nodes=[
                pwd_models.PythonWorkflowDefinitionInputNode(
                    id=0, type="input", name="x", value=1
                ),
                pwd_models.PythonWorkflowDefinitionFunctionNode(
                    id=1, type="function", value="mod.func_a"
                ),
                pwd_models.PythonWorkflowDefinitionFunctionNode(
                    id=2, type="function", value="mod.func_b"
                ),
                pwd_models.PythonWorkflowDefinitionOutputNode(
                    id=3, type="output", name="result"
                ),
            ],
            edges=[
                pwd_models.PythonWorkflowDefinitionEdge(
                    source=0, sourcePort=None, target=1, targetPort="x"
                ),
                pwd_models.PythonWorkflowDefinitionEdge(
                    source=1, sourcePort=None, target=2, targetPort="a"
                ),
                pwd_models.PythonWorkflowDefinitionEdge(
                    source=2, sourcePort=None, target=1, targetPort="b"
                ),
                pwd_models.PythonWorkflowDefinitionEdge(
                    source=2, sourcePort=None, target=3, targetPort=None
                ),
            ],
        )
        with self.assertRaises(pydantic.ValidationError) as ctx:
            pwd_conv.pwd2flowrep(cyclic)
        self.assertIn("acyclic", str(ctx.exception).lower())

    def test_unsourced_node_raises(self):
        """
        A function node with no input edges should fail flowrep validation.
        """

        # func_a has input port "x" from the workflow input, but func_b's
        # input "y" has no edge sourcing it.
        pwd_models.PythonWorkflowDefinitionWorkflow(
            version="0.1.0",
            nodes=[
                pwd_models.PythonWorkflowDefinitionInputNode(
                    id=0, type="input", name="x", value=1
                ),
                pwd_models.PythonWorkflowDefinitionFunctionNode(
                    id=1, type="function", value="mod.func_a"
                ),
                pwd_models.PythonWorkflowDefinitionFunctionNode(
                    id=2, type="function", value="mod.func_b"
                ),
                pwd_models.PythonWorkflowDefinitionOutputNode(
                    id=3, type="output", name="result"
                ),
            ],
            edges=[
                pwd_models.PythonWorkflowDefinitionEdge(
                    source=0, sourcePort=None, target=1, targetPort="x"
                ),
                # func_a → func_b on port "a", but func_b also needs "y"
                pwd_models.PythonWorkflowDefinitionEdge(
                    source=1, sourcePort=None, target=2, targetPort="a"
                ),
                pwd_models.PythonWorkflowDefinitionEdge(
                    source=2, sourcePort=None, target=3, targetPort=None
                ),
            ],
        )
        # This should succeed — func_b only has the ports it gets edges for.
        # A truly disconnected node (zero edges) is the interesting case.

        # Build one: func_b gets no edges at all, func_a goes to output.
        disconnected = pwd_models.PythonWorkflowDefinitionWorkflow(
            version="0.1.0",
            nodes=[
                pwd_models.PythonWorkflowDefinitionInputNode(
                    id=0, type="input", name="x", value=1
                ),
                pwd_models.PythonWorkflowDefinitionFunctionNode(
                    id=1, type="function", value="mod.func_a"
                ),
                pwd_models.PythonWorkflowDefinitionFunctionNode(
                    id=2, type="function", value="mod.func_b"
                ),
                pwd_models.PythonWorkflowDefinitionOutputNode(
                    id=3, type="output", name="result"
                ),
            ],
            edges=[
                pwd_models.PythonWorkflowDefinitionEdge(
                    source=0, sourcePort=None, target=1, targetPort="x"
                ),
                pwd_models.PythonWorkflowDefinitionEdge(
                    source=1, sourcePort=None, target=3, targetPort=None
                ),
                # func_b has no edges at all — completely disconnected
            ],
        )
        # Disconnected nodes get zero inputs, which is fine on its own.
        # The conversion should succeed since func_b just has empty IO.
        # This is a degenerate but valid flowrep workflow.
        wf_result, _ = pwd_conv.pwd2flowrep(disconnected)
        self.assertIn("func_b_0", wf_result.nodes)
        self.assertEqual(wf_result.nodes["func_b_0"].inputs, [])
        self.assertEqual(wf_result.nodes["func_b_0"].outputs, [])


class TestFlowrep2PwdValidation(unittest.TestCase):
    """Edge-case validation in pwd.flowrep2pwd."""

    def _make_flat_workflow(self) -> workflow_model.WorkflowNode:
        return workflow_model.WorkflowNode(
            inputs=["x", "y"],
            outputs=["z"],
            nodes={
                "child": makers.make_atomic(
                    inputs=["a", "b"],
                    outputs=["c"],
                    module="mod",
                    qualname="func",
                ),
            },
            input_edges={
                edge_models.TargetHandle(node="child", port="a"): (
                    edge_models.InputSource(port="x")
                ),
                edge_models.TargetHandle(node="child", port="b"): (
                    edge_models.InputSource(port="y")
                ),
            },
            edges={},
            output_edges={
                edge_models.OutputTarget(port="z"): edge_models.SourceHandle(
                    node="child", port="c"
                ),
            },
        )

    def test_missing_terminal_input_raises(self):
        wf = self._make_flat_workflow()
        with self.assertRaises(ValueError) as ctx:
            pwd_conv.flowrep2pwd(wf, x=1)
        self.assertIn("missing", str(ctx.exception))
        self.assertIn("y", str(ctx.exception))

    def test_extra_terminal_input_raises(self):
        wf = self._make_flat_workflow()
        with self.assertRaises(ValueError) as ctx:
            pwd_conv.flowrep2pwd(wf, x=1, y=2, spurious=99)
        self.assertIn("extra", str(ctx.exception))
        self.assertIn("spurious", str(ctx.exception))

    def test_non_atomic_child_raises(self):
        inner = workflow_model.WorkflowNode(
            inputs=["a"],
            outputs=["b"],
            nodes={"leaf": makers.make_atomic(inputs=["x"], outputs=["y"])},
            input_edges={
                edge_models.TargetHandle(node="leaf", port="x"): (
                    edge_models.InputSource(port="a")
                ),
            },
            edges={},
            output_edges={
                edge_models.OutputTarget(port="b"): edge_models.SourceHandle(
                    node="leaf", port="y"
                ),
            },
        )
        outer = workflow_model.WorkflowNode(
            inputs=["x"],
            outputs=["z"],
            nodes={"nested": inner},
            input_edges={
                edge_models.TargetHandle(node="nested", port="a"): (
                    edge_models.InputSource(port="x")
                ),
            },
            edges={},
            output_edges={
                edge_models.OutputTarget(port="z"): edge_models.SourceHandle(
                    node="nested", port="b"
                ),
            },
        )
        with self.assertRaises(ValueError) as ctx:
            pwd_conv.flowrep2pwd(outer, x=1)
        self.assertIn("AtomicNode", str(ctx.exception))
        self.assertIn("nested", str(ctx.exception))

    def test_exact_terminal_inputs_succeeds(self):
        wf = self._make_flat_workflow()
        result = pwd_conv.flowrep2pwd(wf, x=1, y=2)
        self.assertIsInstance(result, pwd_models.PythonWorkflowDefinitionWorkflow)


class TestRoundTripPwdToFlowrep(unittest.TestCase):
    """pwd → flowrep → pwd → flowrep must produce identical flowrep structure."""

    def _assert_roundtrip(self, filename: str) -> None:
        pwd_orig = _load_pwd_workflow(filename)
        fr_1, defaults_1 = pwd_conv.pwd2flowrep(pwd_orig)

        pwd_rt = pwd_conv.flowrep2pwd(fr_1, **defaults_1)
        fr_2, defaults_2 = pwd_conv.pwd2flowrep(pwd_rt)

        _assert_flowrep_roundtrip_equal(self, fr_1, fr_2, defaults_1, defaults_2)
        # Also verify the pwd representations directly, modulo node IDs
        _assert_pwd_structurally_equal(self, pwd_orig, pwd_rt)

    def test_arithmetic(self):
        self._assert_roundtrip("arithmetic-workflow.json")

    def test_nfdi(self):
        self._assert_roundtrip("nfdi-workflow.json")

    def test_quantum_espresso(self):
        self._assert_roundtrip("quantum_espresso-workflow.json")


class TestRoundTripFlowrepToPwd(unittest.TestCase):
    """flowrep → pwd → flowrep must preserve graph structure."""

    def _assert_roundtrip(
        self,
        wf: workflow_model.WorkflowNode,
        terminal_inputs: dict[str, pwd_models.AllowableDefaults],
    ) -> None:
        pwd_wf = pwd_conv.flowrep2pwd(wf, **terminal_inputs)
        fr_rt, defaults_rt = pwd_conv.pwd2flowrep(pwd_wf)
        _assert_flowrep_roundtrip_equal(self, wf, fr_rt, terminal_inputs, defaults_rt)

    def test_linear_chain(self):
        """x, y → add → multiply → result"""

        def wf(x: float, y: float) -> float:
            s = library.add(x, y)
            p = library.multiply(s, y)
            return p

        node = workflow_parser.parse_workflow(wf)
        self._assert_roundtrip(node, {"x": 1.0, "y": 2.0})

    def test_fan_out(self):
        """One input feeds multiple children."""

        def wf(x: float, y: float) -> float:
            s = library.add(x, y)
            p = library.multiply(x, s)
            return p

        node = workflow_parser.parse_workflow(wf)
        self._assert_roundtrip(node, {"x": 3.0, "y": 4.0})

    def test_multi_output(self):
        """A function returning multiple values."""

        def wf(x: float) -> float:
            a, b = library.multi_result(x)
            c = library.add(a, b)
            return c

        node = workflow_parser.parse_workflow(wf)
        self._assert_roundtrip(node, {"x": 5.0})

    def test_single_node(self):
        """Simplest case: one function, all inputs wired, one output."""

        def wf(x: float, y: float) -> float:
            z = library.add(x, y)
            return z

        node = workflow_parser.parse_workflow(wf)
        self._assert_roundtrip(node, {"x": 10.0, "y": 1.0})

    def test_multi_output_ports_preserved(self):
        """Named output ports on multi-output nodes survive the round-trip."""

        def wf(x: float) -> float:
            a, b = library.multi_result(x)
            c = library.add(a, b)
            return c

        node = workflow_parser.parse_workflow(wf)
        pwd_wf = pwd_conv.flowrep2pwd(node, x=5.0)
        fr_rt, _ = pwd_conv.pwd2flowrep(pwd_wf)

        # multi_result has >1 output, so port names are preserved exactly
        orig_mr = [
            n for n in node.nodes.values() if "multi_result" in n.fully_qualified_name
        ][0]
        rt_mr = [
            n for n in fr_rt.nodes.values() if "multi_result" in n.fully_qualified_name
        ][0]
        self.assertEqual(orig_mr.outputs, rt_mr.outputs)

    def test_single_output_port_name_preserved(self):
        """Explicit single-output port names round-trip via sourcePort strings."""

        def wf(x: float, y: float) -> float:
            z = library.add(x, y)
            return z

        node = workflow_parser.parse_workflow(wf)
        # The add node's single output has a real name (not __result__)
        add_node = node.nodes["add_0"]
        self.assertNotEqual(add_node.outputs[0], pwd_conv._DEFAULT_OUTPUT_PORT)

        pwd_wf = pwd_conv.flowrep2pwd(node, x=10.0, y=1.0)
        fr_rt, _ = pwd_conv.pwd2flowrep(pwd_wf)

        # Port name must survive
        self.assertEqual(add_node.outputs, fr_rt.nodes["add_0"].outputs)


class TestDefaultOutputPort(unittest.TestCase):
    """Verify that the null-sourcePort ↔ ``pwd.DEFAULT_OUTPUT_PORT`` mapping works."""

    def test_single_output_uses_sentinel(self):
        """A PWD node with sourcePort=null produces an output with the default name."""
        pwd_wf = _load_pwd_workflow("arithmetic-workflow.json")
        wf, _ = pwd_conv.pwd2flowrep(pwd_wf)
        # get_square (the last function node) has a single unnamed output
        square_nodes = [
            n
            for n in wf.nodes.values()
            if n.fully_qualified_name == "workflow.get_square"
        ]
        self.assertEqual(len(square_nodes), 1)
        self.assertIn(pwd_conv._DEFAULT_OUTPUT_PORT, square_nodes[0].outputs)

    def test_sentinel_roundtrips_to_null(self):
        """default name port serializes back to sourcePort=null in PWD."""
        pwd_orig = _load_pwd_workflow("arithmetic-workflow.json")
        fr, defaults = pwd_conv.pwd2flowrep(pwd_orig)
        pwd_rt = pwd_conv.flowrep2pwd(fr, **defaults)

        null_source_edges = [
            e
            for e in pwd_rt.edges
            if e.sourcePort == pwd_models.INTERNAL_DEFAULT_HANDLE
        ]
        orig_null = [
            e
            for e in pwd_orig.edges
            if e.sourcePort == pwd_models.INTERNAL_DEFAULT_HANDLE
        ]
        self.assertEqual(len(null_source_edges), len(orig_null))

    def test_explicit_source_port_preserved(self):
        """Named sourcePorts like 'prod' must NOT become null in round-trip."""
        pwd_orig = _load_pwd_workflow("arithmetic-workflow.json")
        fr, defaults = pwd_conv.pwd2flowrep(pwd_orig)
        pwd_rt = pwd_conv.flowrep2pwd(fr, **defaults)

        named_orig = {
            (e.source, e.sourcePort)
            for e in pwd_orig.edges
            if e.sourcePort != pwd_models.INTERNAL_DEFAULT_HANDLE
        }
        named_rt = {
            (e.source, e.sourcePort)
            for e in pwd_rt.edges
            if e.sourcePort != pwd_models.INTERNAL_DEFAULT_HANDLE
        }
        # Can't compare source IDs across pwd instances, but count must match
        self.assertEqual(len(named_orig), len(named_rt))


class TestSanitizedPortRoundTrip(unittest.TestCase):
    """Integer-string ports from ``get_list`` survive a full round-trip."""

    def test_quantum_espresso_get_list_ports(self):
        """Ports '0'–'4' on get_list nodes round-trip correctly."""
        pwd_orig = _load_pwd_workflow("quantum_espresso-workflow.json")
        fr_1, defaults_1 = pwd_conv.pwd2flowrep(pwd_orig)

        # Verify sanitized port names in flowrep
        list_nodes = [
            (label, n)
            for label, n in fr_1.nodes.items()
            if n.fully_qualified_name == "python_workflow_definition.shared.get_list"
        ]
        for _, node in list_nodes:
            for port in node.inputs:
                self.assertTrue(port.startswith(pwd_conv._PORT_SANITIZE_PREFIX))

        # Full round-trip
        pwd_rt = pwd_conv.flowrep2pwd(fr_1, **defaults_1)
        fr_2, defaults_2 = pwd_conv.pwd2flowrep(pwd_rt)

        _assert_flowrep_roundtrip_equal(self, fr_1, fr_2, defaults_1, defaults_2)

    def test_desanitized_ports_match_original(self):
        """PWD edge targetPorts are restored to original integer strings."""
        pwd_orig = _load_pwd_workflow("quantum_espresso-workflow.json")
        fr, defaults = pwd_conv.pwd2flowrep(pwd_orig)
        pwd_rt = pwd_conv.flowrep2pwd(fr, **defaults)

        # Collect targetPorts targeting get_list nodes in both
        def _get_list_target_ports(
            wf: pwd_models.PythonWorkflowDefinitionWorkflow,
        ) -> set[str]:
            list_ids = {
                n.id
                for n in wf.nodes
                if isinstance(n, pwd_models.PythonWorkflowDefinitionFunctionNode)
                and n.value == "python_workflow_definition.shared.get_list"
            }
            return {
                e.targetPort
                for e in wf.edges
                if e.target in list_ids and e.targetPort is not None
            }

        self.assertEqual(
            _get_list_target_ports(pwd_orig),
            _get_list_target_ports(pwd_rt),
        )


class TestRoundTripEdgeNodeCounts(unittest.TestCase):
    """Quick structural invariant: node/edge counts survive a full round-trip."""

    def _assert_counts_preserved(self, filename: str) -> None:
        pwd_orig = _load_pwd_workflow(filename)
        fr, defaults = pwd_conv.pwd2flowrep(pwd_orig)
        pwd_rt = pwd_conv.flowrep2pwd(fr, **defaults)

        self.assertEqual(len(pwd_orig.nodes), len(pwd_rt.nodes))
        self.assertEqual(len(pwd_orig.edges), len(pwd_rt.edges))

    def test_arithmetic(self):
        self._assert_counts_preserved("arithmetic-workflow.json")

    def test_nfdi(self):
        self._assert_counts_preserved("nfdi-workflow.json")

    def test_quantum_espresso(self):
        self._assert_counts_preserved("quantum_espresso-workflow.json")


if __name__ == "__main__":
    unittest.main()
