"""
Cross-format execution parity for the pwd converter.

Structural round-tripping is covered by the unit tests.  These tests check the
thing that actually matters: that a converted recipe *computes the same answer*
under the target workflow manager — and that recipes which could not compute the
same answer are refused instead.

Requires ``PYTHONPATH=tests`` so that pwd's ``purepython`` runner can import
``flowrep_static.library`` by dotted path.
"""

from __future__ import annotations

import pathlib
import tempfile
import unittest

from flowrep import wfms
from flowrep.converters import python_workflow_definition as pwd_conv
from flowrep.parsers import workflow_parser

from flowrep_static import library, makers  # noqa: E402

try:
    from python_workflow_definition import models as pwd_models
    from python_workflow_definition import purepython

    _has_pwd = True
except ImportError:
    _has_pwd = False


def _run_purepython(wf: pwd_models.PythonWorkflowDefinitionWorkflow):
    """Execute a pwd workflow via its reference runner, which needs a file."""
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "workflow.json"
        path.write_text(wf.model_dump_json(), encoding="utf-8")
        return purepython.load_workflow_json(str(path))


def _linear(x: float, y: float) -> float:
    """s = x + y; p = s * y.  Every node has exactly one output."""
    s = library.typed_add(x, y)
    p = library.typed_multiply(s, y)
    return p


_KEYED_PWD_WORKFLOW = {
    "version": "0.1.0",
    "nodes": [
        {
            "id": 0,
            "type": "function",
            "value": "flowrep_static.library.prod_and_div_dict",
        },
        {"id": 1, "type": "function", "value": "flowrep_static.library.typed_add"},
        {"id": 2, "type": "input", "value": 6, "name": "x"},
        {"id": 3, "type": "input", "value": 3, "name": "y"},
        {"id": 4, "type": "output", "name": "result"},
    ],
    "edges": [
        {"target": 0, "targetPort": "x", "source": 2, "sourcePort": None},
        {"target": 0, "targetPort": "y", "source": 3, "sourcePort": None},
        {"target": 1, "targetPort": "x", "source": 0, "sourcePort": "prod"},
        {"target": 1, "targetPort": "y", "source": 0, "sourcePort": "div"},
        {"target": 4, "targetPort": None, "source": 1, "sourcePort": None},
    ],
}

_MONO_PWD_WORKFLOW = {
    "version": "0.1.0",
    "nodes": [
        {"id": 0, "type": "function", "value": "flowrep_static.library.typed_add"},
        {"id": 1, "type": "function", "value": "flowrep_static.library.typed_multiply"},
        {"id": 2, "type": "input", "value": 3.0, "name": "x"},
        {"id": 3, "type": "input", "value": 4.0, "name": "y"},
        {"id": 4, "type": "output", "name": "result"},
    ],
    "edges": [
        {"target": 0, "targetPort": "x", "source": 2, "sourcePort": None},
        {"target": 0, "targetPort": "y", "source": 3, "sourcePort": None},
        {"target": 1, "targetPort": "x", "source": 0, "sourcePort": None},
        {"target": 1, "targetPort": "y", "source": 3, "sourcePort": None},
        {"target": 4, "targetPort": None, "source": 1, "sourcePort": None},
    ],
}


@unittest.skipUnless(_has_pwd, "python_workflow_definition not installed")
class TestMonoOutputExecutionParity(unittest.TestCase):
    """Naked python == flowrep WfMS == pwd purepython, in both directions."""

    def setUp(self):
        self.expected = library.typed_multiply(library.typed_add(3.0, 4.0), 4.0)

    def test_expected_value(self):
        """Guard the guard: (3 + 4) * 4 == 28."""
        self.assertEqual(self.expected, 28.0)

    def test_flowrep_to_pwd(self):
        recipe = workflow_parser.parse_workflow(_linear)

        flowrep_result = wfms.run_recipe(recipe, x=3.0, y=4.0)
        self.assertEqual(
            flowrep_result.output_ports[recipe.outputs[0]].value, self.expected
        )

        pwd_wf = pwd_conv.flowrep2pwd(recipe, x=3.0, y=4.0)
        self.assertEqual(_run_purepython(pwd_wf), self.expected)

    def test_pwd_to_flowrep(self):
        pwd_wf = pwd_models.PythonWorkflowDefinitionWorkflow.model_validate(
            _MONO_PWD_WORKFLOW
        )
        self.assertEqual(_run_purepython(pwd_wf), self.expected)

        recipe, defaults = pwd_conv.pwd2flowrep(pwd_wf)
        flowrep_result = wfms.run_recipe(recipe, **defaults)
        self.assertEqual(flowrep_result.output_ports["result"].value, self.expected)


@unittest.skipUnless(_has_pwd, "python_workflow_definition not installed")
class TestCrossAxiomRefusal(unittest.TestCase):
    """Workflows each format runs happily but neither can hand to the other."""

    def test_pwd_keyed_workflow_runs_but_does_not_convert(self):
        """
        The pwd side is valid and runnable — it is only *unconvertible*.

        Before the guard this produced a flowrep recipe that silently computed
        ``'proddiv'`` instead of ``20.0``, by unpacking the dict's keys.
        """
        pwd_wf = pwd_models.PythonWorkflowDefinitionWorkflow.model_validate(
            _KEYED_PWD_WORKFLOW
        )
        self.assertEqual(_run_purepython(pwd_wf), 20.0)

        with self.assertRaises(pwd_conv.OutputContractError):
            pwd_conv.pwd2flowrep(pwd_wf)

    def test_flowrep_tuple_workflow_runs_but_does_not_convert(self):
        """
        The flowrep side is valid and runnable — it is only *unconvertible*.

        Before the guard the converted pwd workflow raised
        ``TypeError: tuple indices must be integers`` at run time.
        """

        def wf(x: float) -> float:
            a, b = library.multi_result(x)
            c = library.typed_add(a, b)
            return c

        # reference_free (rather than a bare parse_workflow) because `wf` is
        # locally scoped here; wfms.run_recipe would otherwise try to import it
        # by its <locals>-bearing qualified name and fail.
        recipe = makers.reference_free(wf)
        result = wfms.run_recipe(recipe, x=5.0)
        self.assertEqual(result.output_ports[recipe.outputs[0]].value, 10.0)

        with self.assertRaises(pwd_conv.OutputContractError):
            pwd_conv.flowrep2pwd(recipe, x=5.0)
