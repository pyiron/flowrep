import subprocess
import sys
import unittest

import flowrep
from flowrep import edge_models, std
from flowrep.prospective import workflow_recipe
from flowrep.retrospective import datastructures

from flowrep_static import library


def _nested_workflow() -> workflow_recipe.WorkflowRecipe:
    """A workflow whose child is itself a workflow, so depth actually bites.

    ``simple_workflow`` alone will not do: its only child is atomic, and an
    atomic node is a leaf at every depth, so depth 0 and depth 1 would render
    identically.
    """
    inner = library.simple_workflow.flowrep_recipe
    return workflow_recipe.WorkflowRecipe(
        inputs=["p", "q"],
        outputs=["r"],
        nodes={"inner": inner},
        input_edges={
            edge_models.TargetHandle(node="inner", port="a"): edge_models.InputSource(
                port="p"
            ),
            edge_models.TargetHandle(node="inner", port="b"): edge_models.InputSource(
                port="q"
            ),
        },
        edges={},
        output_edges={
            edge_models.OutputTarget(port="r"): edge_models.SourceHandle(
                node="inner", port=inner.outputs[0]
            )
        },
    )


class TestDispatch(unittest.TestCase):
    def test_recipe_dispatches_prospective(self):
        self.assertIn("neg", flowrep.draw(std.neg.flowrep_recipe).source)

    def test_data_dispatches_retrospective(self):
        data = datastructures.recipe2data(library.decrement.flowrep_recipe)
        self.assertIn("decrement", flowrep.draw(data).source)

    def test_unknown_type_raises(self):
        with self.assertRaises(TypeError):
            flowrep.draw("not a graph")


class TestDepthDefaults(unittest.TestCase):
    def setUp(self):
        self.recipe = _nested_workflow()

    def test_prospective_default_is_one(self):
        self.assertEqual(
            flowrep.drawing.draw_prospective(self.recipe).source,
            flowrep.drawing.draw_prospective(self.recipe, depth=1).source,
        )

    def test_explicit_depth_overrides(self):
        shallow = flowrep.drawing.draw_prospective(self.recipe, depth=0).source
        deep = flowrep.drawing.draw_prospective(self.recipe, depth=1).source
        self.assertNotEqual(shallow, deep)

    def test_retrospective_default_is_zero(self):
        data = datastructures.recipe2data(self.recipe)
        self.assertEqual(
            flowrep.drawing.draw_retrospective(data).source,
            flowrep.drawing.draw_retrospective(data, depth=0).source,
        )

    def test_dispatcher_forwards_explicit_depth(self):
        self.assertEqual(
            flowrep.draw(self.recipe, depth=0).source,
            flowrep.drawing.draw_prospective(self.recipe, depth=0).source,
        )


class TestExposure(unittest.TestCase):
    def test_draw_is_top_level(self):
        self.assertTrue(callable(flowrep.draw))

    def test_draw_is_in_tools(self):
        self.assertIs(flowrep.tools.draw, flowrep.draw)

    def test_specific_drawers_stay_in_the_subpackage(self):
        self.assertFalse(hasattr(flowrep, "draw_prospective"))
        self.assertTrue(callable(flowrep.drawing.draw_prospective))
        self.assertTrue(callable(flowrep.drawing.draw_retrospective))


class TestImportSafetyWithoutGraphviz(unittest.TestCase):
    """``import flowrep`` must succeed even without graphviz installed.

    ``flowrep/__init__.py`` transitively imports ``flowrep.drawing.render``,
    which imports graphviz. The ``ImportAlarm(..., raise_exception=True)``
    guard there is meant to swallow the ``ImportError`` at import time and
    re-raise only when a drawing callable is actually invoked. This is run in
    a subprocess, with a meta path finder blocking graphviz, so it exercises a
    genuinely fresh import rather than relying on already-imported modules.
    """

    _SCRIPT = """
import sys
import importlib.abc


class _BlockGraphviz(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "graphviz" or name.startswith("graphviz."):
            raise ImportError("graphviz blocked for test")
        return None


sys.meta_path.insert(0, _BlockGraphviz())

import flowrep
from flowrep import std

assert callable(flowrep.draw), "flowrep.draw must be importable without graphviz"

try:
    flowrep.draw(std.neg.flowrep_recipe)
except ImportError:
    print("RAISED")
else:
    print("DID NOT RAISE")
"""

    def test_import_succeeds_and_draw_raises_without_graphviz(self):
        result = subprocess.run(
            [sys.executable, "-c", self._SCRIPT],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(result.stdout.strip(), "RAISED")
