"""
The ``.draw()`` convenience methods hung off recipes and data objects.

These are the shortest path to a picture, so they get their own coverage
separate from the :mod:`flowrep.drawing.interface` callables they delegate to.
"""

import subprocess
import sys
import textwrap
import unittest

from flowrep import drawing, edge_models, std
from flowrep.prospective import helper_models, while_recipe, workflow_recipe
from flowrep.retrospective import datastructures

from flowrep_static import library

try:
    import graphviz  # noqa: F401

    _has_graphviz = True
except ImportError:
    _has_graphviz = False


def _nested_workflow() -> workflow_recipe.WorkflowRecipe:
    """A workflow whose child is itself a workflow, so ``depth`` actually bites."""
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


@unittest.skipUnless(_has_graphviz, "graphviz not installed")
class TestRecipeDraw(unittest.TestCase):
    def test_matches_the_module_level_drawer(self):
        recipe = std.neg.flowrep_recipe
        self.assertEqual(recipe.draw().source, drawing.draw_prospective(recipe).source)

    def test_default_depth_is_the_prospective_default(self):
        recipe = _nested_workflow()
        self.assertEqual(
            recipe.draw().source, drawing.draw_prospective(recipe, depth=1).source
        )

    def test_explicit_depth_is_forwarded(self):
        recipe = _nested_workflow()
        self.assertEqual(
            recipe.draw(depth=0).source,
            drawing.draw_prospective(recipe, depth=0).source,
        )

    def test_depth_actually_changes_the_drawing(self):
        recipe = _nested_workflow()
        self.assertNotEqual(recipe.draw(depth=0).source, recipe.draw(depth=1).source)

    def test_negative_depth_rejected(self):
        with self.assertRaises(ValueError):
            std.neg.flowrep_recipe.draw(depth=-1)

    def test_available_on_a_flow_control_recipe(self):
        """``draw`` lives on the base class, so every recipe type inherits it."""
        recipe = while_recipe.WhileRecipe(
            inputs=["x"],
            outputs=["x"],
            case=helper_models.ConditionalCase(
                condition=helper_models.LabeledRecipe(
                    label="cond", recipe=library.is_positive.flowrep_recipe
                ),
                body=helper_models.LabeledRecipe(
                    label="body", recipe=library.loop_inc.flowrep_recipe
                ),
            ),
            input_edges={
                edge_models.TargetHandle(
                    node="cond", port="n"
                ): edge_models.InputSource(port="x"),
                edge_models.TargetHandle(
                    node="body", port="x"
                ): edge_models.InputSource(port="x"),
            },
            output_edges={
                edge_models.OutputTarget(port="x"): edge_models.SourceHandle(
                    node="body", port="y"
                )
            },
        )
        self.assertIn("cond_i", recipe.draw().source)


@unittest.skipUnless(_has_graphviz, "graphviz not installed")
class TestDataDraw(unittest.TestCase):
    def setUp(self):
        self.data = datastructures.recipe2data(library.simple_workflow.flowrep_recipe)

    def test_matches_the_module_level_drawer(self):
        self.assertEqual(
            self.data.draw().source, drawing.draw_retrospective(self.data).source
        )

    def test_default_depth_is_the_retrospective_default(self):
        self.assertEqual(
            self.data.draw().source,
            drawing.draw_retrospective(self.data, depth=0).source,
        )

    def test_explicit_depth_is_forwarded(self):
        self.assertEqual(
            self.data.draw(depth=1).source,
            drawing.draw_retrospective(self.data, depth=1).source,
        )

    def test_negative_depth_rejected(self):
        with self.assertRaises(ValueError):
            self.data.draw(depth=-1)

    def test_sits_alongside_view(self):
        """``view`` shows the data; ``draw`` shows the shape."""
        self.assertTrue(callable(self.data.view))
        self.assertTrue(callable(self.data.draw))


_WITHOUT_GRAPHVIZ = textwrap.dedent("""
    import sys


    class _BlockGraphviz:
        def find_spec(self, name, path=None, target=None):
            if name == "graphviz":
                raise ImportError("graphviz is blocked for this test")
            return None


    sys.meta_path.insert(0, _BlockGraphviz())

    from flowrep import std
    from flowrep.retrospective import datastructures

    recipe = std.neg.flowrep_recipe
    for drawable in (recipe, datastructures.recipe2data(recipe)):
        try:
            drawable.draw()
        except Exception as error:
            print(f"{type(drawable).__name__}|{type(error).__name__}|{error}")
        else:
            print(f"{type(drawable).__name__}|NO_ERROR|")
    """)


class TestMissingDependencyMessage(unittest.TestCase):
    """
    Both methods must fail helpfully, not cryptically, in a bare install.

    Run in a subprocess so blocking ``graphviz`` cannot corrupt the interpreter
    state the rest of the suite shares.
    """

    @classmethod
    def setUpClass(cls):
        result = subprocess.run(
            [sys.executable, "-c", _WITHOUT_GRAPHVIZ],
            capture_output=True,
            text=True,
            check=True,
        )
        cls.lines = [
            line.split("|", 2) for line in result.stdout.strip().splitlines() if line
        ]

    def test_both_drawables_raise(self):
        self.assertEqual(len(self.lines), 2)
        for name, error, _ in self.lines:
            with self.subTest(drawable=name):
                self.assertEqual(error, "ImportAlarmError")

    def test_message_names_the_package(self):
        for name, _, message in self.lines:
            with self.subTest(drawable=name):
                self.assertIn("graphviz", message)

    def test_message_names_the_pip_route_and_its_binary_caveat(self):
        for name, _, message in self.lines:
            with self.subTest(drawable=name):
                self.assertIn("pip install flowrep[drawing]", message)
                self.assertIn("dot", message)

    def test_message_names_the_conda_route(self):
        for name, _, message in self.lines:
            with self.subTest(drawable=name):
                self.assertIn("conda install", message)
                self.assertIn("python-graphviz", message)


if __name__ == "__main__":
    unittest.main()
