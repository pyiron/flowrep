import dataclasses
import unittest

from flowrep import base_models
from flowrep.drawing import model


def _port(label: str) -> model.DrawPort:
    return model.DrawPort(label=label, hint=None, has_default=False, badge=None)


def _leaf(path: str, label: str) -> model.DrawNode:
    return model.DrawNode(
        path=path,
        label=label,
        kind=base_models.RecipeElementType.ATOMIC,
        subtitle=None,
        inputs=(_port("x"),),
        outputs=(_port("y"),),
        children=(),
        edges=(),
    )


class TestDrawNode(unittest.TestCase):
    def test_is_frozen(self):
        node = _leaf("a", "a")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            node.path = "b"  # type: ignore[misc]

    def test_leaf_has_no_children(self):
        self.assertTrue(_leaf("a", "a").is_leaf)

    def test_composite_is_not_leaf(self):
        parent = model.DrawNode(
            path="",
            label="wf",
            kind=base_models.RecipeElementType.WORKFLOW,
            subtitle=None,
            inputs=(),
            outputs=(),
            children=(_leaf("a", "a"),),
            edges=(),
        )
        self.assertFalse(parent.is_leaf)

    def test_walk_is_self_then_descendants(self):
        grandchild = _leaf("a.b", "b")
        child = dataclasses.replace(_leaf("a", "a"), children=(grandchild,))
        root = dataclasses.replace(_leaf("", "root"), children=(child,))
        self.assertEqual([n.path for n in root.walk()], ["", "a", "a.b"])

    def test_optional_fields_default(self):
        node = _leaf("a", "a")
        self.assertEqual(node.groups, ())
        self.assertIsNone(node.note)


class TestDrawEdge(unittest.TestCase):
    def test_defaults_to_unconditional(self):
        edge = model.DrawEdge(
            source=model.PortRef("a", base_models.IOTypes.OUTPUTS, "y"),
            target=model.PortRef("b", base_models.IOTypes.INPUTS, "x"),
        )
        self.assertFalse(edge.conditional)

    def test_is_hashable(self):
        """Edges land in sets during cycle detection."""
        edge = model.DrawEdge(
            source=model.PortRef("a", base_models.IOTypes.OUTPUTS, "y"),
            target=model.PortRef("b", base_models.IOTypes.INPUTS, "x"),
        )
        self.assertEqual(len({edge, edge}), 1)


class TestPortRef(unittest.TestCase):
    def test_lexical_path(self):
        ref = model.PortRef("sub.body_0", base_models.IOTypes.INPUTS, "x")
        self.assertEqual(ref.lexical_path, "sub.body_0.inputs.x")

    def test_empty_node_path_means_enclosing_node(self):
        ref = model.PortRef("", base_models.IOTypes.OUTPUTS, "y")
        self.assertEqual(ref.lexical_path, "outputs.y")
