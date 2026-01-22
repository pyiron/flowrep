import unittest

from flowrep.models import edge_models


class TestHandleModels(unittest.TestCase):
    """Tests for HandleModel subclasses and their type constraints."""

    def test_source_handle_requires_node(self):
        """SourceHandle must have a non-None node."""
        h = edge_models.SourceHandle(node="child", port="out")
        self.assertEqual(h.node, "child")
        self.assertEqual(h.port, "out")

    def test_target_handle_requires_node(self):
        """TargetHandle must have a non-None node."""
        h = edge_models.TargetHandle(node="child", port="inp")
        self.assertEqual(h.node, "child")
        self.assertEqual(h.port, "inp")

    def test_input_source_has_none_node(self):
        """InputSource always has node=None."""
        h = edge_models.InputSource(port="x")
        self.assertIsNone(h.node)
        self.assertEqual(h.port, "x")

    def test_output_target_has_none_node(self):
        """OutputTarget always has node=None."""
        h = edge_models.OutputTarget(port="y")
        self.assertIsNone(h.node)
        self.assertEqual(h.port, "y")

    def test_handle_serialization(self):
        """Handles serialize to dot-notation strings."""
        self.assertEqual(
            edge_models.SourceHandle(node="n", port="p").model_dump(mode="json"), "n.p"
        )
        self.assertEqual(edge_models.InputSource(port="x").model_dump(mode="json"), "x")

    def test_handle_deserialization(self):
        """Handles deserialize from dot-notation strings."""
        h = edge_models.SourceHandle.model_validate("child.output")
        self.assertEqual(h.node, "child")
        self.assertEqual(h.port, "output")

        h = edge_models.InputSource.model_validate("workflow_input")
        self.assertIsNone(h.node)
        self.assertEqual(h.port, "workflow_input")
