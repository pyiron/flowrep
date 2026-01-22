import unittest

from flowrep.models import edges_model


class TestHandleModels(unittest.TestCase):
    """Tests for HandleModel subclasses and their type constraints."""

    def test_source_handle_requires_node(self):
        """SourceHandle must have a non-None node."""
        h = edges_model.SourceHandle(node="child", port="out")
        self.assertEqual(h.node, "child")
        self.assertEqual(h.port, "out")

    def test_target_handle_requires_node(self):
        """TargetHandle must have a non-None node."""
        h = edges_model.TargetHandle(node="child", port="inp")
        self.assertEqual(h.node, "child")
        self.assertEqual(h.port, "inp")

    def test_input_source_has_none_node(self):
        """InputSource always has node=None."""
        h = edges_model.InputSource(port="x")
        self.assertIsNone(h.node)
        self.assertEqual(h.port, "x")

    def test_output_target_has_none_node(self):
        """OutputTarget always has node=None."""
        h = edges_model.OutputTarget(port="y")
        self.assertIsNone(h.node)
        self.assertEqual(h.port, "y")

    def test_handle_serialization(self):
        """Handles serialize to dot-notation strings."""
        self.assertEqual(
            edges_model.SourceHandle(node="n", port="p").model_dump(mode="json"), "n.p"
        )
        self.assertEqual(edges_model.InputSource(port="x").model_dump(mode="json"), "x")

    def test_handle_deserialization(self):
        """Handles deserialize from dot-notation strings."""
        h = edges_model.SourceHandle.model_validate("child.output")
        self.assertEqual(h.node, "child")
        self.assertEqual(h.port, "output")

        h = edges_model.InputSource.model_validate("workflow_input")
        self.assertIsNone(h.node)
        self.assertEqual(h.port, "workflow_input")
