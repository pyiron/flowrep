"""Unit tests for flowrep.models.subgraph_protocols"""

import unittest
from unittest import mock

from flowrep.models import edge_models, subgraph_protocols


class MockNode:
    """Minimal node for testing."""

    def __init__(self, inputs: list[str], outputs: list[str]):
        self.inputs = inputs
        self.outputs = outputs


class TestValidateInputSources(unittest.TestCase):
    """Tests for validate_input_sources."""

    def test_valid_sources(self):
        macro = mock.Mock(spec=subgraph_protocols.HasSubgraphInput)
        macro.inputs = ["x", "y"]
        macro.input_edges = {
            edge_models.TargetHandle(node="a", port="inp"): edge_models.InputSource(
                port="x"
            ),
        }
        subgraph_protocols.validate_input_sources(macro)  # Should not raise

    def test_invalid_source_port(self):
        macro = mock.Mock(spec=subgraph_protocols.HasSubgraphInput)
        macro.inputs = ["x"]
        macro.input_edges = {
            edge_models.TargetHandle(node="a", port="inp"): edge_models.InputSource(
                port="nonexistent"
            ),
        }
        with self.assertRaises(ValueError) as ctx:
            subgraph_protocols.validate_input_sources(macro)
        self.assertIn("nonexistent", str(ctx.exception))

    def test_empty_edges(self):
        macro = mock.Mock(spec=subgraph_protocols.HasSubgraphInput)
        macro.inputs = ["x"]
        macro.input_edges = {}
        subgraph_protocols.validate_input_sources(macro)  # Should not raise


class TestValidateInputTargets(unittest.TestCase):
    """Tests for validate_input_targets (static subgraph)."""

    def test_valid_targets(self):
        macro = mock.Mock(spec=subgraph_protocols.HasStaticSubgraph)
        macro.nodes = {"a": MockNode(inputs=["inp"], outputs=["out"])}
        macro.input_edges = {
            edge_models.TargetHandle(node="a", port="inp"): edge_models.InputSource(
                port="x"
            ),
        }
        subgraph_protocols.validate_input_targets(macro)  # Should not raise

    def test_invalid_target_node(self):
        macro = mock.Mock(spec=subgraph_protocols.HasStaticSubgraph)
        macro.nodes = {"a": MockNode(inputs=["inp"], outputs=["out"])}
        macro.input_edges = {
            edge_models.TargetHandle(
                node="nonexistent", port="inp"
            ): edge_models.InputSource(port="x"),
        }
        with self.assertRaises(ValueError) as ctx:
            subgraph_protocols.validate_input_targets(macro)
        self.assertIn("nonexistent", str(ctx.exception))

    def test_invalid_target_port(self):
        macro = mock.Mock(spec=subgraph_protocols.HasStaticSubgraph)
        macro.nodes = {"a": MockNode(inputs=["inp"], outputs=["out"])}
        macro.input_edges = {
            edge_models.TargetHandle(
                node="a", port="wrong_port"
            ): edge_models.InputSource(port="x"),
        }
        with self.assertRaises(ValueError) as ctx:
            subgraph_protocols.validate_input_targets(macro)
        self.assertIn("wrong_port", str(ctx.exception))


class TestValidateProspectiveInputTargets(unittest.TestCase):
    """Tests for validate_prospective_input_targets."""

    def test_valid_prospective_targets(self):
        macro = mock.Mock(spec=subgraph_protocols.BuildsSubgraph)
        macro.prospective_nodes = {"a": MockNode(inputs=["inp"], outputs=["out"])}
        macro.input_edges = {
            edge_models.TargetHandle(node="a", port="inp"): edge_models.InputSource(
                port="x"
            ),
        }
        subgraph_protocols.validate_prospective_input_targets(macro)

    def test_invalid_prospective_target_node(self):
        macro = mock.Mock(spec=subgraph_protocols.BuildsSubgraph)
        macro.prospective_nodes = {"a": MockNode(inputs=["inp"], outputs=["out"])}
        macro.input_edges = {
            edge_models.TargetHandle(
                node="nonexistent", port="inp"
            ): edge_models.InputSource(port="x"),
        }
        with self.assertRaises(ValueError):
            subgraph_protocols.validate_prospective_input_targets(macro)


class TestValidateOutputTargets(unittest.TestCase):
    """Tests for validate_output_targets."""

    def test_valid_output_targets(self):
        macro = mock.Mock(spec=subgraph_protocols.HasStaticSubgraphOutput)
        macro.outputs = ["y"]
        macro.output_edges = {
            edge_models.OutputTarget(port="y"): edge_models.SourceHandle(
                node="a", port="out"
            ),
        }
        subgraph_protocols.validate_output_targets(macro)

    def test_valid_multiple_outputs(self):
        macro = mock.Mock(spec=subgraph_protocols.HasStaticSubgraphOutput)
        macro.outputs = ["y", "z"]
        macro.output_edges = {
            edge_models.OutputTarget(port="y"): edge_models.SourceHandle(
                node="a", port="out1"
            ),
            edge_models.OutputTarget(port="z"): edge_models.SourceHandle(
                node="a", port="out2"
            ),
        }
        subgraph_protocols.validate_output_targets(macro)

    def test_invalid_output_target(self):
        macro = mock.Mock(spec=subgraph_protocols.HasStaticSubgraphOutput)
        macro.outputs = ["y"]
        macro.output_edges = {
            edge_models.OutputTarget(port="nonexistent"): edge_models.SourceHandle(
                node="a", port="out"
            ),
        }
        with self.assertRaises(ValueError) as ctx:
            subgraph_protocols.validate_output_targets(macro)
        self.assertIn("nonexistent", str(ctx.exception))

    def test_missing_output_edge(self):
        macro = mock.Mock(spec=subgraph_protocols.HasStaticSubgraphOutput)
        macro.outputs = ["y", "z"]
        macro.output_edges = {
            edge_models.OutputTarget(port="y"): edge_models.SourceHandle(
                node="a", port="out"
            ),
            # Missing edge for "z"
        }
        with self.assertRaises(ValueError) as ctx:
            subgraph_protocols.validate_output_targets(macro)
        self.assertIn("Missing", str(ctx.exception))
        self.assertIn("z", str(ctx.exception))

    def test_empty_outputs_and_edges(self):
        macro = mock.Mock(spec=subgraph_protocols.HasStaticSubgraphOutput)
        macro.outputs = []
        macro.output_edges = {}
        subgraph_protocols.validate_output_targets(macro)


class TestValidateProspectiveOutputTargets(unittest.TestCase):
    """Tests for validate_prospective_output_targets."""

    def test_valid_prospective_output_targets(self):
        macro = mock.Mock(spec=subgraph_protocols.BuildsSubgraphWithDynamicOutput)
        macro.outputs = ["y"]
        macro.prospective_output_edges = {
            edge_models.OutputTarget(port="y"): [
                edge_models.SourceHandle(node="a", port="out")
            ],
        }
        subgraph_protocols.validate_prospective_output_targets(macro)

    def test_valid_multiple_outputs(self):
        macro = mock.Mock(spec=subgraph_protocols.BuildsSubgraphWithDynamicOutput)
        macro.outputs = ["y", "z"]
        macro.prospective_output_edges = {
            edge_models.OutputTarget(port="y"): [
                edge_models.SourceHandle(node="a", port="out1")
            ],
            edge_models.OutputTarget(port="z"): [
                edge_models.SourceHandle(node="a", port="out2")
            ],
        }
        subgraph_protocols.validate_prospective_output_targets(macro)

    def test_invalid_prospective_output_target(self):
        macro = mock.Mock(spec=subgraph_protocols.BuildsSubgraphWithDynamicOutput)
        macro.outputs = ["y"]
        macro.prospective_output_edges = {
            edge_models.OutputTarget(port="nonexistent"): [
                edge_models.SourceHandle(node="a", port="out")
            ],
        }
        with self.assertRaises(ValueError) as ctx:
            subgraph_protocols.validate_prospective_output_targets(macro)
        self.assertIn("nonexistent", str(ctx.exception))

    def test_missing_prospective_output_edge(self):
        macro = mock.Mock(spec=subgraph_protocols.BuildsSubgraphWithDynamicOutput)
        macro.outputs = ["y", "z"]
        macro.prospective_output_edges = {
            edge_models.OutputTarget(port="y"): [
                edge_models.SourceHandle(node="a", port="out")
            ],
            # Missing edge for "z"
        }
        with self.assertRaises(ValueError) as ctx:
            subgraph_protocols.validate_prospective_output_targets(macro)
        self.assertIn("Missing", str(ctx.exception))
        self.assertIn("z", str(ctx.exception))

    def test_empty_outputs_and_edges(self):
        macro = mock.Mock(spec=subgraph_protocols.BuildsSubgraphWithDynamicOutput)
        macro.outputs = []
        macro.prospective_output_edges = {}
        subgraph_protocols.validate_prospective_output_targets(macro)


class TestValidateOutputSources(unittest.TestCase):
    """Tests for validate_output_sources."""

    def test_valid_output_sources(self):
        macro = mock.Mock(spec=subgraph_protocols.HasStaticSubgraph)
        macro.nodes = {"a": MockNode(inputs=["inp"], outputs=["out"])}
        macro.output_edges = {
            edge_models.OutputTarget(port="y"): edge_models.SourceHandle(
                node="a", port="out"
            ),
        }
        subgraph_protocols.validate_output_sources(macro)

    def test_invalid_source_node(self):
        macro = mock.Mock(spec=subgraph_protocols.HasStaticSubgraph)
        macro.nodes = {"a": MockNode(inputs=["inp"], outputs=["out"])}
        macro.output_edges = {
            edge_models.OutputTarget(port="y"): edge_models.SourceHandle(
                node="nonexistent", port="out"
            ),
        }
        with self.assertRaises(ValueError) as ctx:
            subgraph_protocols.validate_output_sources(macro)
        self.assertIn("nonexistent", str(ctx.exception))

    def test_invalid_source_port(self):
        macro = mock.Mock(spec=subgraph_protocols.HasStaticSubgraph)
        macro.nodes = {"a": MockNode(inputs=["inp"], outputs=["out"])}
        macro.output_edges = {
            edge_models.OutputTarget(port="y"): edge_models.SourceHandle(
                node="a", port="wrong_port"
            ),
        }
        with self.assertRaises(ValueError) as ctx:
            subgraph_protocols.validate_output_sources(macro)
        self.assertIn("wrong_port", str(ctx.exception))


class TestValidateOutputSourcesFromProspectiveNodes(unittest.TestCase):
    """Tests for validate_output_sources_from_prospective_nodes."""

    def test_valid_output_sources(self):
        macro = mock.Mock(spec=subgraph_protocols.BuildsSubgraphWithStaticOutput)
        macro.prospective_nodes = {"a": MockNode(inputs=["inp"], outputs=["out"])}
        macro.output_edges = {
            edge_models.OutputTarget(port="y"): edge_models.SourceHandle(
                node="a", port="out"
            ),
        }
        subgraph_protocols.validate_output_sources_from_prospective_nodes(macro)

    def test_invalid_source_node(self):
        macro = mock.Mock(spec=subgraph_protocols.BuildsSubgraphWithStaticOutput)
        macro.prospective_nodes = {"a": MockNode(inputs=["inp"], outputs=["out"])}
        macro.output_edges = {
            edge_models.OutputTarget(port="y"): edge_models.SourceHandle(
                node="nonexistent", port="out"
            ),
        }
        with self.assertRaises(ValueError) as ctx:
            subgraph_protocols.validate_output_sources_from_prospective_nodes(macro)
        self.assertIn("nonexistent", str(ctx.exception))

    def test_invalid_source_port(self):
        macro = mock.Mock(spec=subgraph_protocols.BuildsSubgraphWithStaticOutput)
        macro.prospective_nodes = {"a": MockNode(inputs=["inp"], outputs=["out"])}
        macro.output_edges = {
            edge_models.OutputTarget(port="y"): edge_models.SourceHandle(
                node="a", port="wrong_port"
            ),
        }
        with self.assertRaises(ValueError) as ctx:
            subgraph_protocols.validate_output_sources_from_prospective_nodes(macro)
        self.assertIn("wrong_port", str(ctx.exception))

    def test_multiple_output_edges(self):
        macro = mock.Mock(spec=subgraph_protocols.BuildsSubgraphWithStaticOutput)
        macro.prospective_nodes = {
            "a": MockNode(inputs=[], outputs=["out1", "out2"]),
            "b": MockNode(inputs=[], outputs=["out"]),
        }
        macro.output_edges = {
            edge_models.OutputTarget(port="x"): edge_models.SourceHandle(
                node="a", port="out1"
            ),
            edge_models.OutputTarget(port="y"): edge_models.SourceHandle(
                node="b", port="out"
            ),
        }
        subgraph_protocols.validate_output_sources_from_prospective_nodes(macro)


class TestValidateProspectiveOutputSources(unittest.TestCase):
    """Tests for validate_prospective_output_sources."""

    def test_valid_prospective_sources(self):
        macro = mock.Mock(spec=subgraph_protocols.BuildsSubgraphWithDynamicOutput)
        macro.prospective_nodes = {
            "a": MockNode(inputs=["inp"], outputs=["out1", "out2"]),
            "b": MockNode(inputs=["inp"], outputs=["out"]),
        }
        macro.prospective_output_edges = {
            edge_models.OutputTarget(port="y"): [
                edge_models.SourceHandle(node="a", port="out1"),
                edge_models.SourceHandle(node="b", port="out"),
            ],
        }
        subgraph_protocols.validate_prospective_output_sources(macro)

    def test_duplicate_source_nodes_rejected(self):
        macro = mock.Mock(spec=subgraph_protocols.BuildsSubgraphWithDynamicOutput)
        macro.prospective_nodes = {"a": MockNode(inputs=[], outputs=["out1", "out2"])}
        macro.prospective_output_edges = {
            edge_models.OutputTarget(port="y"): [
                edge_models.SourceHandle(node="a", port="out1"),
                edge_models.SourceHandle(node="a", port="out2"),  # Same node twice
            ],
        }
        with self.assertRaises(ValueError) as ctx:
            subgraph_protocols.validate_prospective_output_sources(macro)
        self.assertIn("Duplicate", str(ctx.exception))

    def test_invalid_prospective_source_node(self):
        macro = mock.Mock(spec=subgraph_protocols.BuildsSubgraphWithDynamicOutput)
        macro.prospective_nodes = {"a": MockNode(inputs=[], outputs=["out"])}
        macro.prospective_output_edges = {
            edge_models.OutputTarget(port="y"): [
                edge_models.SourceHandle(node="nonexistent", port="out"),
            ],
        }
        with self.assertRaises(ValueError) as ctx:
            subgraph_protocols.validate_prospective_output_sources(macro)
        self.assertIn("nonexistent", str(ctx.exception))

    def test_invalid_prospective_source_port(self):
        macro = mock.Mock(spec=subgraph_protocols.BuildsSubgraphWithDynamicOutput)
        macro.prospective_nodes = {"a": MockNode(inputs=[], outputs=["out"])}
        macro.prospective_output_edges = {
            edge_models.OutputTarget(port="y"): [
                edge_models.SourceHandle(node="a", port="wrong_port"),
            ],
        }
        with self.assertRaises(ValueError) as ctx:
            subgraph_protocols.validate_prospective_output_sources(macro)
        self.assertIn("wrong_port", str(ctx.exception))


class TestValidateExtantEdges(unittest.TestCase):
    """Tests for validate_extant_edges."""

    def test_valid_edges(self):
        nodes = {
            "a": MockNode(inputs=["inp"], outputs=["out"]),
            "b": MockNode(inputs=["inp"], outputs=["out"]),
        }
        edges = {
            edge_models.TargetHandle(node="b", port="inp"): edge_models.SourceHandle(
                node="a", port="out"
            ),
        }
        subgraph_protocols.validate_extant_edges(edges, nodes)

    def test_invalid_target_node(self):
        nodes = {"a": MockNode(inputs=["inp"], outputs=["out"])}
        edges = {
            edge_models.TargetHandle(
                node="nonexistent", port="inp"
            ): edge_models.SourceHandle(node="a", port="out"),
        }
        with self.assertRaises(ValueError) as ctx:
            subgraph_protocols.validate_extant_edges(edges, nodes)
        self.assertIn("target", str(ctx.exception).lower())

    def test_invalid_source_node(self):
        nodes = {"a": MockNode(inputs=["inp"], outputs=["out"])}
        edges = {
            edge_models.TargetHandle(node="a", port="inp"): edge_models.SourceHandle(
                node="nonexistent", port="out"
            ),
        }
        with self.assertRaises(ValueError) as ctx:
            subgraph_protocols.validate_extant_edges(edges, nodes)
        self.assertIn("source", str(ctx.exception).lower())

    def test_invalid_target_port(self):
        nodes = {
            "a": MockNode(inputs=["inp"], outputs=["out"]),
            "b": MockNode(inputs=["inp"], outputs=["out"]),
        }
        edges = {
            edge_models.TargetHandle(
                node="b", port="wrong_port"
            ): edge_models.SourceHandle(node="a", port="out"),
        }
        with self.assertRaises(ValueError) as ctx:
            subgraph_protocols.validate_extant_edges(edges, nodes)
        self.assertIn("target", str(ctx.exception).lower())

    def test_invalid_source_port(self):
        nodes = {
            "a": MockNode(inputs=["inp"], outputs=["out"]),
            "b": MockNode(inputs=["inp"], outputs=["out"]),
        }
        edges = {
            edge_models.TargetHandle(node="b", port="inp"): edge_models.SourceHandle(
                node="a", port="wrong_port"
            ),
        }
        with self.assertRaises(ValueError) as ctx:
            subgraph_protocols.validate_extant_edges(edges, nodes)
        self.assertIn("source", str(ctx.exception).lower())

    def test_empty_edges(self):
        nodes = {"a": MockNode(inputs=["inp"], outputs=["out"])}
        subgraph_protocols.validate_extant_edges({}, nodes)


class TestValidateAcyclicEdges(unittest.TestCase):
    """Tests for validate_acyclic_edges."""

    def test_acyclic_linear(self):
        edges = {
            edge_models.TargetHandle(node="b", port="inp"): edge_models.SourceHandle(
                node="a", port="out"
            ),
            edge_models.TargetHandle(node="c", port="inp"): edge_models.SourceHandle(
                node="b", port="out"
            ),
        }
        subgraph_protocols.validate_acyclic_edges(edges)

    def test_acyclic_diamond(self):
        edges = {
            edge_models.TargetHandle(node="b", port="inp"): edge_models.SourceHandle(
                node="a", port="out"
            ),
            edge_models.TargetHandle(node="c", port="inp"): edge_models.SourceHandle(
                node="a", port="out"
            ),
            edge_models.TargetHandle(node="d", port="i1"): edge_models.SourceHandle(
                node="b", port="out"
            ),
            edge_models.TargetHandle(node="d", port="i2"): edge_models.SourceHandle(
                node="c", port="out"
            ),
        }
        subgraph_protocols.validate_acyclic_edges(edges)

    def test_simple_cycle_rejected(self):
        edges = {
            edge_models.TargetHandle(node="b", port="inp"): edge_models.SourceHandle(
                node="a", port="out"
            ),
            edge_models.TargetHandle(
                node="a", port="feedback"
            ): edge_models.SourceHandle(node="b", port="out"),
        }
        with self.assertRaises(ValueError) as ctx:
            subgraph_protocols.validate_acyclic_edges(edges)
        self.assertIn("cycle", str(ctx.exception).lower())

    def test_self_loop_rejected(self):
        edges = {
            edge_models.TargetHandle(
                node="a", port="feedback"
            ): edge_models.SourceHandle(node="a", port="out"),
        }
        with self.assertRaises(ValueError) as ctx:
            subgraph_protocols.validate_acyclic_edges(edges)
        self.assertIn("cycle", str(ctx.exception).lower())

    def test_custom_message(self):
        edges = {
            edge_models.TargetHandle(
                node="a", port="feedback"
            ): edge_models.SourceHandle(node="a", port="out"),
        }
        with self.assertRaises(ValueError) as ctx:
            subgraph_protocols.validate_acyclic_edges(edges, message="Custom error")
        self.assertIn("Custom error", str(ctx.exception))

    def test_empty_edges(self):
        subgraph_protocols.validate_acyclic_edges({})


class TestRuntimeCheckableProtocols(unittest.TestCase):
    """Tests for runtime_checkable protocol isinstance checks."""

    def test_has_static_subgraph_protocol(self):
        """HasStaticSubgraph is runtime_checkable."""

        class ValidImpl:
            inputs = ["x"]
            outputs = ["y"]
            nodes = {}
            input_edges = {}
            edges = {}
            output_edges = {}

        self.assertIsInstance(ValidImpl(), subgraph_protocols.HasStaticSubgraph)

    def test_builds_subgraph_with_static_output_protocol(self):
        """BuildsSubgraphWithStaticOutput is runtime_checkable."""

        class ValidImpl:
            inputs = ["x"]
            outputs = ["y"]
            input_edges = {}
            output_edges = {}

            @property
            def prospective_nodes(self):
                return {}

        self.assertIsInstance(
            ValidImpl(), subgraph_protocols.BuildsSubgraphWithStaticOutput
        )

    def test_builds_subgraph_with_dynamic_output_protocol(self):
        """BuildsSubgraphWithDynamicOutput is runtime_checkable."""

        class ValidImpl:
            inputs = ["x"]
            outputs = ["y"]
            input_edges = {}
            prospective_output_edges = {}

            @property
            def prospective_nodes(self):
                return {}

        self.assertIsInstance(
            ValidImpl(), subgraph_protocols.BuildsSubgraphWithDynamicOutput
        )


if __name__ == "__main__":
    unittest.main()
