"""Unit tests for flowrep.models.subgraph_validation"""

import unittest

from flowrep.models import edge_models, subgraph_validation


class MockNode:
    """Minimal node for testing."""

    def __init__(self, inputs: list[str], outputs: list[str]):
        self.inputs = inputs
        self.outputs = outputs


class TestValidateInputEdgeSources(unittest.TestCase):
    """Tests for validate_input_edge_sources."""

    def test_valid_sources(self):
        input_edges = {
            edge_models.TargetHandle(node="a", port="inp"): edge_models.InputSource(
                port="x"
            ),
        }
        subgraph_validation.validate_input_edge_sources(input_edges, ["x", "y"])

    def test_invalid_source_port(self):
        input_edges = {
            edge_models.TargetHandle(node="a", port="inp"): edge_models.InputSource(
                port="nonexistent"
            ),
        }
        with self.assertRaises(ValueError) as ctx:
            subgraph_validation.validate_input_edge_sources(input_edges, ["x"])
        self.assertIn("nonexistent", str(ctx.exception))

    def test_empty_edges(self):
        subgraph_validation.validate_input_edge_sources({}, ["x"])


class TestValidateInputEdgeTargets(unittest.TestCase):
    """Tests for validate_input_edge_targets."""

    def test_valid_targets(self):
        nodes = {"a": MockNode(inputs=["inp"], outputs=["out"])}
        input_edges = {
            edge_models.TargetHandle(node="a", port="inp"): edge_models.InputSource(
                port="x"
            ),
        }
        subgraph_validation.validate_input_edge_targets(input_edges, nodes)

    def test_invalid_target_node(self):
        nodes = {"a": MockNode(inputs=["inp"], outputs=["out"])}
        input_edges = {
            edge_models.TargetHandle(
                node="nonexistent", port="inp"
            ): edge_models.InputSource(port="x"),
        }
        with self.assertRaises(ValueError) as ctx:
            subgraph_validation.validate_input_edge_targets(input_edges, nodes)
        self.assertIn("nonexistent", str(ctx.exception))

    def test_invalid_target_port(self):
        nodes = {"a": MockNode(inputs=["inp"], outputs=["out"])}
        input_edges = {
            edge_models.TargetHandle(
                node="a", port="wrong_port"
            ): edge_models.InputSource(port="x"),
        }
        with self.assertRaises(ValueError) as ctx:
            subgraph_validation.validate_input_edge_targets(input_edges, nodes)
        self.assertIn("wrong_port", str(ctx.exception))

    def test_empty_edges(self):
        nodes = {"a": MockNode(inputs=["inp"], outputs=["out"])}
        subgraph_validation.validate_input_edge_targets({}, nodes)


class TestValidateOutputEdgeTargets(unittest.TestCase):
    """Tests for validate_output_edge_targets."""

    def test_valid_output_targets(self):
        output_targets = [edge_models.OutputTarget(port="y")]
        subgraph_validation.validate_output_edge_targets(output_targets, ["y"])

    def test_valid_multiple_outputs(self):
        output_targets = [
            edge_models.OutputTarget(port="y"),
            edge_models.OutputTarget(port="z"),
        ]
        subgraph_validation.validate_output_edge_targets(output_targets, ["y", "z"])

    def test_invalid_output_target(self):
        output_targets = [edge_models.OutputTarget(port="nonexistent")]
        with self.assertRaises(ValueError) as ctx:
            subgraph_validation.validate_output_edge_targets(output_targets, ["y"])
        self.assertIn("nonexistent", str(ctx.exception))

    def test_missing_output_edge(self):
        output_targets = [edge_models.OutputTarget(port="y")]
        with self.assertRaises(ValueError) as ctx:
            subgraph_validation.validate_output_edge_targets(output_targets, ["y", "z"])
        self.assertIn("Missing", str(ctx.exception))
        self.assertIn("z", str(ctx.exception))

    def test_empty_outputs_and_edges(self):
        subgraph_validation.validate_output_edge_targets([], [])


class TestValidateOutputEdgeSources(unittest.TestCase):
    """Tests for validate_output_edge_sources."""

    def test_valid_output_sources(self):
        nodes = {"a": MockNode(inputs=["inp"], outputs=["out"])}
        sources = [edge_models.SourceHandle(node="a", port="out")]
        subgraph_validation.validate_output_edge_sources(sources, nodes)

    def test_invalid_source_node(self):
        nodes = {"a": MockNode(inputs=["inp"], outputs=["out"])}
        sources = [edge_models.SourceHandle(node="nonexistent", port="out")]
        with self.assertRaises(ValueError) as ctx:
            subgraph_validation.validate_output_edge_sources(sources, nodes)
        self.assertIn("nonexistent", str(ctx.exception))

    def test_invalid_source_port(self):
        nodes = {"a": MockNode(inputs=["inp"], outputs=["out"])}
        sources = [edge_models.SourceHandle(node="a", port="wrong_port")]
        with self.assertRaises(ValueError) as ctx:
            subgraph_validation.validate_output_edge_sources(sources, nodes)
        self.assertIn("wrong_port", str(ctx.exception))

    def test_multiple_sources(self):
        nodes = {
            "a": MockNode(inputs=[], outputs=["out1", "out2"]),
            "b": MockNode(inputs=[], outputs=["out"]),
        }
        sources = [
            edge_models.SourceHandle(node="a", port="out1"),
            edge_models.SourceHandle(node="b", port="out"),
        ]
        subgraph_validation.validate_output_edge_sources(sources, nodes)

    def test_empty_sources(self):
        nodes = {"a": MockNode(inputs=[], outputs=["out"])}
        subgraph_validation.validate_output_edge_sources([], nodes)


class TestValidateProspectiveSources(unittest.TestCase):
    """Tests for validate_prospective_sources."""

    def test_valid_sources(self):
        target = edge_models.OutputTarget(port="y")
        sources = [
            edge_models.SourceHandle(node="a", port="out1"),
            edge_models.SourceHandle(node="b", port="out"),
        ]
        subgraph_validation.validate_prospective_sources_list(target, sources)

    def test_empty_sources_rejected(self):
        target = edge_models.OutputTarget(port="y")
        with self.assertRaises(ValueError) as ctx:
            subgraph_validation.validate_prospective_sources_list(target, [])
        self.assertIn("empty", str(ctx.exception).lower())

    def test_duplicate_source_nodes_rejected(self):
        target = edge_models.OutputTarget(port="y")
        sources = [
            edge_models.SourceHandle(node="a", port="out1"),
            edge_models.SourceHandle(node="a", port="out2"),
        ]
        with self.assertRaises(ValueError) as ctx:
            subgraph_validation.validate_prospective_sources_list(target, sources)
        self.assertIn("Duplicate", str(ctx.exception))

    def test_single_source_valid(self):
        target = edge_models.OutputTarget(port="y")
        sources = [edge_models.SourceHandle(node="a", port="out")]
        subgraph_validation.validate_prospective_sources_list(target, sources)


class TestValidateSiblingEdges(unittest.TestCase):
    """Tests for validate_sibling_edges."""

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
        subgraph_validation.validate_sibling_edges(edges, nodes)

    def test_invalid_target_node(self):
        nodes = {"a": MockNode(inputs=["inp"], outputs=["out"])}
        edges = {
            edge_models.TargetHandle(
                node="nonexistent", port="inp"
            ): edge_models.SourceHandle(node="a", port="out"),
        }
        with self.assertRaises(ValueError) as ctx:
            subgraph_validation.validate_sibling_edges(edges, nodes)
        self.assertIn("target", str(ctx.exception).lower())

    def test_invalid_source_node(self):
        nodes = {"a": MockNode(inputs=["inp"], outputs=["out"])}
        edges = {
            edge_models.TargetHandle(node="a", port="inp"): edge_models.SourceHandle(
                node="nonexistent", port="out"
            ),
        }
        with self.assertRaises(ValueError) as ctx:
            subgraph_validation.validate_sibling_edges(edges, nodes)
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
            subgraph_validation.validate_sibling_edges(edges, nodes)
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
            subgraph_validation.validate_sibling_edges(edges, nodes)
        self.assertIn("source", str(ctx.exception).lower())

    def test_empty_edges(self):
        nodes = {"a": MockNode(inputs=["inp"], outputs=["out"])}
        subgraph_validation.validate_sibling_edges({}, nodes)

    def test_separate_source_and_target_nodes(self):
        """Test with different source_nodes and target_nodes dicts."""
        target_nodes = {"b": MockNode(inputs=["inp"], outputs=["out"])}
        source_nodes = {"a": MockNode(inputs=["inp"], outputs=["out"])}
        edges = {
            edge_models.TargetHandle(node="b", port="inp"): edge_models.SourceHandle(
                node="a", port="out"
            ),
        }
        subgraph_validation.validate_sibling_edges(edges, target_nodes, source_nodes)

    def test_separate_nodes_invalid_source(self):
        """Source node must be in source_nodes, not target_nodes."""
        target_nodes = {
            "a": MockNode(inputs=["inp"], outputs=["out"]),
            "b": MockNode(inputs=["inp"], outputs=["out"]),
        }
        source_nodes = {"b": MockNode(inputs=["inp"], outputs=["out"])}
        edges = {
            edge_models.TargetHandle(node="b", port="inp"): edge_models.SourceHandle(
                node="a", port="out"
            ),
        }
        with self.assertRaises(ValueError) as ctx:
            subgraph_validation.validate_sibling_edges(
                edges, target_nodes, source_nodes
            )
        self.assertIn("source", str(ctx.exception).lower())


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
        subgraph_validation.validate_acyclic_edges(edges)

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
        subgraph_validation.validate_acyclic_edges(edges)

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
            subgraph_validation.validate_acyclic_edges(edges)
        self.assertIn("cycle", str(ctx.exception).lower())

    def test_self_loop_rejected(self):
        edges = {
            edge_models.TargetHandle(
                node="a", port="feedback"
            ): edge_models.SourceHandle(node="a", port="out"),
        }
        with self.assertRaises(ValueError) as ctx:
            subgraph_validation.validate_acyclic_edges(edges)
        self.assertIn("cycle", str(ctx.exception).lower())

    def test_custom_message(self):
        edges = {
            edge_models.TargetHandle(
                node="a", port="feedback"
            ): edge_models.SourceHandle(node="a", port="out"),
        }
        with self.assertRaises(ValueError) as ctx:
            subgraph_validation.validate_acyclic_edges(edges, message="Custom error")
        self.assertIn("Custom error", str(ctx.exception))

    def test_empty_edges(self):
        subgraph_validation.validate_acyclic_edges({})


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

        self.assertIsInstance(ValidImpl(), subgraph_validation.StaticSubgraphOwner)

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
            ValidImpl(), subgraph_validation.DynamicSubgraphStaticOutput
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
            ValidImpl(), subgraph_validation.DynamicSubgraphDynamicOutput
        )


if __name__ == "__main__":
    unittest.main()
