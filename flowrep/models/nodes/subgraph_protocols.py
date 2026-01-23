import abc
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flowrep.models import base_models, edge_models

ProspectiveOutputEdges = dict[edge_models.OutputTarget, list[edge_models.SourceHandle]]


class NodeProtocol(Protocol):
    inputs: base_models.Labels
    outputs: base_models.Labels


NodesAlias = dict[base_models.Label, NodeProtocol]


class HasSubgraphInput(NodeProtocol, Protocol):
    input_edges: edge_models.InputEdges


class HasStaticSubgraphOutput(HasSubgraphInput, Protocol):
    output_edges: edge_models.OutputEdges


@runtime_checkable
class HasStaticSubgraph(HasStaticSubgraphOutput, Protocol):
    nodes: NodesAlias


class BuildsSubgraph(HasSubgraphInput, Protocol):
    @property
    @abc.abstractmethod
    def prospective_nodes(self) -> NodesAlias: ...


@runtime_checkable
class BuildsSubgraphWithStaticOutput(
    HasStaticSubgraphOutput, BuildsSubgraph, Protocol
): ...


@runtime_checkable
class BuildsSubgraphWithDynamicOutput(BuildsSubgraph, Protocol):
    prospective_output_edges: ProspectiveOutputEdges


def validate_input_sources(macro: HasSubgraphInput) -> None:
    invalid_sources = {
        source.port
        for source in macro.input_edges.values()
        if source.port not in macro.inputs
    }
    if invalid_sources:
        raise ValueError(f"Invalid input_edges sources: {invalid_sources}")


def validate_output_targets(macro: HasStaticSubgraphOutput):
    invalid_targets = {
        target.port for target in macro.output_edges if target.port not in macro.outputs
    }
    if invalid_targets:
        raise ValueError(f"Invalid output_edges targets: {invalid_targets}")
