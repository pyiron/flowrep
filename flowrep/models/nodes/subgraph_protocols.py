import abc
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flowrep.models import base_models, edge_models

if TYPE_CHECKING:
    from flowrep.models.nodes.union import NodeType  # Satisfies mpypy

    # Still not enough to satisfy ruff, which doesn't understand the string forward
    # reference, even with the TYPE_CHECKING import
    # Better to nonetheless leave the references as strings to make sure the pydantic
    # handling of forward references is maximally robust through the model_rebuild()
    # Ultimately, just silence ruff as needed


Nodes = dict[base_models.Label, "NodeType"]
ProspectiveOutputEdges = dict[edge_models.OutputTarget, list[edge_models.SourceHandle]]


class NodeProtocol(Protocol):
    inputs: base_models.Labels
    outputs: base_models.Labels


class HasSubgraphInput(NodeProtocol, Protocol):
    input_edges: edge_models.InputEdges


class HasStaticSubgraphOutput(HasSubgraphInput, Protocol):
    output_edges: edge_models.OutputEdges


@runtime_checkable
class HasStaticSubgraph(HasStaticSubgraphOutput, Protocol):
    nodes: Nodes


class BuildsSubgraph(HasSubgraphInput, Protocol):
    @property
    @abc.abstractmethod
    def prospective_nodes(self) -> Nodes: ...


@runtime_checkable
class BuildsSubgraphWithDynamicOutput(BuildsSubgraph, Protocol):
    prospective_output_edges: ProspectiveOutputEdges


@runtime_checkable
class BuildsSubgraphWithStaticOutput(
    HasStaticSubgraphOutput, BuildsSubgraph, Protocol
): ...


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
