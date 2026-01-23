import abc
from typing import TYPE_CHECKING, Protocol

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


class HasSubgraphInput(NodeProtocol):
    input_edges: edge_models.InputEdges


class HasStaticSubgraphOutput(HasSubgraphInput):
    output_edges: edge_models.OutputEdges


class HasStaticSubgraph(HasStaticSubgraphOutput):
    nodes: Nodes


class BuildsSubgraph(HasSubgraphInput):
    @property
    @abc.abstractmethod
    def prospective_nodes(self) -> Nodes: ...


class BuildsSubgraphWithDynamicOutput(BuildsSubgraph):
    prospective_output_edges: ProspectiveOutputEdges


class BuildsSubgraphWithStaticOutput(HasStaticSubgraphOutput, BuildsSubgraph): ...
