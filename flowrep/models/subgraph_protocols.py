import abc
import collections
import itertools
from collections.abc import Collection
from typing import Protocol, runtime_checkable

import networkx as nx

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


def validate_input_edge_sources(
    input_edges: edge_models.InputEdges,
    available_inputs: base_models.Labels,
) -> None:
    if invalid := {
        s.serialize() for s in input_edges.values() if s.port not in available_inputs
    }:
        raise ValueError(f"Invalid input_edges source ports: {invalid}")


def validate_input_edge_targets(
    input_edges: edge_models.InputEdges,
    target_nodes: NodesAlias,
) -> None:
    if invalid_nodes := {
        t.serialize() for t in input_edges if t.node not in target_nodes
    }:
        raise ValueError(
            f"Invalid input_edges target nodes {invalid_nodes}, "
            f"available: {tuple(target_nodes.keys())}"
        )
    if invalid_ports := {
        t.serialize() for t in input_edges if t.port not in target_nodes[t.node].inputs
    }:
        raise ValueError(f"Invalid input_edges target ports: {invalid_ports}")


def validate_output_edge_targets(
    output_targets: Collection[edge_models.OutputTarget],
    available_outputs: base_models.Labels,
) -> None:
    target_ports = {t.port for t in output_targets}
    if invalid := target_ports - set(available_outputs):
        raise ValueError(f"Invalid output target ports: {invalid}")
    if missing := set(available_outputs) - target_ports:
        raise ValueError(f"Missing output edge for: {missing}")


def validate_output_edge_sources(
    sources: Collection[edge_models.SourceHandle],
    source_nodes: NodesAlias,
) -> None:
    if invalid_nodes := {s.serialize() for s in sources if s.node not in source_nodes}:
        raise ValueError(f"Invalid output source nodes: {invalid_nodes}")
    if invalid_ports := {
        s.serialize() for s in sources if s.port not in source_nodes[s.node].outputs
    }:
        raise ValueError(f"Invalid output source ports: {invalid_ports}")


def validate_prospective_sources_list(
    target: edge_models.OutputTarget,
    sources: Collection[edge_models.SourceHandle],
) -> None:
    if not sources:
        raise ValueError(f"Sources for '{target.serialize()}' cannot be empty.")
    node_counts = collections.Counter(source.node for source in sources)
    if duplicate_nodes := {node for node, count in node_counts.items() if count > 1}:
        raise ValueError(
            f"Sources for {target.serialize()} must be unique. "
            f"Duplicate source nodes: {duplicate_nodes}"
        )


def validate_sibling_edges(
    edges: edge_models.Edges,
    target_nodes: NodesAlias,
    source_nodes: NodesAlias | None = None,
) -> None:
    if source_nodes is None:
        source_nodes = target_nodes
    for target, source in edges.items():
        if target.node not in target_nodes:
            raise ValueError(f"Invalid edge target node: {target.serialize()}")
        if source.node not in source_nodes:
            raise ValueError(f"Invalid edge source node: {source.serialize()}")
        if target.port not in target_nodes[target.node].inputs:
            raise ValueError(f"Invalid edge target port: {target.serialize()}")
        if source.port not in source_nodes[source.node].outputs:
            raise ValueError(f"Invalid edge source port: {source.serialize()}")


def validate_acyclic_edges(
    edges: edge_models.Edges, message="Edges contain cycle(s)"
) -> None:
    g = nx.DiGraph()
    g.add_nodes_from({h.node for h in itertools.chain(edges, edges.values())})

    for target, source in edges.items():
        if target.node is not None and source.node is not None:
            g.add_edge(source.node, target.node)

    try:
        cycles = list(nx.find_cycle(g, orientation="original"))
        raise ValueError(f"{message}: {cycles}. ")
    except nx.NetworkXNoCycle:
        pass
