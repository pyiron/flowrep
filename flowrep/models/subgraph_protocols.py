import abc
import collections
import itertools
from collections.abc import Collection, Iterable
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


def validate_input_sources(macro: HasSubgraphInput) -> None:
    if invalid_sources := {
        source.port
        for source in macro.input_edges.values()
        if source.port not in macro.inputs
    }:
        raise ValueError(
            f"Invalid input_edges, source port not found in inputs: {invalid_sources}"
        )


def _validate_input_targets(
    input_edges: edge_models.InputEdges, nodes: NodesAlias
) -> None:
    if invalid_nodes := {
        target.node for target in input_edges if target.node not in nodes
    }:
        raise ValueError(
            f"Invalid input_edges targets. Could not find target nodes "
            f"{invalid_nodes} among available nodes {tuple(nodes.keys())}"
        )
    if invalid_ports := {
        (target.port, tuple(nodes[target.node].inputs))
        for target in input_edges
        if target.port not in nodes[target.node].inputs
    }:
        raise ValueError(
            f"Invalid input_edges targets. Could not find port among node inputs "
            f"(target port, available inputs): {invalid_ports}"
        )


def validate_input_targets(macro: HasStaticSubgraph) -> None:
    _validate_input_targets(macro.input_edges, macro.nodes)


def validate_prospective_input_targets(macro: BuildsSubgraph) -> None:
    _validate_input_targets(macro.input_edges, macro.prospective_nodes)


def _validate_output_targets(
    outputs: base_models.Labels,
    edge_targets: Iterable[edge_models.OutputTarget],
) -> None:
    edge_target_ports = {target.port for target in edge_targets}
    if invalid_targets := edge_target_ports - set(outputs):
        raise ValueError(
            f"Invalid edge, target port not found in outputs: {invalid_targets}"
        )
    if missing_outputs := set(outputs) - edge_target_ports:
        raise ValueError(f"Missing edge for output(s): {missing_outputs}")


def validate_output_targets(macro: HasStaticSubgraphOutput) -> None:
    _validate_output_targets(macro.outputs, macro.output_edges)


def validate_prospective_output_targets(macro: BuildsSubgraphWithDynamicOutput) -> None:
    _validate_output_targets(macro.outputs, macro.prospective_output_edges)


def _validate_output_sources(
    sources: Collection[edge_models.SourceHandle], nodes: NodesAlias
) -> None:
    if invalid_nodes := {source.node for source in sources if source.node not in nodes}:
        raise ValueError(
            f"Invalid output sources. Could not find source nodes "
            f"({invalid_nodes}) among available nodes ({nodes})"
        )
    if invalid_ports := {
        (source.port, tuple(nodes[source.node].outputs))
        for source in sources
        if source.port not in nodes[source.node].outputs
    }:
        raise ValueError(
            f"Invalid output source. Could not find port among node outputs "
            f"(source port, available outputs): {invalid_ports}"
        )


def validate_output_sources(macro: HasStaticSubgraph) -> None:
    _validate_output_sources(macro.output_edges.values(), macro.nodes)


def validate_output_sources_from_prospective_nodes(
    macro: BuildsSubgraphWithStaticOutput,
) -> None:
    _validate_output_sources(macro.output_edges.values(), macro.prospective_nodes)


def validate_prospective_output_sources(macro: BuildsSubgraphWithDynamicOutput) -> None:
    nodes = macro.prospective_nodes
    for target, sources in macro.prospective_output_edges.items():
        if not sources:
            raise ValueError(
                f"Invalid prospective_output_edges for {target}. "
                f"Sources list cannot be empty."
            )
        node_counts = collections.Counter(source.node for source in sources)
        if duplicate_nodes := {
            node for node, count in node_counts.items() if count > 1
        }:
            raise ValueError(
                f"Invalid prospective_output_edges for {target}. "
                f"Duplicate source nodes: {duplicate_nodes}"
            )
    for sources in macro.prospective_output_edges.values():
        _validate_output_sources(sources, nodes)


def validate_extant_edges(edges: edge_models.Edges, nodes: NodesAlias) -> None:
    for target, source in edges.items():
        if target.node not in nodes:
            raise ValueError(f"Invalid edge target, node not found in nodes: {target}")
        if source.node not in nodes:
            raise ValueError(f"Invalid edge source, node not found in nodes: {source}")
        if target.port not in nodes[target.node].inputs:
            raise ValueError(
                f"Invalid edge target, port not found in node inputs: {target}"
            )
        if source.port not in nodes[source.node].outputs:
            raise ValueError(
                f"Invalid edge source, port not found in node outputs: {source}"
            )


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
