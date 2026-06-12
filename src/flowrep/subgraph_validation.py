import collections
import heapq
from collections.abc import Callable, Collection, Iterable
from typing import Any, Protocol, runtime_checkable

from flowrep import base_models, edge_models

ProspectiveOutputEdges = dict[edge_models.OutputTarget, list[edge_models.SourceHandle]]


class RecipeProtocol(Protocol):
    inputs: base_models.Labels
    outputs: base_models.Labels

    @property
    def inputs_with_defaults(self) -> base_models.Labels: ...

    def validate_internal_data_completeness(self): ...


NodesAlias = dict[base_models.Label, RecipeProtocol]


@runtime_checkable
class StaticSubgraphOwner(Protocol):
    """Owns a concrete subgraph known at definition time (WorkflowRecipe)."""

    inputs: base_models.Labels
    outputs: base_models.Labels
    nodes: NodesAlias
    input_edges: edge_models.InputEdges
    edges: edge_models.Edges
    output_edges: edge_models.OutputEdges


class DynamicSubgraphOwner(Protocol):
    """
    Owns a subgraph instantiated at runtime (ForEachRecipe, WhileRecipe, IfRecipe,
    TryRecipe).
    """

    inputs: base_models.Labels
    outputs: base_models.Labels
    input_edges: edge_models.InputEdges

    @property
    def prospective_nodes(self) -> NodesAlias: ...


@runtime_checkable
class DynamicSubgraphStaticOutput(DynamicSubgraphOwner, Protocol):
    """
    Dynamic subgraph with output interface known a-priori (ForEachRecipe, WhileRecipe).
    """

    output_edges: edge_models.OutputEdges


@runtime_checkable
class DynamicSubgraphDynamicOutput(DynamicSubgraphOwner, Protocol):
    """Dynamic subgraph with output interface known at runtime (IfRecipe, TryRecipe)."""

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
    sources: Collection[edge_models.SourceHandle | edge_models.InputSource],
    source_nodes: NodesAlias,
    inputs: base_models.Labels,
) -> None:
    if invalid_nodes := {
        s.serialize()
        for s in sources
        if s.node is not None and s.node not in source_nodes
    }:
        raise ValueError(f"Invalid output source nodes: {invalid_nodes}")
    if invalid_ports := {
        s.serialize()
        for s in sources
        if (
            (s.node is None and s.port not in inputs)
            or (s.node is not None and s.port not in source_nodes[s.node].outputs)
        )
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
    # InputSource/OutputTarget handles carry node=None; they cannot form cycles.
    pairs = [
        (source.node, target.node)
        for target, source in edges.items()
        if target.node is not None and source.node is not None
    ]
    nodes = {node for pair in pairs for node in pair}
    topological_sort(nodes, pairs, cycle_message=message)


def topological_sort(
    nodes: Iterable[base_models.Label],
    dependencies: Iterable[tuple[base_models.Label, base_models.Label]],
    *,
    tie_breaker: Callable[[base_models.Label], Any] | None = None,
    cycle_message: str = "Graph contains cycle(s)",
) -> list[base_models.Label]:
    """Return ``nodes`` in dependency order via Kahn's algorithm.

    Args:
        nodes: All node labels to order.
        dependencies: ``(before, after)`` pairs; ``before`` must precede
            ``after``. Every label referenced must be present in ``nodes``.
        tie_breaker: Key applied to ready nodes to break ties deterministically.
            Defaults to insertion order within ``nodes``.
        cycle_message: Message for the ``ValueError`` raised on a cycle.

    Returns:
        The node labels in a valid topological order.

    Raises:
        ValueError: If the graph contains a cycle.
    """
    node_list = list(nodes)
    key: Callable[[base_models.Label], Any]
    if tie_breaker is None:
        position = {label: index for index, label in enumerate(node_list)}
        key = position.__getitem__
    else:
        key = tie_breaker

    in_degree: dict[base_models.Label, int] = {label: 0 for label in node_list}
    successors: dict[base_models.Label, list[base_models.Label]] = {
        label: [] for label in node_list
    }
    for before, after in dependencies:
        in_degree[after] += 1
        successors[before].append(after)

    ready = [(key(label), label) for label in node_list if in_degree[label] == 0]
    heapq.heapify(ready)
    order: list[base_models.Label] = []
    while ready:
        _, label = heapq.heappop(ready)
        order.append(label)
        for successor in successors[label]:
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                heapq.heappush(ready, (key(successor), successor))

    if len(order) != len(node_list):
        raise ValueError(cycle_message)
    return order


def validate_nodes_are_fully_sourced(
    nodes: NodesAlias,
    context: Collection[edge_models.TargetHandle],
):
    for label, node in nodes.items():
        for port in node.inputs:
            target = edge_models.TargetHandle(node=label, port=port)
            if port not in node.inputs_with_defaults and target not in context:
                raise ValueError(
                    f"Could not find a source or default for the target: {label}.{port}"
                )
    for node in nodes.values():
        node.validate_internal_data_completeness()
