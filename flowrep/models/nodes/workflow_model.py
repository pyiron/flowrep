from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pydantic

from flowrep.models import base_models, edge_models, subgraph_validation

if TYPE_CHECKING:
    from flowrep.models.nodes.union import Nodes


class WorkflowNode(base_models.NodeModel):
    """
    Hold and execute a subgraph of nodes.
    This is a completely static graph; everything is known about it at the class level,
    and its retrospective version looks identical to its prospective version (modulo
    actually having all the output data).

    Intended recipe realization:
    - WfMS are expected to make the IO of nodes available retrospectively, regardless of
        how deeply nested in subgraphs they are.

    Attributes:
        type: The node type -- always "workflow".
        inputs: The available input port names.
        outputs: The available output port names.
        nodes: The nodes of the subgraph.
        input_edges: Edges from workflow inputs to inputs of subgraph nodes.
        edges: Edges between subgraph nodes.
        output_edges: Edges from subgraph nodes back to workflow outputs.
    """

    type: Literal[base_models.RecipeElementType.WORKFLOW] = pydantic.Field(
        default=base_models.RecipeElementType.WORKFLOW, frozen=True
    )
    nodes: "Nodes"  # noqa: UP037
    input_edges: edge_models.InputEdges
    edges: edge_models.Edges
    output_edges: edge_models.OutputEdges

    @pydantic.model_validator(mode="after")
    def validate_io_edges(self):
        subgraph_validation.validate_input_edge_sources(self.input_edges, self.inputs)
        subgraph_validation.validate_input_edge_targets(self.input_edges, self.nodes)
        subgraph_validation.validate_output_edge_sources(
            self.output_edges.values(), self.nodes
        )
        subgraph_validation.validate_output_edge_targets(
            self.output_edges, self.outputs
        )
        return self

    @pydantic.model_validator(mode="after")
    def validate_subgraph(self):
        subgraph_validation.validate_sibling_edges(self.edges, self.nodes)
        subgraph_validation.validate_acyclic_edges(
            self.edges,
            message="Workflow models must be acyclic (DAG), but found cycle(s)",
        )
        return self
