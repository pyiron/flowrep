from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pydantic

from flowrep.models import base_models, edge_models, subgraph_protocols

if TYPE_CHECKING:
    from flowrep.models.nodes.union import Nodes


class WorkflowNode(base_models.NodeModel):
    type: Literal[base_models.RecipeElementType.WORKFLOW] = pydantic.Field(
        default=base_models.RecipeElementType.WORKFLOW, frozen=True
    )
    nodes: "Nodes"  # noqa: UP037
    input_edges: edge_models.InputEdges
    edges: edge_models.Edges
    output_edges: edge_models.OutputEdges

    @pydantic.model_validator(mode="after")
    def validate_io_edges(self):
        subgraph_protocols.validate_input_edge_sources(self.input_edges, self.inputs)
        subgraph_protocols.validate_input_edge_targets(self.input_edges, self.nodes)
        subgraph_protocols.validate_output_edge_sources(
            self.output_edges.values(), self.nodes
        )
        subgraph_protocols.validate_output_edge_targets(self.output_edges, self.outputs)
        return self

    @pydantic.model_validator(mode="after")
    def validate_subgraph(self):
        subgraph_protocols.validate_sibling_edges(self.edges, self.nodes)
        subgraph_protocols.validate_acyclic_edges(
            self.edges,
            message="Workflow models must be acyclic (DAG), but found cycle(s)",
        )
        return self
