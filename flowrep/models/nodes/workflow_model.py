from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import networkx as nx
import pydantic

from flowrep.models import edges_model
from flowrep.models.nodes import base_models

if TYPE_CHECKING:
    from flowrep.models.nodes.union import NodeType  # Satisfies mypy

    # Still not enough to satisfy ruff, which doesn't understand the string forward
    # reference, even with the TYPE_CHECKING import
    # Better to nonetheless leave the references as strings to make sure the pydantic
    # handling of forward references is maximally robust through the model_rebuild()
    # Ultimately, just silence ruff as needed


class WorkflowNode(base_models.NodeModel):
    type: Literal[base_models.RecipeElementType.WORKFLOW] = pydantic.Field(
        default=base_models.RecipeElementType.WORKFLOW, frozen=True
    )
    nodes: dict[str, "NodeType"]  # noqa: F821, UP037
    input_edges: dict[edges_model.TargetHandle, edges_model.InputSource]
    edges: dict[edges_model.TargetHandle, edges_model.SourceHandle]
    output_edges: dict[edges_model.OutputTarget, edges_model.SourceHandle]

    @pydantic.field_validator("nodes")
    @classmethod
    def validate_node_labels(cls, v, info):
        base_models._validate_labels(set(v.keys()), info)
        return v

    @pydantic.model_validator(mode="after")
    def validate_edge_references(self):
        """Validate that edges reference existing nodes and valid ports."""
        node_labels = set(self.nodes.keys())
        workflow_inputs = set(self.inputs)
        workflow_outputs = set(self.outputs)

        for target, source in (
            self.input_edges.items() | self.edges.items() | self.output_edges.items()
        ):
            # Validate source
            if source.node is None:
                if source.port not in workflow_inputs:
                    raise ValueError(
                        f"Invalid edge source: '{source.port}' is not a workflow "
                        f"input. Available inputs: {self.inputs}"
                    )
            else:
                if source.node not in node_labels:
                    raise ValueError(
                        f"Invalid edge source: node '{source.node}' is not a child node"
                    )
                if source.port not in self.nodes[source.node].outputs:
                    raise ValueError(
                        f"Invalid edge source: node '{source.node}' has no output port "
                        f"'{source.port}'. "
                        f"Available outputs: {self.nodes[source.node].outputs}"
                    )

            # Validate target
            if target.node is None:
                if target.port not in workflow_outputs:
                    raise ValueError(
                        f"Invalid edge target: '{target.port}' is not a workflow "
                        f"output. Available outputs: {self.outputs}"
                    )
            else:
                if target.node not in node_labels:
                    raise ValueError(
                        f"Invalid edge target: node '{target.node}' is not a child node"
                    )
                if target.port not in self.nodes[target.node].inputs:
                    raise ValueError(
                        f"Invalid edge target: node '{target.node}' has no input port "
                        f"'{target.port}'. "
                        f"Available inputs: {self.nodes[target.node].inputs}"
                    )

        return self

    @pydantic.model_validator(mode="after")
    def validate_acyclic(self):
        """Ensure the workflow graph is acyclic (DAG)."""
        g = nx.DiGraph()
        g.add_nodes_from(self.nodes.keys())

        for target, source in self.edges.items():
            if target.node is not None and source.node is not None:
                g.add_edge(source.node, target.node)

        try:
            cycles = list(nx.find_cycle(g, orientation="original"))
            raise ValueError(
                f"Workflow graph contains cycle(s): {cycles}. "
                f"Workflows must be acyclic (DAG)."
            )
        except nx.NetworkXNoCycle:
            pass

        return self
