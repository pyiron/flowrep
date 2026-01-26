from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pydantic

from flowrep.models import base_models, edge_models, subgraph_protocols
from flowrep.models.nodes import helper_models

if TYPE_CHECKING:
    from flowrep.models.nodes.union import Nodes


class TryNode(base_models.NodeModel):
    """
    Try and except your way through a series of exceptions, with the option to perform
    a finally step.
    This is a dynamic node, which must actualize the body of its subgraph at runtime.

    Intended recipe realization:
    1. Instantiate the try node
    2. Connect input to this node according to input edges
    3. Execute and evaluate the try node
    4. If an exception is encountered, walk through the exception cases
    5. If an exception match is found encountered, repeat (1-3) for the case body
    7. Use the matrix of output edges to connect the greatest possible extent of
        output to the outputs of the successful try/except case (if any)

    Attributes:
        type: The node type -- always "try".
        inputs: The available input port names.
        outputs: The available output port names.
        try_node: The primary body node to execute.
        exception_cases: The exception type-body pairs to be walked over searching for
            an exception match in the event that the try node fails.
        input_edges: Edges from workflow inputs to inputs of the prospective nodes.
        prospective_output_edges: For each output, sources from possible try/except nodes.
            Note that at most one of these possible edges will be actualized at runtime
            based on which try/except case node actually runs without exception (if
            any).

    Note:
        While each available output must be represented in the `prospective_output_edges`,
        not every possible try/except branch needs to be represented in the rows of
        this matrix. It is thus possible that some outputs are left with no source and
        thus non-data values at the end of the node's execution.
    """

    type: Literal[base_models.RecipeElementType.TRY] = pydantic.Field(
        default=base_models.RecipeElementType.TRY, frozen=True
    )
    try_node: helper_models.LabeledNode
    exception_cases: list[helper_models.ExceptionCase]
    input_edges: edge_models.InputEdges
    prospective_output_edges: dict[
        edge_models.OutputTarget, base_models.UniqueList[edge_models.SourceHandle]
    ]

    @property
    def prospective_nodes(self) -> Nodes:
        nodes = {self.try_node.label: self.try_node.node}
        for case in self.exception_cases:
            nodes[case.body.label] = case.body.node
        return nodes

    @pydantic.model_validator(mode="after")
    def validate_prospective_nodes_have_unique_labels(self):
        labels = [self.try_node.label] + [c.body.label for c in self.exception_cases]
        base_models.validate_unique(labels)
        return self

    @pydantic.field_validator("exception_cases")
    @classmethod
    def validate_exception_cases_not_empty(cls, v):
        if len(v) < 1:
            raise ValueError("TryNode must have at least one exception case")
        return v

    @pydantic.model_validator(mode="after")
    def validate_io_edges(self):
        subgraph_protocols.validate_input_sources(self)
        subgraph_protocols.validate_prospective_input_targets(self)
        subgraph_protocols.validate_prospective_output_sources(self)
        subgraph_protocols.validate_prospective_output_targets(self)
        return self
