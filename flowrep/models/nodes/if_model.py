from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pydantic

from flowrep.models import base_models, edge_models, subgraph_validation
from flowrep.models.nodes import helper_models

if TYPE_CHECKING:
    from flowrep.models.nodes.union import Nodes


class IfNode(base_models.NodeModel):
    """
    Walk through one or more cases, executing and returning the body result for the
    first case with a positive condition evaluation.
    This is a dynamic node, which must actualize the body of its subgraph at runtime.

    Intended recipe realization:
    1. Instantiate the first case's condition node
    2. Connect input to this node according to input edges
    3. Execute and evaluate the condition node
    4. If it evaluates negatively, repeat steps (1-3) as long as new cases are available
    5. If it evaluates positively (or finally for the else case if it is provided),
        instantiate, connect, and execute the body node as for the condition node(s)
    6. Use the matrix of output edges to connect the output of the actualized case
        body/else case to the node outputs

    Attributes:
        type: The node type -- always "if".
        inputs: The available input port names.
        outputs: The available output port names.
        cases: The condition-body pairs to be walked over searching for a positive
            condition evaluation.
        input_edges: Edges from workflow inputs to inputs of body node instances.
        prospective_output_edges: For each output, sources from each possible body node to
            fill that output. Note that exactly one of these possible edges will be
            actualized at runtime based on which body/else case node actually runs.
        else_case: Optional body node to execute if no positive case condition can be
            found.

    Note:
        In this way, the if-node is guaranteed to have a concrete set of outputs which
        are fulfilled, regardless of which case runs internally. In the event that none
        of the conditional cases evaluate and no else case is provided, these outputs
        will be left in a state of non-data.
    """

    type: Literal[base_models.RecipeElementType.IF] = pydantic.Field(
        default=base_models.RecipeElementType.IF, frozen=True
    )
    cases: list[helper_models.ConditionalCase]
    input_edges: edge_models.InputEdges
    prospective_output_edges: dict[
        edge_models.OutputTarget, base_models.UniqueList[edge_models.SourceHandle]
    ]
    else_case: helper_models.LabeledNode | None = None

    @property
    def prospective_nodes(self) -> Nodes:
        nodes = {}
        for case in self.cases:
            nodes[case.condition.label] = case.condition.node
            nodes[case.body.label] = case.body.node

        if self.else_case:
            nodes[self.else_case.label] = self.else_case.node
        return nodes

    @pydantic.model_validator(mode="after")
    def validate_prospective_nodes_have_unique_labels(self):
        base_models.validate_unique(
            [
                label
                for case in self.cases
                for label in [case.condition.label, case.body.label]
            ]
            + [self.else_case.label]
            if self.else_case
            else []
        )
        return self

    @pydantic.model_validator(mode="after")
    def validate_io_edges(self):
        subgraph_validation.validate_input_edge_sources(self.input_edges, self.inputs)
        subgraph_validation.validate_input_edge_targets(
            self.input_edges,
            self.prospective_nodes,
        )
        for target, prospective_sources in self.prospective_output_edges.items():
            subgraph_validation.validate_prospective_sources_list(
                target, prospective_sources
            )
            subgraph_validation.validate_output_edge_sources(
                prospective_sources,
                self.prospective_nodes,
            )
        subgraph_validation.validate_output_edge_targets(
            self.prospective_output_edges, self.outputs
        )
        return self

    @pydantic.field_validator("cases")
    @classmethod
    def validate_cases_not_empty(cls, v):
        if len(v) < 1:
            raise ValueError("If nodes must have at least one explicit case")
        return v
