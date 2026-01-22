from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pydantic

from flowrep.models import edges_model
from flowrep.models.nodes import helper_models, model

if TYPE_CHECKING:
    from flowrep.models.nodes.union import NodeType  # Satisfies mypy


class IfNode(model.NodeModel):
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
        output_edges_matrix: For each output, sources from each possible body node to
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

    type: Literal[model.RecipeElementType.IF] = pydantic.Field(
        default=model.RecipeElementType.IF, frozen=True
    )
    cases: list[helper_models.ConditionalCase]
    input_edges: dict[edges_model.TargetHandle, edges_model.InputSource]
    output_edges_matrix: dict[edges_model.OutputTarget, list[edges_model.SourceHandle]]
    else_case: helper_models.LabeledNode | None = None

    @property
    def prospective_nodes(self) -> dict[str, "NodeType"]:  # noqa: F821, UP037
        nodes = {}
        for case in self.cases:
            nodes[case.condition.label] = case.condition.node
            nodes[case.body.label] = case.body.node

        if self.else_case:
            nodes[self.else_case.label] = self.else_case.node
        return nodes

    @pydantic.field_validator("cases")
    @classmethod
    def validate_cases_not_empty(cls, v):
        if len(v) < 1:
            raise ValueError("If nodes must have at least one explicit case")
        return v

    @pydantic.model_validator(mode="after")
    def validate_unique_labels(self):
        labels = (
            [case.condition.label for case in self.cases]
            + [case.body.label for case in self.cases]
            + ([self.else_case.label] if self.else_case else [])
        )
        if not model._has_unique_elements(labels):
            raise ValueError(
                f"All prospective node labels must be unique. Got: {labels}"
            )
        return self

    @pydantic.model_validator(mode="after")
    def validate_input_edges_targets_are_extant_child_nodes(self):
        invalid = {
            target.node
            for target in self.input_edges
            if target.node not in self.prospective_nodes
        }
        if invalid:
            raise ValueError(
                f"input_edges targets must be a body node based on for-node naming "
                f"schemes -- i.e. map data from parent input to child nodes. "
                f"Got invalid target nodes: {invalid}"
            )
        return self

    @pydantic.model_validator(mode="after")
    def validate_input_edges_ports_exist(self):
        """Validate that input_edges target ports exist on their target nodes."""
        for target in self.input_edges:
            node = self.prospective_nodes[target.node]
            if target.port not in node.inputs:
                raise ValueError(
                    f"Invalid input_edge target: {target.node} has no input port "
                    f"'{target.port}'. Available inputs: {node.inputs}"
                )
        return self

    @pydantic.model_validator(mode="after")
    def validate_output_edges_matrix_keys_match_outputs(self):
        edge_ports = {target.port for target in self.output_edges_matrix}
        output_ports = set(self.outputs)
        if edge_ports != output_ports:
            missing = output_ports - edge_ports
            extra = edge_ports - output_ports
            raise ValueError(
                f"output_edges_matrix keys must match outputs. "
                f"Missing: {missing or 'none'}, Extra: {extra or 'none'}"
            )
        return self

    @pydantic.model_validator(mode="after")
    def validate_output_edges_matrix_sources(self):
        expected_nodes = list(self.prospective_nodes)
        for target, sources in self.output_edges_matrix.items():
            source_nodes = [s.node for s in sources]
            invalid_nodes = set(source_nodes) - set(expected_nodes)
            if len(source_nodes) == 0:
                raise ValueError(
                    f"output_edges_matrix['{target.port}'] must have at least one source"
                )
            if invalid_nodes:
                raise ValueError(
                    f"output_edges_matrix['{target.port}'] sources must be from "
                    f"{expected_nodes}, got invalid sources: {invalid_nodes}"
                )
            if not model._has_unique_elements(source_nodes):
                raise ValueError(
                    f"output_edges_matrix['{target.port}'] must have at most one "
                    f"source from each other node. Got duplicates in: {source_nodes}"
                )
            for source in sources:
                node = self.prospective_nodes[source.node]
                if source.port not in node.outputs:
                    raise ValueError(
                        f"Invalid output_edges_matrix source: {source.node} has no "
                        f"output port '{source.port}'. Available outputs: {node.outputs}"
                    )
        return self
