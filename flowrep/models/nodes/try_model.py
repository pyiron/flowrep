from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pydantic

from flowrep.models import base_models, edge_models
from flowrep.models.nodes import helper_models

if TYPE_CHECKING:
    from flowrep.models.nodes.union import NodeType  # Satisfies mypy

    # Still not enough to satisfy ruff, which doesn't understand the string forward
    # reference, even with the TYPE_CHECKING import
    # Better to nonetheless leave the references as strings to make sure the pydantic
    # handling of forward references is maximally robust through the model_rebuild()
    # Ultimately, just silence ruff as needed


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
        output_edges_matrix: For each output, sources from possible try/except nodes.
            Note that at most one of these possible edges will be actualized at runtime
            based on which try/except case node actually runs without exception (if
            any).

    Note:
        While each available output must be represented in the `output_edges_matrix`,
        not every possible try/except branch needs to be represented in the rows of
        this matrix. It is thus possible that some outputs are left with no source and
        thus non-data values at the end of the node's execution.
    """

    type: Literal[base_models.RecipeElementType.TRY] = pydantic.Field(
        default=base_models.RecipeElementType.TRY, frozen=True
    )
    try_node: helper_models.LabeledNode
    exception_cases: list[helper_models.ExceptionCase]
    input_edges: dict[edge_models.TargetHandle, edge_models.InputSource]
    output_edges_matrix: dict[edge_models.OutputTarget, list[edge_models.SourceHandle]]

    @property
    def prospective_nodes(self) -> dict[str, "NodeType"]:  # noqa: F821, UP037
        nodes = {self.try_node.label: self.try_node.node}
        for case in self.exception_cases:
            nodes[case.body.label] = case.body.node
        return nodes

    @pydantic.field_validator("exception_cases")
    @classmethod
    def validate_exception_cases_not_empty(cls, v):
        if len(v) < 1:
            raise ValueError("TryNode must have at least one exception case")
        return v

    @pydantic.model_validator(mode="after")
    def validate_unique_labels(self):
        labels = [self.try_node.label] + [c.body.label for c in self.exception_cases]
        if not base_models._has_unique_elements(labels):
            raise ValueError(f"All node labels must be unique. Got: {labels}")
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
                f"input_edges targets must reference try_node or exception_cases. Got "
                f"invalid target nodes: {invalid}"
            )
        return self

    @pydantic.model_validator(mode="after")
    def validate_input_edges_ports_exist(self):
        for target, source in self.input_edges.items():
            node = self.prospective_nodes[target.node]
            if target.port not in node.inputs:
                raise ValueError(
                    f"Invalid input_edge target: {target.node} has no input port "
                    f"'{target.port}'. Available inputs: {node.inputs}"
                )
            if source.port not in self.inputs:
                raise ValueError(
                    f"Invalid input_edge source: '{source.port}' is not a TryNode "
                    f"input. Available inputs: {self.inputs}"
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
                    f"{expected_nodes}, got invalid: {invalid_nodes}"
                )
            if not base_models._has_unique_elements(source_nodes):
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
