from __future__ import annotations

from typing import Literal

import pydantic

from flowrep.models import base_models, edge_models
from flowrep.models.nodes import helper_models


class WhileNode(base_models.NodeModel):
    """
    A loop node that repeatedly executes a body while a condition is true.
    This is a dynamic node, which must actualize the body of its subgraph at runtime.

    Intended recipe realization:
        1. The case condition recipe is used to instantiate a condition node
        2. Input edges are routed to the condition
        3. The condition is executed and evaluated
            a) The evaluation port is specified in the case
        4. If the condition evaluates `False`, terminate
            b) Node outputs will remain in a state of not having data
        5. Else, the case body recipe is used to instantiate a body node
        6. Input edges are routed to the body
        7. The body is executed
        8. Another condition recipe is evaluated
        9. Input and/xor body-condition edges are routed to it, prioritizing b-c edges
        10.The new condition is executed and evaluated
        11. If the condition evaluates `False`, terminate and use output edges to route
            data from the most recent body node to the output
        12. Else, repeat steps 7-11
            a) In step 6, use body-body edges to connect data output from the last body
                node instance, taking precedence over input edges

    Attributes:
        type: The node type -- always "while".
        inputs: The available input port names.
        outputs: The available output port names.
        case: The condition-body pair to be looped over by repeated instantiation.
            The condition node must produce a boolean output (specified by
            condition_output or inferred if the condition has exactly one output).
        input_edges: Edges from workflow inputs to the initial condition/body nodes.
            Keys are targets on condition/body, values are workflow input ports.
        output_edges: Edges from final condition/body outputs to workflow outputs.
            Keys are workflow output ports, values are sources on condition/body.
        body_body_edges: Edges carrying data between body iterations. Maps body
            outputs to body inputs for the next iteration.
        body_condition_edges: Edges from body outputs to condition inputs for
            re-evaluation after each iteration.
    """

    type: Literal[base_models.RecipeElementType.WHILE] = pydantic.Field(
        default=base_models.RecipeElementType.WHILE, frozen=True
    )
    case: helper_models.ConditionalCase
    input_edges: edge_models.InputEdges
    output_edges: edge_models.OutputEdges
    body_body_edges: dict[edge_models.TargetHandle, edge_models.SourceHandle]
    body_condition_edges: dict[edge_models.TargetHandle, edge_models.SourceHandle]

    @pydantic.model_validator(mode="after")
    def validate_input_edges(self):
        """
        Validate input_edges: sources from workflow inputs, targets to case nodes.
        """
        valid_targets = {
            self.case.condition.label: self.case.condition.node.inputs,
            self.case.body.label: self.case.body.node.inputs,
        }
        workflow_inputs = self.inputs

        for target, source in self.input_edges.items():
            if source.port not in workflow_inputs:
                raise ValueError(
                    f"Invalid input_edge source: '{source.port}' is not a workflow "
                    f"input. Available inputs: {self.inputs}"
                )
            if target.node not in valid_targets:
                raise ValueError(
                    f"Invalid input_edge target: node '{target.node}' must be "
                    f"'{self.case.condition.label}' or '{self.case.body.label}'"
                )
            if target.port not in valid_targets[target.node]:
                raise ValueError(
                    f"Invalid input_edge target: node '{target.node}' has no input "
                    f"port '{target.port}'. Available inputs: "
                    f"{list(valid_targets[target.node])}"
                )
        return self

    @pydantic.model_validator(mode="after")
    def validate_output_edges(self):
        """
        Validate output_edges: sources from case nodes, targets to workflow outputs.
        """
        valid_sources = {
            self.case.condition.label: self.case.condition.node.outputs,
            self.case.body.label: self.case.body.node.outputs,
        }
        workflow_outputs = self.outputs

        for target, source in self.output_edges.items():
            if target.port not in workflow_outputs:
                raise ValueError(
                    f"Invalid output_edge target: '{target.port}' is not a workflow "
                    f"output. Available outputs: {self.outputs}"
                )
            if source.node not in valid_sources:
                raise ValueError(
                    f"Invalid output_edge source: node '{source.node}' must be "
                    f"'{self.case.condition.label}' or '{self.case.body.label}'"
                )
            if source.port not in valid_sources[source.node]:
                raise ValueError(
                    f"Invalid output_edge source: node '{source.node}' has no output "
                    f"port '{source.port}'. Available outputs: "
                    f"{list(valid_sources[source.node])}"
                )
        return self

    @pydantic.model_validator(mode="after")
    def validate_body_body_edges(self):
        """Validate body_body_edges: body outputs -> body inputs."""
        body_outputs = self.case.body.node.outputs
        body_inputs = self.case.body.node.inputs
        body_label = self.case.body.label

        for target, source in self.body_body_edges.items():
            if source.node != body_label:
                raise ValueError(
                    f"Invalid body_body_edge source: node must be '{body_label}', "
                    f"got '{source.node}'"
                )
            if source.port not in body_outputs:
                raise ValueError(
                    f"Invalid body_body_edge source: '{source.port}' is not an output "
                    f"of body node. Available outputs: {list(body_outputs)}"
                )
            if target.node != body_label:
                raise ValueError(
                    f"Invalid body_body_edge target: node must be '{body_label}', "
                    f"got '{target.node}'"
                )
            if target.port not in body_inputs:
                raise ValueError(
                    f"Invalid body_body_edge target: '{target.port}' is not an input "
                    f"of body node. Available inputs: {list(body_inputs)}"
                )
        return self

    @pydantic.model_validator(mode="after")
    def validate_body_condition_edges(self):
        """Validate body_condition_edges: body outputs -> condition inputs."""
        body_outputs = set(self.case.body.node.outputs)
        body_label = self.case.body.label
        condition_inputs = set(self.case.condition.node.inputs)
        condition_label = self.case.condition.label

        for target, source in self.body_condition_edges.items():
            if source.node != body_label:
                raise ValueError(
                    f"Invalid body_condition_edge source: node must be '{body_label}', "
                    f"got '{source.node}'"
                )
            if source.port not in body_outputs:
                raise ValueError(
                    f"Invalid body_condition_edge source: '{source.port}' is not an "
                    f"output of body node. Available outputs: {list(body_outputs)}"
                )
            if target.node != condition_label:
                raise ValueError(
                    f"Invalid body_condition_edge target: node must be "
                    f"'{condition_label}', got '{target.node}'"
                )
            if target.port not in condition_inputs:
                raise ValueError(
                    f"Invalid body_condition_edge target: '{target.port}' is not an "
                    f"input of condition node. Available inputs: "
                    f"{list(condition_inputs)}"
                )
        return self
