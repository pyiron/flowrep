from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pydantic

from flowrep.models import base_models, edge_models, subgraph_protocols
from flowrep.models.nodes import helper_models

if TYPE_CHECKING:
    from flowrep.models.nodes.union import Nodes


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
    body_body_edges: edge_models.Edges
    body_condition_edges: edge_models.Edges

    @property
    def prospective_nodes(self) -> Nodes:
        return {
            self.case.condition.label: self.case.condition.node,
            self.case.body.label: self.case.body.node,
        }

    @pydantic.field_validator("case")
    @classmethod
    def validate_prospective_nodes_have_unique_labels(
        cls, v: helper_models.ConditionalCase
    ):
        base_models.validate_unique([v.condition.label, v.body.label])
        return v

    @pydantic.model_validator(mode="after")
    def validate_io_edges(self):
        subgraph_protocols.validate_input_sources(self)
        subgraph_protocols.validate_prospective_input_targets(self)
        subgraph_protocols.validate_output_sources_from_prospective_nodes(self)
        subgraph_protocols.validate_output_targets(self)
        return self

    @pydantic.model_validator(mode="after")
    def validate_body_body_edges(self):
        """Validate body_body_edges: body outputs -> body inputs."""
        subgraph_protocols.validate_extant_edges(
            self.body_body_edges,
            {self.case.body.label: self.case.body.node},
        )
        return self

    @pydantic.model_validator(mode="after")
    def validate_body_condition_edges(self):
        """Validate body_condition_edges: body outputs -> condition inputs."""
        subgraph_protocols.validate_extant_edges(
            self.body_condition_edges,
            self.prospective_nodes,
        )
        for source in self.body_condition_edges.values():
            if source.node != self.case.body.label:
                raise ValueError(
                    f"body_condition_edges must have the body node as the source, but "
                    f"{source.model_dump(mode='json')} does not"
                )
        for target in self.body_condition_edges:
            if target.node != self.case.condition.label:
                raise ValueError(
                    f"body_condition_edges must have the condition node as the target, "
                    f"but {target.model_dump(mode='json')} does not"
                )
        return self
