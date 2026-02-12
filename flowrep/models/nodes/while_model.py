from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pydantic

from flowrep.models import base_models, edge_models, subgraph_validation
from flowrep.models.nodes import helper_models

if TYPE_CHECKING:
    from flowrep.models.nodes.union import Nodes


class WhileNode(base_models.NodeModel):
    """
    A loop node that repeatedly executes a body while a condition is true.
    This is a dynamic node, which must actualize the body of its subgraph at runtime.
    Output labels must be a subset of input labels to facilitate unambiguous looping
    and guarantee output availability even if the body never executes.
    Output edges _must_ come from the case's body node, not its condition node.
    It is the responsibility of the WfMS to repeat execution, inferring from the
    outputs (and their correspondence with input labels) which data needs to looped on
    such that body instance outputs get passed to the next set of condition and body
    instances.
    It is also the responsibility of the WfMS to leverage loop inputs as fallbacks in
    case the body never executes, in this way we are a little cheeky about the recipe
    having static outputs -- the actual sourcing is runtime dependent, but you are
    guaranteed to have data there at the end of the day.

    Intended recipe realization:
        1. The case condition recipe is used to instantiate a condition node
        2. Input edges are routed to the condition
        3. The condition is executed and evaluated
            a) The evaluation port is specified in the case
        4. If the condition evaluates `False`, terminate; route while-loop inputs to
            outputs of matching label to guarantee data availability.
        5. Else, the case body recipe is used to instantiate a body node
        6. Input edges are routed to the body, and it is executed
        7. Another condition node is instantiated, and the correspondence of loop
            output labels and input labels is leveraged to infer what inputs need to
            come from the loop input, and what from the most recent body node instance
        8. If the condition evaluates `False`, terminate and route the most recent
            body node output to the loop output
        9. Else, instantiate another body node and follow the same output-input label
            comparison to infer which input data is still coming from the loop vs.
            which is coming from the last body node instance
        10. Repeat steps 7-9 until the condition evaluates False.

    Attributes:
        type: The node type -- always "while".
        inputs: The available input port names.
        outputs: The available output port names. For while-nodes these _must_ be a
            subset of the input port names.
        case: The condition-body pair to be looped over by repeated instantiation.
            The condition node must produce a boolean output (specified by
            condition_output or inferred if the condition has exactly one output).
        input_edges: Edges from workflow inputs to the initial condition/body nodes.
            Keys are targets on condition/body, values are workflow input ports.
        output_edges: Edges from the body of the conditional case to the outputs.
            The actual runtime output edges are dependent on whether the condition
            ever evaluated to be true, so these output edges are constrained (no
            coming from the condition node) and pseudo-prospective (pass-through input
            may be leveraged at runtime in a fixed way).
    """

    type: Literal[base_models.RecipeElementType.WHILE] = pydantic.Field(
        default=base_models.RecipeElementType.WHILE, frozen=True
    )
    case: helper_models.ConditionalCase
    input_edges: edge_models.InputEdges
    output_edges: edge_models.OutputEdges

    @property
    def prospective_nodes(self) -> Nodes:
        return {
            self.case.condition.label: self.case.condition.node,
            self.case.body.label: self.case.body.node,
        }

    @property
    def body_body_edges(self) -> edge_models.Edges:
        """Inferred edges for passing body output to next body iteration."""
        return self._inferred_iteration_edges(self.case.body.label)

    @property
    def body_condition_edges(self) -> edge_models.Edges:
        """Inferred edges for passing body output to next condition evaluation."""
        return self._inferred_iteration_edges(self.case.condition.label)

    def _inferred_iteration_edges(
        self, target_label: base_models.Label
    ) -> edge_models.Edges:
        output_to_body_port = {t.port: s.port for t, s in self.output_edges.items()}
        return {
            target: edge_models.SourceHandle(
                node=self.case.body.label, port=output_to_body_port[source.port]
            )
            for target, source in self.input_edges.items()
            if target.node == target_label and source.port in output_to_body_port
        }

    @pydantic.model_validator(mode="after")
    def validate_output_is_subset_of_inputs(self):
        if not set(self.outputs).issubset(self.inputs):
            raise ValueError(
                f"While-loop outputs must be a subset of inputs, got: "
                f"{self.outputs} vs. {self.inputs}"
            )
        return self

    @pydantic.model_validator(mode="after")
    def validate_io_edges(self):
        subgraph_validation.validate_input_edge_sources(self.input_edges, self.inputs)
        subgraph_validation.validate_input_edge_targets(
            self.input_edges,
            self.prospective_nodes,
        )
        subgraph_validation.validate_output_edge_targets(
            self.output_edges, self.outputs
        )
        subgraph_validation.validate_output_edge_sources(
            self.output_edges.values(),
            self.prospective_nodes,
            self.inputs,
        )

        # Finally, do the stricter validation that the output edges only come from the
        # body node
        if invalid_nodes := {
            f"{t.serialize()}: {s.serialize()}"
            for t, s in self.output_edges.items()
            if s.node != self.case.body.label
        }:
            raise ValueError(
                f"Output edges may only be specified from the body node, but found: "
                f"{invalid_nodes}"
            )
        return self
