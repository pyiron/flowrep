from __future__ import annotations

from typing import Literal

import pydantic

from flowrep.models import edges_model
from flowrep.models.nodes import helper_models, model


class ForNode(model.NodeModel):
    """
    Loop over a body node and collect outputs as a list.
    This is a dynamic node, which must actualize the body of its subgraph at runtime.

    Loops can be done with a combination of nested iteration and zipping values.
    The `transfer_edges` field allows you to optionally indicate which looping values
    should be returned listed alongside the lists of body node outputs, so that outputs
    can be linked directly to the input that generated them.

    Intended recipe realization:
    1. Assess the number of body executions necessary by examining the lengths of
        nested and zipped ports, and the length of data on the corresponding inputs.
        a) The data in all inputs being passed to zipped ports should be
            length-validated
        b) The count scales multiplicatively with the data in each input passed to
            nested ports, and finally multiplied once more by the zipped length (or
            directly the zipped length if no nested ports are present).
    2. Create the appropriate number of body node instances in the subgraph
    3. Broadcast input edges not involved in nested or zipped ports to each child
    4. Decompose input for input edges used for zipped or nested ports and scatter
        edges to each child accordingly
        a) The manner of this decomposition is an implementation detail for which the
            WfMS is responsible
    5. Collect output of child nodes into list fields and connect these to the output
        according to the output edges.
        a) The manner of this collection is an implementation detail for which the
            WfMS is responsible
    6. Collect the looped inputs used for each child as indicated in the transfer edges
        and connect these to output according to the transfer edges
        a) The manner of this collection is an implementation detail for which the
            WfMS is responsible

    Attributes:
        type: The node type -- always "for".
        inputs: The available input port names.
        outputs: The available output port names.
        input_edges: Edges from workflow inputs to inputs of body node instances.
        output_edges: Edges from final condition/body outputs to workflow outputs.
            Keys are workflow output ports, values are sources on condition/body.
        nested_ports: The body node ports over which to do nested iteration. Input
            edges will map parent input elements to each child node accordingly.
        zipped_ports: The body node ports over which to do zipped iteration. Input
            edges will map parent input elements to each child node accordingly.
        transfer_edges: Any inputs that are used to forward data to the nested or
            zipped ports which you wish to have as outputs. This is to (optionally)
            provide a direct map between each output element and the input used to
            produce it (output _not_ looped on is static and presumed to be available
            from another source; it does not need to be synchronized with the output).

    Note:
        In this way, all output -- whether it is collected output of child executions
        of the specified body node, or whether it is forwarded input data specified by
        transfer edges, should have the same length. Thus, in the event that transfer
        edges are specified, they should empower the node output to precisely provide
        which input was used to produce each output element.
    """

    type: Literal[model.RecipeElementType.FOR] = pydantic.Field(
        default=model.RecipeElementType.FOR, frozen=True
    )
    body_node: helper_models.LabeledNode
    input_edges: dict[edges_model.TargetHandle, edges_model.InputSource]
    output_edges: dict[edges_model.OutputTarget, edges_model.SourceHandle]
    nested_ports: list[str] = pydantic.Field(default_factory=list)
    zipped_ports: list[str] = pydantic.Field(default_factory=list)
    transfer_edges: dict[edges_model.OutputTarget, edges_model.InputSource] = (
        pydantic.Field(default_factory=dict)
    )

    @pydantic.field_validator("nested_ports", "zipped_ports")
    @classmethod
    def validate_single_appearance(cls, v, info):
        if not model._has_unique_elements(v):
            raise ValueError(
                f"'{info.field_name}' must contain unique values, but got {v}."
            )
        return v

    @pydantic.model_validator(mode="after")
    def validate_some_loop(self):
        if not (self.nested_ports or self.zipped_ports):
            raise ValueError("For loop must have at least one nested or zipped port")
        return self

    @pydantic.model_validator(mode="after")
    def validate_non_overlapping_loops(self):
        if not set(self.nested_ports).isdisjoint(self.zipped_ports):
            raise ValueError(
                f"Loop values in nested_ports or zipped_ports must not overlap, but "
                f"share {set(self.nested_ports).intersection(self.zipped_ports)}."
            )
        return self

    @pydantic.model_validator(mode="after")
    def validate_loop_ports_exist(self):
        invalid = [
            port
            for port in self.nested_ports + self.zipped_ports
            if port not in self.body_node.node.inputs
        ]
        if invalid:
            raise ValueError(
                f"For loop must loop on body node ports ({self.body_node.node.inputs}) but got: {invalid}"
            )
        return self

    @pydantic.model_validator(mode="after")
    def validate_edge_nodes(self):
        """Validate that all edge handles reference the correct nodes."""
        # Input edges must target the body node
        invalid_input_targets = [
            str(target)
            for target in self.input_edges
            if (
                target.node != self.body_node.label
                or target.port not in self.body_node.node.inputs
            )
        ]
        if invalid_input_targets:
            raise ValueError(
                f"All input_edges must target body_node '{self.body_node.label}'. "
                f"Found invalid targets: {invalid_input_targets}"
            )

        # Output edges must source from the body node
        invalid_output_sources = [
            str(source)
            for source in self.output_edges.values()
            if (
                source.node != self.body_node.label
                or source.port not in self.body_node.node.outputs
            )
        ]
        if invalid_output_sources:
            raise ValueError(
                f"All output_edges must source from body_node '{self.body_node.label}'. "
                f"Found invalid sources: {invalid_output_sources}"
            )

        return self

    @pydantic.model_validator(mode="after")
    def validate_input_sources_are_inputs(self):
        invalid_input_ports = [
            str(source)
            for source in self.input_edges.values()
            if source.port not in self.inputs
        ]
        if invalid_input_ports:
            raise ValueError(
                f"All input_edges sources must reference valid inputs. "
                f"Found invalid: {invalid_input_ports}"
            )
        return self

    @pydantic.model_validator(mode="after")
    def validate_output_targets_are_outputs(self):
        invalid_output_ports = [
            str(target)
            for target in self.output_edges
            if target.port not in self.outputs
        ]
        if invalid_output_ports:
            raise ValueError(
                f"All output_edges targets must reference valid outputs. "
                f"Found invalid: {invalid_output_ports}"
            )
        return self

    @pydantic.model_validator(mode="after")
    def validate_transfer_sources_are_inputs(self):
        invalid = [
            source.port
            for source in self.transfer_edges.values()
            if source.port not in self.inputs
        ]
        if invalid:
            raise ValueError(
                f"Transfer edge sources must be ForNode inputs. "
                f"Unknown inputs: {invalid}. Available: {self.inputs}"
            )
        return self

    @pydantic.model_validator(mode="after")
    def validate_transfer_targets_are_outputs(self):
        invalid = [
            target.port
            for target in self.transfer_edges
            if target.port not in self.outputs
        ]
        if invalid:
            raise ValueError(
                f"Transfer edge targets must be ForNode outputs. "
                f"Unknown outputs: {invalid}. Available: {self.outputs}"
            )
        return self

    @pydantic.model_validator(mode="after")
    def validate_transfer_sources_are_looped(self):
        """
        Looped ports are scoped on the body node itself, so here we need to follow the
        data flow to check what for-node inputs feed through to these.
        """
        looped_ports = set(self.nested_ports) | set(self.zipped_ports)

        input_to_body_port = {
            source.port: target.port for target, source in self.input_edges.items()
        }

        non_looped = [
            source.port
            for source in self.transfer_edges.values()
            if input_to_body_port.get(source.port) not in looped_ports
        ]
        if non_looped:
            raise ValueError(
                f"Transfer edges can only forward inputs that feed looped body ports. "
                f"Non-looped inputs: {non_looped}. Looped body ports: {sorted(looped_ports)}"
            )
        return self

    @pydantic.model_validator(mode="after")
    def validate_transfer_targets_not_in_output_edges(self):
        output_edge_ports = {target.port for target in self.output_edges}
        collision = [
            target.port
            for target in self.transfer_edges
            if target.port in output_edge_ports
        ]
        if collision:
            raise ValueError(
                f"Transfer edges cannot target outputs already used by output_edges. "
                f"Conflicting outputs: {collision}"
            )
        return self
