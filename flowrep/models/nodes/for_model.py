from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pydantic

from flowrep.models import base_models, edge_models, subgraph_validation
from flowrep.models.nodes import helper_models

if TYPE_CHECKING:
    from flowrep.models.nodes.union import Nodes


class ForNode(base_models.NodeModel):
    """
    Loop over a body node and collect outputs as a list.
    This is a dynamic node, which must actualize the body of its subgraph at runtime.

    Loops can be done with a combination of nested iteration and zipping values.
    Output edges whose source is an `InputSource` indicate data forwarded directly from
    the for-node's own inputs. In the even that these are inputs that are scattered to
    body nodes fro the iteration, it is the responsibility of the WfMS to collect these
    into lists alongside the body node outputs. This allows outputs to be linked
    directly to the input that generated them.

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
    6. For output edges sourced from InputSource (rather than body SourceHandle),
        collect the corresponding input values used for each iteration and connect
        to output accordingly, or pass the broadcast inputs trivially through.

    Attributes:
        type: The node type -- always "for".
        inputs: The available input port names.
        outputs: The available output port names.
        body_node: The labeled node to execute for each iteration.
        input_edges: Edges from workflow inputs to inputs of body node instances.
        output_edges: Edges from body node outputs or for-node inputs to workflow
            outputs. Sources that are InputSource values indicate forwarded input data
            (collected per-iteration); SourceHandle values indicate body node outputs.
        nested_ports: The body node ports over which to do nested iteration. Input
            edges will map parent input elements to each child node accordingly.
        zipped_ports: The body node ports over which to do zipped iteration. Input
            edges will map parent input elements to each child node accordingly.

    Note:
        All iterated output — whether collected from body executions or forwarded from
        scattered inputs — should have the same length. Thus, forwarded inputs empower
        the node output to precisely provide which input was used to produce each
        output element.
    """

    type: Literal[base_models.RecipeElementType.FOR] = pydantic.Field(
        default=base_models.RecipeElementType.FOR, frozen=True
    )
    body_node: helper_models.LabeledNode
    input_edges: edge_models.InputEdges
    output_edges: edge_models.OutputEdges
    nested_ports: base_models.Labels = pydantic.Field(default_factory=list)
    zipped_ports: base_models.Labels = pydantic.Field(default_factory=list)

    @property
    def prospective_nodes(self) -> Nodes:
        return {self.body_node.label: self.body_node.node}

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
        return self

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
        if invalid := {
            port
            for port in self.nested_ports + self.zipped_ports
            if port not in self.body_node.node.inputs
        }:
            raise ValueError(
                f"For loop must loop on body node ports ({self.body_node.node.inputs}) "
                f"but got: {invalid}"
            )
        return self
