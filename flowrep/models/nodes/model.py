from __future__ import annotations

import keyword
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

import networkx as nx
import pydantic
import pydantic_core

from flowrep.models import edges as edges_model

if TYPE_CHECKING:
    from flowrep.models.nodes.union import NodeType  # Satisfies mypy

    # Still not enough to satisfy ruff, which doesn't understand the string forward
    # reference, even with the TYPE_CHECKING import
    # Better to nonetheless leave the references as strings to make sure the pydantic
    # handling of forward references is maximally robust through the model_rebuild()
    # Ultimately, just silence ruff as needed


class RecipeElementType(StrEnum):
    ATOMIC = "atomic"
    WORKFLOW = "workflow"
    FOR = "for"
    WHILE = "while"
    IF = "if"
    TRY = "try"


class IOTypes(StrEnum):
    INPUTS = "inputs"
    OUTPUTS = "outputs"


RESERVED_NAMES = {"inputs", "outputs"}  # No having child nodes with these names


def _has_unique_elements(values: list[Any]) -> bool:
    return len(values) == len(set(values))


def _valid_label(label: str) -> bool:
    return (
        label.isidentifier()
        and not keyword.iskeyword(label)
        and label not in RESERVED_NAMES
    )


def _get_invalid_labels(labels: list[str] | set[str]) -> set[str]:
    return {label for label in labels if not _valid_label(label)}


def _validate_labels(labels: list[str] | set[str], info) -> None:
    invalid = _get_invalid_labels(labels)
    if invalid:
        raise ValueError(
            f"All elements of '{info.field_name}' must be a valid Python "
            f"identifier and not in the reserved labels {RESERVED_NAMES}. "
            f"{invalid} are non-compliant."
        )


class UnpackMode(StrEnum):
    """How to handle return values from atomic nodes.

    - NONE: Return the output as a single value
    - TUPLE: Split return into one port per tuple element
    - DATACLASS: Split return into one port per dataclass field
    """

    NONE = "none"
    TUPLE = "tuple"
    DATACLASS = "dataclass"


class NodeModel(pydantic.BaseModel):
    type: RecipeElementType
    inputs: list[str]
    outputs: list[str]

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        super().__pydantic_init_subclass__(**kwargs)
        if cls.__name__ != NodeModel.__name__:  # I.e. for subclasses
            type_field = cls.model_fields["type"]
            if type_field.default is pydantic_core.PydanticUndefined:
                raise TypeError(
                    f"{cls.__name__} must provide a default value for 'type'"
                )
            if not type_field.frozen:
                raise TypeError(f"{cls.__name__} must mark 'type' as frozen")

    @pydantic.field_validator("inputs", "outputs")
    @classmethod
    def validate_io_labels(cls, v, info):
        if len(v) != len(set(v)):
            duplicates = [x for x in v if v.count(x) > 1]
            raise ValueError(
                f"'{info.field_name}' must contain unique values. "
                f"Found duplicates: {set(duplicates)}"
            )

        _validate_labels(v, info)
        return v


class AtomicNode(NodeModel):
    type: Literal[RecipeElementType.ATOMIC] = pydantic.Field(
        default=RecipeElementType.ATOMIC, frozen=True
    )
    fully_qualified_name: str
    unpack_mode: UnpackMode = UnpackMode.TUPLE

    @pydantic.field_validator("fully_qualified_name")
    @classmethod
    def check_name_format(cls, v: str):
        if not v or len(v.split(".")) < 2 or not all(part for part in v.split(".")):
            msg = (
                f"AtomicNode 'fully_qualified_name' must be a non-empty string "
                f"in the format 'module.qualname' with at least one period. Got {v}"
            )
            raise ValueError(msg)
        return v

    @pydantic.model_validator(mode="after")
    def check_outputs_when_not_unpacking(self):
        if self.unpack_mode == UnpackMode.NONE and len(self.outputs) > 1:
            raise ValueError(
                f"Outputs must have exactly one element when unpacking is disabled. "
                f"Got {len(self.outputs)} outputs with "
                f"unpack_mode={self.unpack_mode.value}"
            )
        return self


class WorkflowNode(NodeModel):
    type: Literal[RecipeElementType.WORKFLOW] = pydantic.Field(
        default=RecipeElementType.WORKFLOW, frozen=True
    )
    nodes: dict[str, "NodeType"]  # noqa: F821, UP037
    input_edges: dict[edges_model.TargetHandle, edges_model.InputSource]
    edges: dict[edges_model.TargetHandle, edges_model.SourceHandle]
    output_edges: dict[edges_model.OutputTarget, edges_model.SourceHandle]

    @pydantic.field_validator("nodes")
    @classmethod
    def validate_node_labels(cls, v, info):
        _validate_labels(set(v.keys()), info)
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


class LabeledNode(pydantic.BaseModel):
    label: str
    node: "NodeType"  # noqa: F821, UP037

    @pydantic.field_validator("label")
    @classmethod
    def validate_label(cls, v):
        if not _valid_label(v):
            raise ValueError(
                f"Label must be a valid Python identifier and not in "
                f"reserved labels {RESERVED_NAMES}. Got '{v}'"
            )
        return v


class ForNode(NodeModel):
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

    type: Literal[RecipeElementType.FOR] = pydantic.Field(
        default=RecipeElementType.FOR, frozen=True
    )
    body_node: LabeledNode
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
        if not _has_unique_elements(v):
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


class ConditionalCase(pydantic.BaseModel):
    condition: LabeledNode
    body: LabeledNode
    condition_output: str | None = None

    @pydantic.model_validator(mode="after")
    def validate_condition_is_accessible(self):
        if self.condition_output is None:
            if len(self.condition.node.outputs) != 1:
                raise ValueError(
                    f"condition must have exactly one output if condition_output is not "
                    f"provided. Got condition outputs: {self.condition.node.outputs}"
                )
        elif self.condition_output not in self.condition.node.outputs:
            raise ValueError(
                f"condition_output '{self.condition_output}' is not found among "
                f"available outputs: {self.condition.node.outputs}"
            )
        return self

    @pydantic.model_validator(mode="after")
    def validate_distinct_labels(self):
        if self.condition.label == self.body.label:
            raise ValueError(
                f"Condition and body must have distinct labels, "
                f"both are '{self.condition.label}'"
            )
        return self


class WhileNode(NodeModel):
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

    type: Literal[RecipeElementType.WHILE] = pydantic.Field(
        default=RecipeElementType.WHILE, frozen=True
    )
    case: ConditionalCase
    input_edges: dict[edges_model.TargetHandle, edges_model.InputSource]
    output_edges: dict[edges_model.OutputTarget, edges_model.SourceHandle]
    body_body_edges: dict[edges_model.TargetHandle, edges_model.SourceHandle]
    body_condition_edges: dict[edges_model.TargetHandle, edges_model.SourceHandle]

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


class ExceptionCase(pydantic.BaseModel):
    """
    An exception/node pair.

    Attributes:
        exceptions: The fully qualified names (i.e. module+qualname) of the exception
            types.
        body: The node to couple to these exceptions.
    """

    exceptions: list[str]
    body: LabeledNode

    @pydantic.field_validator("exceptions")
    @classmethod
    def validate_exceptions_not_empty(cls, v):
        if len(v) < 1:
            raise ValueError("ExceptionCase must catch at least one exception type")
        return v
