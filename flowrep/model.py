import keyword
from enum import StrEnum
from typing import Annotated, Any, ClassVar, Literal

import networkx as nx
import pydantic
import pydantic_core


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


class HandleModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)
    node: str | None
    port: str
    delimiter: ClassVar[str] = "."

    @pydantic.model_serializer
    def serialize(self) -> str:
        if self.node is None:
            return self.port
        return self.delimiter.join([self.node, self.port])

    @pydantic.model_validator(mode="before")
    @classmethod
    def deserialize(cls, data):
        if isinstance(data, str):
            parts = data.split(".", 1)
            if len(parts) == 1:
                return {"node": None, "port": parts[0]}
            return {"node": parts[0], "port": parts[1]}
        return data


class SourceHandle(HandleModel):
    node: str


class TargetHandle(HandleModel):
    node: str


class InputSource(HandleModel):
    node: None = pydantic.Field(default=None, frozen=True)


class OutputTarget(HandleModel):
    node: None = pydantic.Field(default=None, frozen=True)


class WorkflowNode(NodeModel):
    type: Literal[RecipeElementType.WORKFLOW] = pydantic.Field(
        default=RecipeElementType.WORKFLOW, frozen=True
    )
    nodes: dict[str, "NodeType"]
    input_edges: dict[TargetHandle, InputSource]
    edges: dict[TargetHandle, SourceHandle]
    output_edges: dict[OutputTarget, SourceHandle]

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
    node: NodeModel


def _has_unique_elements(values: list[Any]) -> bool:
    return len(values) == len(set(values))


class ForNode(NodeModel):
    """
    Loop over a body node and collect outputs as a list.

    Loops can be done with a combination of nested iteration and zipping values.
    The `transfer_edges` field allows you to optionally indicate which looping values
    should be returned listed alongside the lists of body node outputs, so that outputs
    can be linked directly to the input that generated them.
    """

    type: Literal[RecipeElementType.FOR] = pydantic.Field(
        default=RecipeElementType.FOR, frozen=True
    )
    body_node: LabeledNode
    input_edges: dict[TargetHandle, InputSource]
    output_edges: dict[OutputTarget, SourceHandle]
    nested_ports: list[str] = pydantic.Field(default_factory=list)
    zipped_ports: list[str] = pydantic.Field(default_factory=list)
    transfer_edges: dict[OutputTarget, InputSource] = pydantic.Field(
        default_factory=dict
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


class WhileNode(NodeModel):
    type: Literal[RecipeElementType.WHILE] = pydantic.Field(
        default=RecipeElementType.WHILE, frozen=True
    )


class IfNode(NodeModel):
    type: Literal[RecipeElementType.IF] = pydantic.Field(
        default=RecipeElementType.IF, frozen=True
    )


class TryNode(NodeModel):
    type: Literal[RecipeElementType.TRY] = pydantic.Field(
        default=RecipeElementType.TRY, frozen=True
    )


# Discriminated Union
NodeType = Annotated[
    AtomicNode | WorkflowNode | ForNode | WhileNode | IfNode | TryNode,
    pydantic.Field(discriminator="type"),
]

WorkflowNode.model_rebuild()
