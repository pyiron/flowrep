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


class ForNode(NodeModel):
    type: Literal[RecipeElementType.FOR] = pydantic.Field(
        default=RecipeElementType.FOR, frozen=True
    )


class LabeledNode(pydantic.BaseModel):
    label: str
    node: "NodeType"


class ConditionalCase(pydantic.BaseModel):
    condition: LabeledNode
    body: LabeledNode
    condition_output: str | None = None

    @pydantic.field_validator("label")
    @classmethod
    def validate_label(cls, v):
        if not _valid_label(v):
            raise ValueError(
                f"Label must be a valid Python identifier and not in "
                f"reserved labels {RESERVED_NAMES}. Got '{v}'"
            )
        return v

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


def _has_unique_elements(values: list[Any]) -> bool:
    return len(values) == len(set(values))


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
    input_edges: dict[TargetHandle, InputSource]
    output_edges: dict[OutputTarget, SourceHandle]
    body_body_edges: dict[TargetHandle, SourceHandle]
    body_condition_edges: dict[TargetHandle, SourceHandle]

    @pydantic.field_validator("nested_ports", "zipped_ports")
    @classmethod
    def validate_single_appearance(cls, v, info):
        if not _has_unique_elements(v):
            raise ValueError(
                f"'{info.field_name}' must contain unique values, but got {v}."
            )
        return v

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
