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


class WhileNode(NodeModel):
    type: Literal[RecipeElementType.WHILE] = pydantic.Field(
        default=RecipeElementType.WHILE, frozen=True
    )


class LabeledNode(pydantic.BaseModel):
    label: str
    node: NodeModel


def _has_unique_elements(values: list[Any]) -> bool:
    return len(values) == len(set(values))


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


class IfNode(NodeModel):
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

    type: Literal[RecipeElementType.IF] = pydantic.Field(
        default=RecipeElementType.IF, frozen=True
    )
    cases: list[ConditionalCase]
    input_edges: dict[TargetHandle, InputSource]
    output_edges_matrix: dict[OutputTarget, list[SourceHandle]]
    else_case: LabeledNode | None = None

    @property
    def prospective_nodes(self) -> dict[str, NodeModel]:
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
        if not _has_unique_elements(labels):
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
            if not _has_unique_elements(source_nodes):
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
