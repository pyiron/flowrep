from enum import Enum
from typing import Annotated, ClassVar, Literal

import networkx as nx
import pydantic

RecipeElementType = Literal["atomic", "workflow", "for", "while", "try", "if"]

RESERVED_NAMES = {"inputs", "outputs"}  # No having child nodes with these names


class UnpackMode(str, Enum):
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


class AtomicNode(NodeModel):
    type: Literal["atomic"] = "atomic"
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


class SourceHandle(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)
    node: str | None
    port: str

    @pydantic.model_serializer
    def serialize(self) -> str:
        if self.node is None:
            return f"inputs.{self.port}"
        return f"{self.node}.outputs.{self.port}"

    @pydantic.model_validator(mode="before")
    @classmethod
    def deserialize(cls, data):
        if isinstance(data, str):
            parts = data.split(".", 2)
            if parts[0] == "inputs":
                return {"node": None, "port": parts[1]}
            return {"node": parts[0], "port": parts[-1]}
        return data


class TargetHandle(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)
    node: str | None
    port: str

    @pydantic.model_serializer
    def serialize(self) -> str:
        if self.node is None:
            return f"outputs.{self.port}"
        return f"{self.node}.inputs.{self.port}"

    @pydantic.model_validator(mode="before")
    @classmethod
    def deserialize(cls, data):
        if isinstance(data, str):
            parts = data.split(".", 2)
            if parts[0] == "outputs":
                return {"node": None, "port": parts[1]}
            return {"node": parts[0], "port": parts[-1]}
        return data


class WorkflowNode(NodeModel):
    type: Literal["workflow"] = "workflow"
    nodes: dict[str, "NodeType"]
    edges: dict[TargetHandle, SourceHandle]
    reserved_node_names: ClassVar[frozenset[str]] = frozenset({"inputs", "outputs"})

    @pydantic.field_validator("edges")
    @classmethod
    def validate_edges(cls, v):
        for target, source in v.items():
            if target.node is None and source.node is None:
                raise ValueError(
                    f"Invalid edge: No pass-through data -- if a workflow declares IO "
                    f"it should use it. Got target={target}, source={source}"
                )
        return v

    @pydantic.model_validator(mode="after")
    def validate_reserved_node_names(self):
        conflicts = self.reserved_node_names & set(self.nodes.keys())
        if conflicts:
            raise ValueError(
                f"Node labels cannot use reserved names: {conflicts}. "
                f"Reserved names are: {self.reserved_node_names}"
            )
        return self

    @pydantic.model_validator(mode="after")
    def validate_edge_references(self):
        """Validate that edges reference existing nodes and valid ports."""
        node_labels = set(self.nodes.keys())
        workflow_inputs = set(self.inputs)
        workflow_outputs = set(self.outputs)

        for target, source in self.edges.items():
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


# Discriminated Union
NodeType = Annotated[
    AtomicNode | WorkflowNode,
    pydantic.Field(discriminator="type"),
]

WorkflowNode.model_rebuild()
