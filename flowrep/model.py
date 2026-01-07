from enum import Enum
from typing import Annotated, Literal

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


class WorkflowNode(NodeModel):
    type: Literal["workflow"] = "workflow"
    nodes: dict[str, "NodeType"]
    edges: dict[
        str | tuple[str, str],
        str | tuple[str, str],
    ]  # But dict[str, str] gets disallowed in validation

    @pydantic.model_serializer(mode="wrap", when_used="json")
    def serialize_model(self, serializer, info):
        """Convert edges dict to list of pairs for JSON serialization only."""
        data = serializer(self)
        if info.mode == "json":
            # Convert dict to list for JSON (since JSON keys must be strings)
            data["edges"] = [[k, v] for k, v in self.edges.items()]
        # For Python mode, keep the original dict
        return data

    @pydantic.field_validator("edges", mode="before")
    @classmethod
    def deserialize_edges(cls, v):
        """Convert list of pairs back to dict when deserializing from JSON."""
        if isinstance(v, list):
            return {
                tuple(k) if isinstance(k, list) else k: (
                    tuple(val) if isinstance(val, list) else val
                )
                for k, val in v
            }
        # If already a dict (e.g., from Python mode), pass through
        return v

    @pydantic.field_validator("edges")
    @classmethod
    def validate_edges(cls, v):
        for target, source in v.items():
            target_is_str = isinstance(target, str)
            source_is_str = isinstance(source, str)

            if target_is_str and source_is_str:
                raise ValueError(
                    f"Invalid edge: both target and source cannot be plain strings. "
                    f"Got target={target}, source={source}"
                )

            if isinstance(target, tuple) and len(target) != 2:
                raise ValueError(f"Target tuple must have 2 elements, got {target}")
            if isinstance(source, tuple) and len(source) != 2:
                raise ValueError(f"Source tuple must have 2 elements, got {source}")

        return v

    @pydantic.model_validator(mode="after")
    def validate_reserved_node_names(self):
        conflicts = RESERVED_NAMES & set(self.nodes.keys())
        if conflicts:
            raise ValueError(
                f"Node labels cannot use reserved names: {conflicts}. "
                f"Reserved names are: {RESERVED_NAMES}"
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
            if isinstance(source, tuple):
                node_name, port_name = source
                if node_name not in node_labels:
                    raise ValueError(
                        f"Invalid edge source: node '{node_name}' is not a child node"
                    )
                if port_name not in self.nodes[node_name].outputs:
                    raise ValueError(
                        f"Invalid edge source: node '{node_name}' has no output port "
                        f"'{port_name}'. "
                        f"Available outputs: {self.nodes[node_name].outputs}"
                    )
            elif isinstance(source, str):
                if source not in workflow_inputs:
                    raise ValueError(
                        f"Invalid edge source: '{source}' is not a workflow input. "
                        f"Available inputs: {self.inputs}"
                    )

            # Validate target
            if isinstance(target, tuple):
                node_name, port_name = target
                if node_name not in node_labels:
                    raise ValueError(
                        f"Invalid edge target: node '{node_name}' is not a child node"
                    )
                if port_name not in self.nodes[node_name].inputs:
                    raise ValueError(
                        f"Invalid edge target: node '{node_name}' has no input port "
                        f"'{port_name}'. "
                        f"Available inputs: {self.nodes[node_name].inputs}"
                    )
            elif isinstance(target, str):
                if target not in workflow_outputs:
                    raise ValueError(
                        f"Invalid edge target: '{target}' is not a workflow output. "
                        f"Available outputs: {self.outputs}"
                    )

        return self

    @pydantic.model_validator(mode="after")
    def validate_acyclic(self):
        """Ensure the workflow graph is acyclic (DAG)."""
        g = nx.DiGraph()
        g.add_nodes_from(self.nodes.keys())

        for target, source in self.edges.items():
            # Only add edge if both are child nodes (both tuples)
            if isinstance(target, tuple) and isinstance(source, tuple):
                source_node, _ = source
                target_node, _ = target
                g.add_edge(source_node, target_node)

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
