from typing import Annotated, Literal

import pydantic

RecipeElementType = Literal["atomic", "workflow", "for", "while", "try", "if"]

RESERVED_NAMES = {"inputs", "outputs"}  # No having child nodes with these names


class NodeModel(pydantic.BaseModel):
    type: RecipeElementType


class AtomicNode(NodeModel):
    type: Literal["atomic"] = "atomic"
    fully_qualified_name: str

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


class WorkflowNode(NodeModel):
    type: Literal["workflow"] = "workflow"
    inputs: list[str]
    outputs: list[str]
    nodes: dict[str, "NodeType"]
    edges: dict[
        str | tuple[str, str],
        str | tuple[str, str],
    ]  # But dict[str, str] gets disallowed in validation

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
        node_labels = set(self.nodes.keys())
        for target, source in self.edges.items():
            for reference in (target, source):
                if isinstance(reference, tuple) and reference[0] not in node_labels:
                        raise ValueError(
                            f"Invalid edge reference: '{reference}' is not a child node"
                        )
        return self


# Discriminated Union
NodeType = Annotated[
    AtomicNode
    | WorkflowNode,
    pydantic.Field(discriminator="type"),
]

WorkflowNode.model_rebuild()
