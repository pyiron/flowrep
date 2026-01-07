from typing import Annotated, Literal

import pydantic

RecipeElementType = Literal["atomic", "workflow", "for", "while", "try", "if"]


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
    nodes: list["NodeType"]

# Discriminated Union
NodeType = Annotated[
    AtomicNode
    | WorkflowNode,
    pydantic.Field(discriminator="type"),
]

WorkflowNode.model_rebuild()
