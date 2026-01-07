from typing import Literal
import pydantic

RecipeElementType = Literal["atomic", "workflow", "for", "while", "try", "if"]

class NodeModel(pydantic.BaseModel):
    type: RecipeElementType

class AtomicNode(NodeModel):
    type: Literal["atomic"] = "atomic"

class WorkflowNode(NodeModel):
    type: Literal["workflow"] = "workflow"
