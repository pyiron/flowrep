from typing import Annotated

import pydantic

from flowrep.models import model

# Discriminated Union
NodeType = Annotated[
    model.AtomicNode
    | model.WorkflowNode
    | model.ForNode
    | model.WhileNode
    | model.IfNode
    | model.TryNode,
    pydantic.Field(discriminator="type"),
]
