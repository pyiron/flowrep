from typing import Annotated

import pydantic

from flowrep.models import model
from flowrep.models.nodes import (
    for_model,
    if_model,
    try_model,
    while_model,
    workflow_model,
)

# Discriminated Union
NodeType = Annotated[
    model.AtomicNode
    | for_model.ForNode
    | if_model.IfNode
    | try_model.TryNode
    | while_model.WhileNode
    | workflow_model.WorkflowNode,
    pydantic.Field(discriminator="type"),
]
