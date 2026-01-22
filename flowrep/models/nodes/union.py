from typing import Annotated

import pydantic

from flowrep.models import model
from flowrep.models.nodes import for_model, if_model, try_model, while_model

# Discriminated Union
NodeType = Annotated[
    model.AtomicNode
    | model.WorkflowNode
    | for_model.ForNode
    | if_model.IfNode
    | try_model.TryNode
    | while_model.WhileNode,
    pydantic.Field(discriminator="type"),
]
