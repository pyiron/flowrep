from typing import Annotated

import pydantic

from flowrep.models import model
from flowrep.models.nodes import try_model

# Discriminated Union
NodeType = Annotated[
    model.AtomicNode
    | model.WorkflowNode
    | model.ForNode
    | model.WhileNode
    | model.IfNode
    | try_model.TryNode,
    pydantic.Field(discriminator="type"),
]
