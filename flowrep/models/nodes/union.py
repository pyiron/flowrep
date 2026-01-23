from typing import Annotated

import pydantic

from flowrep.models import base_models
from flowrep.models.nodes import (
    atomic_model,
    for_model,
    if_model,
    try_model,
    while_model,
    workflow_model,
)

# Discriminated Union
NodeType = Annotated[
    atomic_model.AtomicNode
    | for_model.ForNode
    | if_model.IfNode
    | try_model.TryNode
    | while_model.WhileNode
    | workflow_model.WorkflowNode,
    pydantic.Field(discriminator="type"),
]

Nodes = dict[base_models.Label, "NodeType"]
