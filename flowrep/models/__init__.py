from typing import cast

import pydantic

from flowrep.models.nodes import (
    atomic_model,
    for_model,
    if_model,
    model,
    try_model,
    while_model,
    workflow_model,
)
from flowrep.models.nodes.union import NodeType

for cls in [
    atomic_model.AtomicNode,
    for_model.ForNode,
    if_model.IfNode,
    model.LabeledNode,
    try_model.TryNode,
    while_model.WhileNode,
    workflow_model.WorkflowNode,
]:
    cast(type[pydantic.BaseModel], cls).model_rebuild()
