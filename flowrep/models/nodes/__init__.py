from typing import cast

import pydantic

from flowrep.models.nodes import (
    atomic_model,
    base_models,
    for_model,
    helper_models,
    if_model,
    try_model,
    while_model,
    workflow_model,
)
from flowrep.models.nodes.union import NodeType

for cls in [
    atomic_model.AtomicNode,
    for_model.ForNode,
    helper_models.LabeledNode,
    if_model.IfNode,
    try_model.TryNode,
    while_model.WhileNode,
    workflow_model.WorkflowNode,
]:
    cast(type[pydantic.BaseModel], cls).model_rebuild()
