from typing import cast

import pydantic

from flowrep.models.nodes import if_model, model, try_model
from flowrep.models.nodes.union import NodeType

for cls in [
    model.AtomicNode,
    model.ForNode,
    if_model.IfNode,
    model.LabeledNode,
    model.WorkflowNode,
    model.WhileNode,
    try_model.TryNode,
]:
    cast(type[pydantic.BaseModel], cls).model_rebuild()
