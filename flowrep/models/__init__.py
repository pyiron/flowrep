from typing import cast

import pydantic

from flowrep.models.nodes import model
from flowrep.models.nodes.union import NodeType

for cls in [
    model.AtomicNode,
    model.ForNode,
    model.IfNode,
    model.LabeledNode,
    model.TryNode,
    model.WorkflowNode,
    model.WhileNode,
]:
    cast(type[pydantic.BaseModel], cls).model_rebuild()
