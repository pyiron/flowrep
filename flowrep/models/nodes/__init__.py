from typing import cast

import pydantic

from flowrep.models.nodes import (
    atomic_model,
    for_model,
    helper_models,
    if_model,
    try_model,
    while_model,
    workflow_model,
)
from flowrep.models.nodes.union import Nodes, NodeType
# Subtlety: Anywhere we use `typing.TYPE_CHECKING` to avoid a real import _and_ use
# the guarded object as a pydantic field annotator, we are going to need to make sure
# the annotation is manually stringified, and rebuild the model with the correct value
# for the stringified hint in-scope
# Running a `model_json_schema()` on the model classes should provide a final layer of
# security that pydantic can find all the necessary types.

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
