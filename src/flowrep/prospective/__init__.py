from typing import cast

import pydantic

from flowrep.prospective import (
    atomic_recipe,
    for_recipe,
    helper_models,
    if_recipe,
    try_recipe,
    while_recipe,
    workflow_recipe,
)
from flowrep.prospective.union_types import RecipeDiscrimination, Recipes

# Subtlety: Anywhere we use `typing.TYPE_CHECKING` to avoid a real import _and_ use
# the guarded object as a pydantic field annotator, we are going to need to make sure
# the annotation is manually stringified, and rebuild the model with the correct value
# for the stringified hint in-scope
# Running a `model_json_schema()` on the model classes should provide a final layer of
# security that pydantic can find all the necessary types.

for cls in [
    atomic_recipe.AtomicRecipe,
    for_recipe.ForEachRecipe,
    helper_models.LabeledRecipe,
    if_recipe.IfRecipe,
    try_recipe.TryRecipe,
    while_recipe.WhileRecipe,
    workflow_recipe.WorkflowRecipe,
]:
    cast(type[pydantic.BaseModel], cls).model_rebuild()
