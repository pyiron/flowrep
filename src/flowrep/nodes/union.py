from typing import Annotated

import pydantic

from flowrep import base_models
from flowrep.nodes import (
    atomic_model,
    for_model,
    if_model,
    try_model,
    while_model,
    workflow_model,
)

# Discriminated Union
RecipeDiscrimination = Annotated[
    atomic_model.AtomicRecipe
    | for_model.ForEachRecipe
    | if_model.IfRecipe
    | try_model.TryRecipe
    | while_model.WhileRecipe
    | workflow_model.WorkflowRecipe,
    pydantic.Field(discriminator="type"),
]

Recipes = dict[base_models.Label, RecipeDiscrimination]
