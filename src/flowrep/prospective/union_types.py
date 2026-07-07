from typing import Annotated

import pydantic

from flowrep import base_models
from flowrep.prospective import (
    atomic_recipe,
    constant_recipe,
    for_recipe,
    if_recipe,
    try_recipe,
    while_recipe,
    workflow_recipe,
)

# Discriminated Union
RecipeDiscrimination = Annotated[
    atomic_recipe.AtomicRecipe
    | constant_recipe.ConstantRecipe
    | for_recipe.ForEachRecipe
    | if_recipe.IfRecipe
    | try_recipe.TryRecipe
    | while_recipe.WhileRecipe
    | workflow_recipe.WorkflowRecipe,
    pydantic.Field(discriminator="type"),
]

Recipes = dict[base_models.Label, RecipeDiscrimination]
