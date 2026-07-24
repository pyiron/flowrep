from __future__ import annotations

from typing import Literal

import pydantic
from pyiron_snippets import retrieve

from flowrep import base_models


class AtomicRecipe(base_models.NodeRecipe):
    """
    Atomos: uncuttable, indivisible.

    A node representing a python function call.

    Intended recipe realization:
    - Atomic nodes do not have internal structure from the perspective of a workflow
        graph.
    - The actions _inside_ them are ephemeral and not available for retrospective
        inspection.
    - As with all nodes, their IO should be available for retrospective inspection.
    - The conversion of function return values to node outputs is controlled via the
        the number of output labels.

    Attributes:
        type: The node type -- always "atomic".
        inputs: The available input port names.
        outputs: The available output port names.
        reference: Info about the underlying python function.

    Properties:
        fully_qualified_name: The fully qualified name of the function to call, i.e.
            module and qualname as a dot-separated string.
    """

    type: Literal[base_models.RecipeElementType.ATOMIC] = pydantic.Field(
        default=base_models.RecipeElementType.ATOMIC, frozen=True
    )
    reference: base_models.PythonReference

    @property
    def inputs_with_defaults(self) -> base_models.Labels:
        return self.reference.inputs_with_defaults

    @property
    def fully_qualified_name(self) -> str:
        return self.reference.info.fully_qualified_name

    def __call__(self, *args, **kwargs):
        func = retrieve.import_from_string(self.reference.info.fully_qualified_name)
        return func(*args, **kwargs)
