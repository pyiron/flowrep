from __future__ import annotations

from enum import StrEnum
from typing import Literal

import pydantic

from flowrep.models import base_models


class UnpackMode(StrEnum):
    """
    How to handle return values from running functions in atomic nodes.

    - NONE: Return the output as a single value
    - TUPLE: Split return into one port per tuple element
    - DATACLASS: Split return into one port per dataclass field
    """

    NONE = "none"
    TUPLE = "tuple"
    DATACLASS = "dataclass"


class AtomicNode(base_models.NodeModel):
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
        `unpack_mode` flag.

    Attributes:
        type: The node type -- always "atomic".
        inputs: The available input port names.
        outputs: The available output port names.
        reference: Info about the underlying python function.
        unpack_mode: How to handle return values from running functions in atomic nodes.

    Properties:
        fully_qualified_name: The fully qualified name of the function to call, i.e.
            module and qualname as a dot-separated string.
    """

    type: Literal[base_models.RecipeElementType.ATOMIC] = pydantic.Field(
        default=base_models.RecipeElementType.ATOMIC, frozen=True
    )
    reference: base_models.PythonReference
    source_code: str | None = None
    unpack_mode: UnpackMode = UnpackMode.TUPLE

    @property
    def has_default(self) -> base_models.Labels:
        return self.reference.has_default

    @property
    def fully_qualified_name(self) -> str:
        return self.reference.info.fully_qualified_name

    @pydantic.model_validator(mode="after")
    def check_outputs_when_not_unpacking(self):
        if self.unpack_mode == UnpackMode.NONE and len(self.outputs) > 1:
            raise ValueError(
                f"Outputs must have exactly one element when unpacking is disabled. "
                f"Got {len(self.outputs)} outputs with "
                f"unpack_mode={self.unpack_mode.value}"
            )
        return self
