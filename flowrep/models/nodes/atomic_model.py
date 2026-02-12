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
        fully_qualified_name: The fully qualified name of the function to call, i.e.
            module and qualname as a dot-separated string.
        unpack_mode: How to handle return values from running functions in atomic nodes.
    """

    type: Literal[base_models.RecipeElementType.ATOMIC] = pydantic.Field(
        default=base_models.RecipeElementType.ATOMIC, frozen=True
    )
    fully_qualified_name: str
    unpack_mode: UnpackMode = UnpackMode.TUPLE

    @pydantic.field_validator("fully_qualified_name")
    @classmethod
    def check_name_format(cls, v: str):
        if not v or len(v.split(".")) < 2 or not all(part for part in v.split(".")):
            msg = (
                f"AtomicNode 'fully_qualified_name' must be a non-empty string "
                f"in the format 'module.qualname' with at least one period. Got {v}"
            )
            raise ValueError(msg)
        return v

    @pydantic.model_validator(mode="after")
    def check_outputs_when_not_unpacking(self):
        if self.unpack_mode == UnpackMode.NONE and len(self.outputs) > 1:
            raise ValueError(
                f"Outputs must have exactly one element when unpacking is disabled. "
                f"Got {len(self.outputs)} outputs with "
                f"unpack_mode={self.unpack_mode.value}"
            )
        return self
