from __future__ import annotations

import math
from typing import Any, ClassVar, Literal

import pydantic
from typing_extensions import TypeAliasType

from flowrep import base_models

JSONABLE = TypeAliasType(
    "JSONABLE",
    "dict[str, JSONABLE] | list[JSONABLE] | str | int | float | bool | None",
)


def _strict_json(value: Any) -> Any:
    """Return *value* unchanged if it is strictly JSON, else raise ``ValueError``.

    Runs on the RAW input (``mode="before"``), so it rejects tuples, sets, and
    non-``str`` dict keys that pydantic's lax coercion would otherwise silently
    turn into lists / coerced keys.
    """
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(
            f"Constant values must be JSON-serializable, but got a non-finite "
            f"float: {value!r}"
        )
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        for element in value:
            _strict_json(element)
        return value
    if isinstance(value, dict):
        for key, element in value.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"JSON object keys must be str, got {type(key).__name__}: {key!r}"
                )
            _strict_json(element)
        return value
    raise ValueError(
        f"Constant values must be JSON-serializable, but got a "
        f"{type(value).__name__}: {value!r}"
    )


def is_jsonable(value: Any) -> bool:
    """Whether *value* is JSON-serializable."""
    try:
        _strict_json(value)
        return True
    except ValueError:
        return False


class ConstantRecipe(base_models.NodeRecipe):
    """A source node emitting a fixed, JSON-serializable value.

    Has no inputs and a single output port named ``constant``. Unlike a defaulted
    input (whose value lives only on the referenced function), the value is stored
    directly in the recipe, so it survives serialization with no Python import.
    """

    std_label: ClassVar[str] = "constant"
    # Standardized node label root, and exclusive output port label

    type: Literal[base_models.RecipeElementType.CONSTANT] = pydantic.Field(
        default=base_models.RecipeElementType.CONSTANT, frozen=True
    )
    inputs: base_models.Labels = pydantic.Field(default_factory=list)
    outputs: base_models.Labels = pydantic.Field(
        default_factory=lambda: [ConstantRecipe.std_label]
    )
    constant: JSONABLE

    @pydantic.field_validator("constant", mode="before")
    @classmethod
    def _reject_non_json(cls, v: Any) -> Any:
        return _strict_json(v)

    @pydantic.model_validator(mode="after")
    def _check_ports(self) -> ConstantRecipe:
        if self.inputs != []:
            raise ValueError(f"ConstantRecipe takes no inputs, got {self.inputs}")
        if self.outputs != [self.std_label]:
            raise ValueError(
                f"ConstantRecipe must have a single output named {self.std_label}, "
                f"got {self.outputs}"
            )
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.constant
