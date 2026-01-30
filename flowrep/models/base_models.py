from __future__ import annotations

import keyword
from collections.abc import Hashable
from enum import StrEnum
from typing import Annotated, TypeVar

import pydantic
import pydantic_core


class RecipeElementType(StrEnum):
    ATOMIC = "atomic"
    WORKFLOW = "workflow"
    FOR = "for"
    WHILE = "while"
    IF = "if"
    TRY = "try"


class IOTypes(StrEnum):
    INPUTS = "inputs"
    OUTPUTS = "outputs"


RESERVED_NAMES = {
    k for k in IOTypes.__members__.values()
}  # No having labels with these names


def _valid_label(label: str) -> bool:
    return (
        label.isidentifier()
        and not keyword.iskeyword(label)
        and label not in RESERVED_NAMES
    )


def _validate_label(v: str) -> str:
    if not _valid_label(v):
        raise ValueError(
            f"Label must be a valid Python identifier and not in "
            f"reserved labels {RESERVED_NAMES}. Got '{v}'"
        )
    return v


Label = Annotated[str, pydantic.BeforeValidator(_validate_label)]


T = TypeVar("T", bound=Hashable)


def validate_unique(v: list[T], message: str | None) -> list[T]:
    if len(v) != len(set(v)):
        dupes = [x for x in v if v.count(x) > 1]
        raise ValueError(
            message or f"List must have unique elements. Duplicates: {set(dupes)}"
        )
    return v


UniqueList = Annotated[list[T], pydantic.AfterValidator(validate_unique)]
# We want the ordered property of a list -- especially during round-trip serialization
# But also the "one of each element" property of a list
# This is useful, e.g., for guaranteeing that outputs are all uniquely named

Labels = UniqueList[Label]


class NodeModel(pydantic.BaseModel):
    type: RecipeElementType
    inputs: Labels
    outputs: Labels

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        super().__pydantic_init_subclass__(**kwargs)
        if cls.__name__ != NodeModel.__name__:  # I.e. for subclasses
            type_field = cls.model_fields["type"]
            if type_field.default is pydantic_core.PydanticUndefined:
                raise TypeError(
                    f"{cls.__name__} must provide a default value for 'type'"
                )
            if not type_field.frozen:
                raise TypeError(f"{cls.__name__} must mark 'type' as frozen")
