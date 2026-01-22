from __future__ import annotations

import keyword
from enum import StrEnum
from typing import Annotated, Any

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


def _has_unique_elements(values: list[Any]) -> bool:
    return len(values) == len(set(values))


def _valid_label(label: str) -> bool:
    return (
        label.isidentifier()
        and not keyword.iskeyword(label)
        and label not in RESERVED_NAMES
    )


def _get_invalid_labels(labels: list[str] | set[str]) -> set[str]:
    return {label for label in labels if not _valid_label(label)}


def _validate_labels(labels: list[str] | set[str], info) -> None:
    invalid = _get_invalid_labels(labels)
    if invalid:
        raise ValueError(
            f"All elements of '{info.field_name}' must be a valid Python "
            f"identifier and not in the reserved labels {RESERVED_NAMES}. "
            f"{invalid} are non-compliant."
        )


def _validate_label(v: str) -> str:
    if not _valid_label(v):
        raise ValueError(
            f"Label must be a valid Python identifier and not in "
            f"reserved labels {RESERVED_NAMES}. Got '{v}'"
        )
    return v


Label = Annotated[str, pydantic.BeforeValidator(_validate_label)]


class NodeModel(pydantic.BaseModel):
    type: RecipeElementType
    inputs: list[Label]
    outputs: list[Label]

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

    @pydantic.field_validator("inputs", "outputs")
    @classmethod
    def validate_io_labels(cls, v, info):
        if len(v) != len(set(v)):
            duplicates = [x for x in v if v.count(x) > 1]
            raise ValueError(
                f"'{info.field_name}' must contain unique values. "
                f"Found duplicates: {set(duplicates)}"
            )
        return v
