from __future__ import annotations

import keyword
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

import pydantic
import pydantic_core

if TYPE_CHECKING:
    from flowrep.models.nodes.union import NodeType  # Satisfies mypy

    # Still not enough to satisfy ruff, which doesn't understand the string forward
    # reference, even with the TYPE_CHECKING import
    # Better to nonetheless leave the references as strings to make sure the pydantic
    # handling of forward references is maximally robust through the model_rebuild()
    # Ultimately, just silence ruff as needed


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


RESERVED_NAMES = {"inputs", "outputs"}  # No having child nodes with these names


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


class UnpackMode(StrEnum):
    """How to handle return values from atomic nodes.

    - NONE: Return the output as a single value
    - TUPLE: Split return into one port per tuple element
    - DATACLASS: Split return into one port per dataclass field
    """

    NONE = "none"
    TUPLE = "tuple"
    DATACLASS = "dataclass"


class NodeModel(pydantic.BaseModel):
    type: RecipeElementType
    inputs: list[str]
    outputs: list[str]

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

        _validate_labels(v, info)
        return v


class AtomicNode(NodeModel):
    type: Literal[RecipeElementType.ATOMIC] = pydantic.Field(
        default=RecipeElementType.ATOMIC, frozen=True
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


class LabeledNode(pydantic.BaseModel):
    label: str
    node: "NodeType"  # noqa: F821, UP037

    @pydantic.field_validator("label")
    @classmethod
    def validate_label(cls, v):
        if not _valid_label(v):
            raise ValueError(
                f"Label must be a valid Python identifier and not in "
                f"reserved labels {RESERVED_NAMES}. Got '{v}'"
            )
        return v


class ConditionalCase(pydantic.BaseModel):
    condition: LabeledNode
    body: LabeledNode
    condition_output: str | None = None

    @pydantic.model_validator(mode="after")
    def validate_condition_is_accessible(self):
        if self.condition_output is None:
            if len(self.condition.node.outputs) != 1:
                raise ValueError(
                    f"condition must have exactly one output if condition_output is not "
                    f"provided. Got condition outputs: {self.condition.node.outputs}"
                )
        elif self.condition_output not in self.condition.node.outputs:
            raise ValueError(
                f"condition_output '{self.condition_output}' is not found among "
                f"available outputs: {self.condition.node.outputs}"
            )
        return self

    @pydantic.model_validator(mode="after")
    def validate_distinct_labels(self):
        if self.condition.label == self.body.label:
            raise ValueError(
                f"Condition and body must have distinct labels, "
                f"both are '{self.condition.label}'"
            )
        return self


class ExceptionCase(pydantic.BaseModel):
    """
    An exception/node pair.

    Attributes:
        exceptions: The fully qualified names (i.e. module+qualname) of the exception
            types.
        body: The node to couple to these exceptions.
    """

    exceptions: list[str]
    body: LabeledNode

    @pydantic.field_validator("exceptions")
    @classmethod
    def validate_exceptions_not_empty(cls, v):
        if len(v) < 1:
            raise ValueError("ExceptionCase must catch at least one exception type")
        return v
