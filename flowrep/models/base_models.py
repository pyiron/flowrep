from __future__ import annotations

import inspect
import keyword
from collections.abc import Hashable
from enum import StrEnum
from typing import Annotated, ClassVar, Self, TypeVar

import pydantic
import pydantic_core
from pyiron_snippets import versions


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


def is_valid_label(label: str) -> bool:
    return (
        label.isidentifier()
        and not keyword.iskeyword(label)
        and label not in RESERVED_NAMES
    )


def _validate_label(v: str) -> str:
    if not isinstance(v, str) or not is_valid_label(v):
        raise ValueError(
            f"Label must be a valid Python identifier and not in "
            f"reserved labels {RESERVED_NAMES}. Got '{v}'"
        )
    return v


Label = Annotated[str, pydantic.BeforeValidator(_validate_label)]


T = TypeVar("T", bound=Hashable)


def validate_unique(v: list[T], message: str | None = None) -> list[T]:
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

    @property
    def inputs_with_defaults(self) -> Labels:
        return []

    @pydantic.model_validator(mode="after")
    def _check_inputs_with_defaults_subset_of_inputs(self) -> Self:
        ref = getattr(self, "reference", None)
        if ref is not None and isinstance(ref, PythonReference):
            invalid = set(ref.inputs_with_defaults) - set(self.inputs)
            if invalid:
                raise ValueError(
                    f"`reference.inputs_with_defaults` contains labels not in `inputs`: {invalid}"
                )
        return self

    @pydantic.model_validator(mode="after")
    def validate_internal_data_completeness(self):
        return self


class RestrictedParamKind(StrEnum):
    """
    Parameter kinds that impose restrictions on how an input can be passed.

    POSITIONAL_OR_KEYWORD is deliberately excluded — it's the default assumption
    for any input not explicitly represented.

    Variadics are not supported
    """

    POSITIONAL_ONLY = "POSITIONAL_ONLY"
    KEYWORD_ONLY = "KEYWORD_ONLY"
    VAR_POSITIONAL = "VAR_POSITIONAL"
    VAR_KEYWORD = "VAR_KEYWORD"

    @classmethod
    def from_param_kind(cls, kind: inspect._ParameterKind) -> Self | None:
        """Returns None for POSITIONAL_OR_KEYWORD (the unrestricted default)."""
        try:
            return cls(kind.name)
        except ValueError:
            return None

    _VARIADIC: ClassVar[frozenset[RestrictedParamKind]]


# New defs inside Enum get treated as fields, so indicate class var and populate later
RestrictedParamKind._VARIADIC = frozenset(
    {
        RestrictedParamKind.VAR_POSITIONAL,
        RestrictedParamKind.VAR_KEYWORD,
    }
)


class PythonReference(pydantic.BaseModel):
    info: versions.VersionInfo
    inputs_with_defaults: Labels = pydantic.Field(default_factory=list)
    restricted_input_kinds: dict[Label, RestrictedParamKind] = pydantic.Field(
        default_factory=dict
    )

    @pydantic.field_validator("restricted_input_kinds", mode="before")
    @classmethod
    def _reject_variadic(
        cls, v: dict[str, RestrictedParamKind | str]
    ) -> dict[str, RestrictedParamKind | str]:
        for label, kind in v.items():
            # Coerce strings so this works both from raw dicts and enum values
            if RestrictedParamKind(kind) in RestrictedParamKind._VARIADIC:
                raise NotImplementedError(
                    f"Variadic parameter kinds are not supported in workflow "
                    f"nodes. Input '{label}' has kind '{kind}'. Consider "
                    f"wrapping variadic arguments in an explicit collection."
                )
        return v
