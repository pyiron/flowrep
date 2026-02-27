from __future__ import annotations

from typing import TYPE_CHECKING

import pydantic
from pyiron_snippets import versions

from flowrep.models import base_models

if TYPE_CHECKING:
    from flowrep.models.nodes.union import NodeType  # Satisfies mypy

    # Still not enough to satisfy ruff, which doesn't understand the string forward
    # reference, even with the TYPE_CHECKING import
    # Better to nonetheless leave the references as strings to make sure the pydantic
    # handling of forward references is maximally robust through the model_rebuild()
    # Ultimately, just silence ruff as needed


class LabeledNode(pydantic.BaseModel):
    label: base_models.Label
    node: "NodeType"  # noqa: F821, UP037


class ConditionalCase(pydantic.BaseModel):
    condition: LabeledNode
    body: LabeledNode
    condition_output: base_models.Label | None = None

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
        exceptions: The version info for exception types against which to except.
        body: The node to couple to these exceptions.

    Note:
        In a try-except case, we expect the exception target to always be a type --
        and more idiomatically a subclass of python's builtin :class:`BaseException`.
        Here, we don't explicitly validate that. Using
        :class:`pyiron_snippets.versions.VersionInfo` allows us to ensure that recipes
        are able to fully specify where exceptions can be found, should a non-builtin
        exception be used.
    """

    exceptions: list[versions.VersionInfo]
    body: LabeledNode

    @pydantic.field_validator("exceptions")
    @classmethod
    def validate_exceptions_not_empty(cls, v):
        if len(v) < 1:
            raise ValueError("ExceptionCase must catch at least one exception type")
        return v
