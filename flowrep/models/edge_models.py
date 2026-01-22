from __future__ import annotations

from typing import ClassVar

import pydantic

from flowrep.models import base_models


class HandleModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)
    node: base_models.Label | None
    port: base_models.Label
    delimiter: ClassVar[str] = "."

    @pydantic.model_serializer
    def serialize(self) -> str:
        if self.node is None:
            return self.port
        return self.delimiter.join([self.node, self.port])

    @pydantic.model_validator(mode="before")
    @classmethod
    def deserialize(cls, data):
        if isinstance(data, str):
            parts = data.split(".", 1)
            if len(parts) == 1:
                return {"node": None, "port": parts[0]}
            return {"node": parts[0], "port": parts[1]}
        return data


class SourceHandle(HandleModel):
    node: base_models.Label
    port: base_models.Label


class TargetHandle(HandleModel):
    node: base_models.Label
    port: base_models.Label


class InputSource(HandleModel):
    node: None = pydantic.Field(default=None, frozen=True)
    port: base_models.Label


class OutputTarget(HandleModel):
    node: None = pydantic.Field(default=None, frozen=True)
    port: base_models.Label


Edges = dict[OutputTarget, SourceHandle]  # Communicate between siblings
InputEdges = dict[TargetHandle, InputSource]  # Pass data into a subgraph
OutputEdges = dict[OutputTarget, SourceHandle]  # Extract data from a subgraph
