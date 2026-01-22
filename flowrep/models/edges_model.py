from __future__ import annotations

from typing import ClassVar

import pydantic


class HandleModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)
    node: str | None
    port: str
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
    node: str
    port: str


class TargetHandle(HandleModel):
    node: str
    port: str


class InputSource(HandleModel):
    node: None = pydantic.Field(default=None, frozen=True)
    port: str


class OutputTarget(HandleModel):
    node: None = pydantic.Field(default=None, frozen=True)
    port: str
