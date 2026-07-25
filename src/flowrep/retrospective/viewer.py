"""
Viewer helpers for retrospective data objects.

Provides a notebook-friendly JSON view and a plain-text fallback.
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from flowrep.retrospective.datastructures import NodeData

try:
    from IPython.display import JSON as _JSON

    _has_ipython = True
except ImportError:
    _has_ipython = False


def _to_jsonable(data: Any) -> Any:
    if data is None or isinstance(data, bool | int | float | str):
        return data
    if isinstance(data, Mapping):
        return {
            _to_jsonable_key(key): _to_jsonable(value) for key, value in data.items()
        }
    if isinstance(data, Sequence) and not isinstance(data, str):
        return [_to_jsonable(item) for item in data]
    if dataclasses.is_dataclass(data) and not isinstance(data, type):
        serialized = {
            field.name: _to_jsonable(getattr(data, field.name))
            for field in dataclasses.fields(data)
        }
        serialized["type"] = data.__class__.__name__
        return serialized
    if hasattr(data, "model_dump"):
        model_dump = data.model_dump(mode="json")
        if isinstance(model_dump, Mapping):
            model_dump = {**model_dump, "type": data.__class__.__name__}
        return _to_jsonable(model_dump)
    if isinstance(data, type):
        return f"{data.__module__}.{data.__qualname__}"
    if inspect.isroutine(data):
        return f"{data.__module__}.{data.__qualname__}"
    return repr(data)


def _to_jsonable_key(key: Any) -> str:
    jsonable = _to_jsonable(key)
    if isinstance(jsonable, str):
        return jsonable
    return repr(jsonable)


def _view_json(data: NodeData, *, expanded: bool = False):
    """Display *data* as structured JSON (requires IPython)."""
    return _JSON(_to_jsonable(data), expanded=expanded)


def _view_str(data: NodeData) -> str:
    """Return a plain-text representation of *data*."""
    return str(data)


def view(data: NodeData, *, expanded: bool = False):
    """
    Display *data* as structured JSON when IPython is available, otherwise
    return a plain-text representation.
    """
    if _has_ipython:
        return _view_json(data, expanded=expanded)
    return _view_str(data)
