import dataclasses
import inspect
from collections.abc import Callable
from typing import Any, TypeVar, overload

from flowrep.parsers import atomic_parser

_Cls = TypeVar("_Cls", bound=type)

_DATACLASS_KWARGS = frozenset(
    name
    for name, param in inspect.signature(dataclasses.dataclass).parameters.items()
    if param.kind is inspect.Parameter.KEYWORD_ONLY
)


@overload
def dataclass(cls: _Cls, /) -> _Cls: ...
@overload
def dataclass(
    label: str | None = None,
    /,
    *output_labels: str,
    **kwargs: Any,
) -> Callable[[_Cls], _Cls]: ...
def dataclass(cls=None, /, *output_labels, **kwargs):
    """Stack ``@dataclasses.dataclass`` then ``@atomic`` on a class.

    Keyword args are routed by name: those matching ``dataclasses.dataclass``
    go to it, the rest go to ``@atomic``. Usable bare or with arguments.
    """
    dataclass_kwargs = {k: v for k, v in kwargs.items() if k in _DATACLASS_KWARGS}
    atomic_kwargs = {k: v for k, v in kwargs.items() if k not in _DATACLASS_KWARGS}

    bare = isinstance(cls, type)
    labels = output_labels if bare or cls is None else (cls, *output_labels)

    def wrap(c):
        c = dataclasses.dataclass(**dataclass_kwargs)(c)
        return atomic_parser.atomic(*labels, **atomic_kwargs)(c)

    return wrap(cls) if bare else wrap
