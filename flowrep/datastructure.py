import dataclasses
from collections.abc import Callable, Iterator, MutableMapping
from typing import Any, Generic, TypeAlias, TypeVar

from semantikon import datastructure as sds

TripleType: TypeAlias = tuple[str | None, str, str | None] | tuple[str, str]
TriplesLike: TypeAlias = tuple[TripleType, ...] | TripleType
RestrictionClause: TypeAlias = tuple[str, str]
RestrictionType: TypeAlias = tuple[RestrictionClause, ...]
RestrictionLike: TypeAlias = (
    tuple[RestrictionType, ...]  # Multiple restrictions
    | RestrictionType
    | RestrictionClause  # Short-hand for a single-clause restriction
)
ShapeType: TypeAlias = tuple[int, ...]


@dataclasses.dataclass(slots=True)
class _Lexical(sds._VariadicDataclass, Generic[sds._MetadataType]):
    label: str
    metadata: sds._MetadataType | sds.Missing

    @property
    def type(self) -> str:
        return self.__class__.__name__

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        yield "type", self.type
        yield from super(_Lexical, self).__iter__()


@dataclasses.dataclass(slots=True)
class _Port(_Lexical[sds.TypeMetadata]):
    metadata: sds.TypeMetadata | sds.Missing = sds.missing()
    dtype: type | sds.Missing = sds.missing()
    value: object | sds.Missing = sds.missing()


@dataclasses.dataclass(slots=True)
class Output(_Port):
    pass


@dataclasses.dataclass(slots=True)
class Input(_Port):
    default: Any | sds.Missing = sds.missing()


_ItemType = TypeVar("_ItemType")


class _HasToDictionarMapping(
    sds._HasToDictionary, MutableMapping[str, _ItemType], Generic[_ItemType]
):
    def __init__(self, **kwargs: _ItemType) -> None:
        self._data: dict[str, _ItemType] = kwargs

    def __getitem__(self, key: str) -> _ItemType:
        return self._data[key]

    def __setitem__(self, key: str, value: _ItemType) -> None:
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[tuple[str, _ItemType]]:
        yield from self._data.items()

    def __len__(self) -> int:
        return len(self._data)

    def __getattr__(self, key: str) -> _ItemType:
        return self.__getitem__(key)


PortType = TypeVar("PortType", bound=_Port)


class _IO(_HasToDictionarMapping[PortType], Generic[PortType]): ...


class Inputs(_IO[Input]): ...


class Outputs(_IO[Output]): ...


@dataclasses.dataclass(slots=True)
class _Node(_Lexical[sds.CoreMetadata]):
    metadata: sds.CoreMetadata | sds.Missing
    inputs: Inputs
    outputs: Outputs


@dataclasses.dataclass(slots=True)
class Function(_Node):
    function: Callable


class Nodes(_HasToDictionarMapping[_Node]): ...


EdgeType: TypeAlias = tuple[str, str]


class Edges(_HasToDictionarMapping[str]):
    """
    Key value pairs are stored as `{target: source}` such that each upstream source can
    be used in multiple places, but each downstream target can have only a single
    source.
    The :meth:`to_tuple` routine offers this reversed so that the returned tuples read
    intuitively as `(source, target)`.
    """

    def to_tuple(self) -> tuple[EdgeType, ...]:
        return tuple((e[1], e[0]) for e in self)


@dataclasses.dataclass(slots=True)
class Workflow(_Node):
    nodes: Nodes
    edges: Edges


@dataclasses.dataclass(slots=True)
class While(Workflow):
    test: _Node


@dataclasses.dataclass(slots=True)
class For(Workflow): ...  # TODO


@dataclasses.dataclass(slots=True)
class If(Workflow): ...  # TODO


class ExplicitDefault:
    def __init__(self, default, msg=None):
        self.default = default
        self.msg = msg
