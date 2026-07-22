import dataclasses
import functools
import inspect
import types
from collections.abc import Callable
from typing import Any, TypeVar, get_type_hints, overload

from pyiron_snippets import versions

from flowrep import base_models
from flowrep.parsers import atomic_parser
from flowrep.prospective import atomic_recipe

_Cls = TypeVar("_Cls", bound=type)

_DATACLASS_KWARGS = frozenset(
    name
    for name, param in inspect.signature(dataclasses.dataclass).parameters.items()
    if param.kind is inspect.Parameter.KEYWORD_ONLY
)

INVERSE_RECIPE_ATTR = "flowrep_recipe_unpacking"


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
        c = atomic_parser.atomic(*labels, **atomic_kwargs)(c)
        c = _bind_inverse_recipe(c, **atomic_kwargs)
        return c

    return wrap(cls) if bare else wrap


def dataclass_fields_to_outputs(dataclass, /):
    field_values = tuple(
        getattr(dataclass, f.name) for f in dataclasses.fields(dataclass)
    )
    if len(field_values) == 1:
        return field_values[0]
    else:
        return field_values


def _bind_inverse_recipe(
    cls,
    version_scraping: versions.VersionScrapingMap | None = None,
    forbid_main: bool = False,
    forbid_locals: bool = False,
    require_version: bool = False,
    **kwargs,  # ignore others
):
    _bound_unpacker = f"_{cls.__name__}_fields_to_outputs"
    _require_absent(cls, _bound_unpacker, INVERSE_RECIPE_ATTR)

    input_name, outputs, func_sig = _parse_dataclass_unpacking(cls)
    unpack = _clone_unpacker_with_signature(func_sig)
    unpack.__name__ = _bound_unpacker
    unpack.__qualname__ = f"{cls.__qualname__}.{_bound_unpacker}"
    unpack.__module__ = cls.__module__
    setattr(cls, _bound_unpacker, staticmethod(unpack))

    dc_version = versions.VersionInfo.of(
        cls,
        version_scraping=version_scraping,
        forbid_main=forbid_main,
        forbid_locals=forbid_locals,
        require_version=require_version,
    )
    reverse_recipe = atomic_recipe.AtomicRecipe(
        inputs=[input_name],
        outputs=outputs,
        description=f"Unpacks {cls.__name__!r} fields into separate outputs into {outputs!r}",
        reference=base_models.PythonReference(
            info=versions.VersionInfo(
                module=dc_version.module,
                qualname=dc_version.qualname + "." + _bound_unpacker,
                version=dc_version.version,
                # A bit of a fib, since we depend on the DC version _and_ flowrep
                # version for complete functionality here. DC is more likely to matter.
            ),
            restricted_input_kinds={
                "dataclass": base_models.RestrictedParamKind.POSITIONAL_ONLY
            },
        ),
        unpack_mode=atomic_recipe.UnpackMode.TUPLE,
    )
    setattr(cls, INVERSE_RECIPE_ATTR, reverse_recipe)
    return cls


def _require_absent(cls, *names):
    for name in names:
        if hasattr(cls, name):
            raise AttributeError(
                f"{cls.__name__!r} already defines {name!r}; refusing to overwrite"
            )


def _parse_dataclass_unpacking(cls):
    hints = get_type_hints(cls)
    fields = dataclasses.fields(cls)

    input_name = "dataclass"
    outputs = [f.name for f in fields]
    # tuple[t0, t1, ...]; __class_getitem__ dodges the type-checker false positive
    if len(fields) == 1:
        return_annotation = hints[outputs[0]]
    else:
        return_annotation = tuple.__class_getitem__(
            tuple(hints[name] for name in outputs)
        )
    func_sig = inspect.Signature(
        parameters=[
            inspect.Parameter(
                input_name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=cls,
            )
        ],
        return_annotation=return_annotation,
    )
    return input_name, outputs, func_sig


def _clone_unpacker_with_signature(signature: inspect.Signature):
    clone = functools.wraps(dataclass_fields_to_outputs)(
        types.FunctionType(
            dataclass_fields_to_outputs.__code__,
            dataclass_fields_to_outputs.__globals__,
        )
    )
    setattr(clone, "__signature__", signature)  # noqa: B010
    clone.__annotations__ = {
        name: param.annotation
        for name, param in signature.parameters.items()
        if param.annotation is not inspect.Parameter.empty
    }
    if signature.return_annotation is not inspect.Signature.empty:
        clone.__annotations__["return"] = signature.return_annotation
    return clone
