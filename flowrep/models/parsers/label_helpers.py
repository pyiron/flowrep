import inspect
from collections.abc import Collection, Iterable
from types import FunctionType
from typing import Annotated, Any, Self, get_args, get_origin, get_type_hints

from pydantic import BaseModel


class OutputMeta(BaseModel, extra="ignore"):
    """
    Metadata for output port annotations.

    Can be used directly in Annotated hints or as a dict (which will be coerced).
    Extra keys are ignored, allowing interoperability with other packages.
    Downstream packages can explicitly `extra="forbid"` to lock things down again.

    Examples:
        # Using the model directly
        def f(x) -> Annotated[float, OutputMeta(label="result")]:
            ...

        # Using a plain dict (coerced automatically)
        def f(x) -> Annotated[float, {"label": "result"}]:
            ...

        # Extra keys are ignored (useful for other packages)
        def f(x) -> Annotated[float, {"label": "result", "units": "m", "iri": "..."}]:
            ...
    """

    label: str | None = None

    @classmethod
    def from_annotation(cls, meta: Any) -> Self | None:
        """
        Attempt to coerce annotation metadata into OutputMeta.

        Returns None if the metadata cannot be interpreted as OutputMeta.
        """
        if isinstance(meta, cls):
            return meta
        if isinstance(meta, dict):
            try:
                return cls.model_validate(meta)
            except Exception:
                return None
        return None


def extract_label_from_annotated(hint: Any) -> str | None:
    """
    Extract label from an Annotated type hint.

    Accepts either OutputMeta instances or dicts with a "label" key.
    Returns None if no label metadata found.
    """
    if get_origin(hint) is Annotated:
        args = get_args(hint)
        # args[0] is the actual type, args[1:] are metadata
        for meta in args[1:]:
            parsed = OutputMeta.from_annotation(meta)
            if parsed is not None and parsed.label is not None:
                return parsed.label
    return None


def get_annotated_output_labels(func: FunctionType) -> list[str | None] | None:
    """
    Extract output labels from return type annotation using Annotated.

    For TUPLE unpacking - looks at tuple element annotations.
    Unwraps outer Annotated wrapper if present to get to tuple elements.

    Supports:
        - Single: `-> Annotated[T, {"label": "name"}]`
        - Tuple:  `-> tuple[Annotated[T1, {"label": "a"}], Annotated[T2, {"label": "b"}]]`
        - Wrapped: `-> Annotated[tuple[Annotated[...], ...], {"label": "ignored"}]`

    Returns None if no annotation or no label metadata found.
    Returns list with None elements for positions without labels.
    """
    try:
        hints = get_type_hints(func, include_extras=True)
    except Exception:
        return None

    return_hint = hints.get("return")
    if return_hint is None:
        return None

    # Unwrap outer Annotated to get to the actual type (for TUPLE mode,
    # we care about element annotations, not the tuple-level annotation)
    inner_type = return_hint
    if get_origin(return_hint) is Annotated:
        inner_type = get_args(return_hint)[0]

    origin = get_origin(inner_type)

    # Handle tuple returns - look at element annotations
    if origin is tuple:
        args = get_args(inner_type)
        # Handle tuple[T, ...] (homogeneous variable-length) - can't extract labels
        if len(args) == 2 and args[1] is ...:
            return None
        labels = [extract_label_from_annotated(arg) for arg in args]
        # Return None if no labels found at all
        if all(label is None for label in labels):
            return None
        return labels

    # Single return value - use original hint (may have Annotated wrapper)
    label = extract_label_from_annotated(return_hint)
    if label is not None:
        return [label]
    return None


def merge_labels(
    first_choice: Collection[str | None] | None,
    fallback: Collection[str],
    message_prefix: str = "",
) -> list[str]:
    if first_choice is None:
        return list(fallback)
    else:
        if len(first_choice) != len(fallback):
            raise ValueError(
                message_prefix + f"Cannot merge {first_choice} and {fallback} because "
                f"number of elements differ."
            )
        return list(
            first if first is not None else fall
            for first, fall in zip(first_choice, fallback, strict=True)
        )


def get_input_labels(func: FunctionType) -> list[str]:
    sig = inspect.signature(func)
    for param in sig.parameters.values():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise ValueError(
                f"Function arguments cannot contain *args or **kwargs, got "
                f"{list(sig.parameters.keys())}"
            )
    return list(sig.parameters.keys())


def default_output_label(i: int) -> str:
    return f"output_{i}"


def unique_suffix(name: str, references: Iterable[str]) -> str:
    # This is obviously horribly inefficient, but fix that later
    i = 0
    new_name = f"{name}_{i}"
    while new_name in references:
        i += 1
        new_name = f"{name}_{i}"
    return new_name
