import ast
import dataclasses
import inspect
import textwrap
from collections.abc import Callable
from types import FunctionType
from typing import Annotated, Any, Self, get_args, get_origin, get_type_hints

from pydantic import BaseModel

from flowrep import model, workflow


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


def atomic(
    func: FunctionType | str | None = None,
    /,
    *output_labels: str,
    unpack_mode: model.UnpackMode = model.UnpackMode.TUPLE,
) -> FunctionType | Callable[[FunctionType], FunctionType]:
    """
    Decorator that attaches a flowrep.model.AtomicNode to the `recipe` attribute of a
    function.

    Can be used as with or without kwargs -- @atomic or @atomic(unpack_mode=...)
    """
    parsed_labels: tuple[str, ...]
    if isinstance(func, FunctionType):
        # Direct decoration: @atomic
        parsed_labels = ()
        target_func = func
    elif func is not None and not isinstance(func, str):
        raise TypeError(
            f"@atomic can only decorate functions, got {type(func).__name__}"
        )
    else:
        # Called with args: @atomic(...) or @atomic("label", ...)
        parsed_labels = (func,) + output_labels if func is not None else output_labels
        target_func = None

    def decorator(f: FunctionType) -> FunctionType:
        _ensure_function(f, "@atomic")
        f.flowrep_recipe = parse_atomic(f, *parsed_labels, unpack_mode=unpack_mode)  # type: ignore[attr-defined]
        return f

    return decorator(target_func) if target_func else decorator


def _ensure_function(f: Any, decorator_name: str) -> None:
    if not isinstance(f, FunctionType):
        raise TypeError(
            f"{decorator_name} can only decorate functions, got {type(f).__name__}"
        )


def parse_atomic(
    func: FunctionType,
    *output_labels: str,
    unpack_mode: model.UnpackMode = model.UnpackMode.TUPLE,
) -> model.AtomicNode:
    fully_qualified_name = f"{func.__module__}.{func.__qualname__}"

    input_labels = _get_input_labels(func)

    scraped_output_labels = _get_output_labels(func, unpack_mode)
    if len(output_labels) > 0 and len(output_labels) != len(scraped_output_labels):
        raise ValueError(
            f"Explicitly provided output labels must match with function analysis and "
            f"unpacking mode. Expected {len(scraped_output_labels)} output labels with "
            f"unpacking mode '{unpack_mode}', got but got {output_labels}."
        )

    return model.AtomicNode(
        fully_qualified_name=fully_qualified_name,
        inputs=input_labels,
        outputs=(
            list(output_labels) if len(output_labels) > 0 else scraped_output_labels
        ),
        unpack_mode=unpack_mode,
    )


def _get_function_definition(tree: ast.Module) -> ast.FunctionDef:
    if len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef):
        return tree.body[0]
    raise ValueError(
        f"Expected ast to receive a single function defintion, but got "
        f"{workflow._function_to_ast_dict(tree.body)}"
    )


def _get_input_labels(func: FunctionType) -> list[str]:
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


def _get_output_labels(func: FunctionType, unpack_mode: model.UnpackMode) -> list[str]:
    if unpack_mode == model.UnpackMode.NONE:
        return _parse_return_label_without_unpacking(func)
    elif unpack_mode == model.UnpackMode.TUPLE:
        return _parse_tuple_return_labels(func)
    elif unpack_mode == model.UnpackMode.DATACLASS:
        return _parse_dataclass_return_labels(func)
    raise TypeError(
        f"Invalid unpack mode: {unpack_mode}. Possible values are "
        f"{', '.join(model.UnpackMode.__members__.values())}"
    )


def _parse_return_label_without_unpacking(func: FunctionType) -> list[str]:
    """
    Get output label for UnpackMode.NONE.

    Looks for annotation on the return type itself (not tuple elements).
    For `-> Annotated[T, {"label": "x"}]` or `-> Annotated[tuple[...], {"label": "x"}]`
    """
    try:
        hints = get_type_hints(func, include_extras=True)
    except Exception:
        return [default_output_label(0)]

    return_hint = hints.get("return")
    if return_hint is None:
        return [default_output_label(0)]

    # Extract label from the outermost Annotated wrapper
    label = _extract_label_from_annotated(return_hint)
    return [label] if label is not None else [default_output_label(0)]


def _extract_label_from_annotated(hint: Any) -> str | None:
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


def _get_annotated_output_labels(func: FunctionType) -> list[str | None] | None:
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
        labels = [_extract_label_from_annotated(arg) for arg in args]
        # Return None if no labels found at all
        if all(label is None for label in labels):
            return None
        return labels

    # Single return value - use original hint (may have Annotated wrapper)
    label = _extract_label_from_annotated(return_hint)
    if label is not None:
        return [label]
    return None


def _parse_tuple_return_labels(func: FunctionType) -> list[str]:
    if func.__name__ == "<lambda>":
        raise ValueError(
            "Cannot parse return labels for lambda functions. "
            "Use a named function with @atomic decorator."
        )

    try:
        source_code = textwrap.dedent(inspect.getsource(func))
    except (OSError, TypeError) as e:
        raise ValueError(
            f"Cannot parse return labels for {func.__qualname__}: "
            f"source code unavailable (lambdas, dynamically defined functions, "
            f"and compiled code are not supported)"
        ) from e

    ast_tree = ast.parse(source_code)
    func_node = _get_function_definition(ast_tree)
    return_labels = _extract_return_labels(func_node)
    if not all(len(ret) == len(return_labels[0]) for ret in return_labels):
        raise ValueError(
            f"All return statements must have the same number of elements, got "
            f"{return_labels}"
        )

    # Get AST-scraped labels
    scraped = list(
        (
            label
            if all(other_branch[i] == label for other_branch in return_labels)
            else default_output_label(i)
        )
        for i, label in enumerate(return_labels[0])
    )

    # Override with annotation-based labels where available
    annotated = _get_annotated_output_labels(func)
    if annotated is not None:
        if len(annotated) != len(scraped):
            raise ValueError(
                f"Annotated return type has {len(annotated)} elements but function "
                f"returns {len(scraped)} values"
            )
        # Merge: annotation takes precedence, fall back to scraped
        return [
            ann if ann is not None else scr
            for ann, scr in zip(annotated, scraped, strict=True)
        ]

    return scraped


def _extract_return_labels(func_node: ast.FunctionDef) -> list[tuple[str, ...]]:
    return_stmts = [n for n in ast.walk(func_node) if isinstance(n, ast.Return)]
    return_labels: list[tuple[str, ...]] = [()] if len(return_stmts) == 0 else []
    for ret in return_stmts:
        if ret.value is None:
            return_labels.append(tuple())
        elif isinstance(ret.value, ast.Tuple):
            return_labels.append(
                tuple(
                    elt.id if isinstance(elt, ast.Name) else default_output_label(i)
                    for i, elt in enumerate(ret.value.elts)
                )
            )
        else:
            return_labels.append(
                (ret.value.id,)
                if isinstance(ret.value, ast.Name)
                else (default_output_label(0),)
            )
    return return_labels


def _parse_dataclass_return_labels(func: FunctionType) -> list[str]:
    source_code_return = _parse_tuple_return_labels(func)
    if len(source_code_return) != 1:
        raise ValueError(
            f"Dataclass unpack mode requires function code to returns to consist of "
            f"exactly one value, i.e. the dataclass instance, but got "
            f"{source_code_return}"
        )

    sig = inspect.signature(func)
    ann = sig.return_annotation

    # unwrap Annotated
    origin = get_origin(ann)
    return_annotation = get_args(ann)[0] if origin is Annotated else ann

    if dataclasses.is_dataclass(return_annotation):
        return [f.name for f in dataclasses.fields(return_annotation)]

    raise ValueError(
        f"Dataclass unpack mode requires a return type annotation that is a "
        f"(perhaps Annotated) dataclass, but got {ann}"
    )
