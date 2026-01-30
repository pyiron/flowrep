import ast
import dataclasses
import inspect
import textwrap
from collections.abc import Callable
from types import FunctionType
from typing import Annotated, get_args, get_origin, get_type_hints

from flowrep.models.nodes import atomic_model
from flowrep.models.parsers import ast_helpers, label_helpers


def atomic(
    func: FunctionType | str | None = None,
    /,
    *output_labels: str,
    unpack_mode: atomic_model.UnpackMode = atomic_model.UnpackMode.TUPLE,
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
        ast_helpers.ensure_function(f, "@atomic")
        f.flowrep_recipe = parse_atomic(f, *parsed_labels, unpack_mode=unpack_mode)  # type: ignore[attr-defined]
        return f

    return decorator(target_func) if target_func else decorator


def parse_atomic(
    func: FunctionType,
    *output_labels: str,
    unpack_mode: atomic_model.UnpackMode = atomic_model.UnpackMode.TUPLE,
) -> atomic_model.AtomicNode:
    fully_qualified_name = f"{func.__module__}.{func.__qualname__}"

    input_labels = label_helpers.get_input_labels(func)

    scraped_output_labels = _get_output_labels(func, unpack_mode)
    if len(output_labels) > 0 and len(output_labels) != len(scraped_output_labels):
        raise ValueError(
            f"Explicitly provided output labels must match with function analysis and "
            f"unpacking mode. Expected {len(scraped_output_labels)} output labels with "
            f"unpacking mode '{unpack_mode}', got but got {output_labels}."
        )

    return atomic_model.AtomicNode(
        fully_qualified_name=fully_qualified_name,
        inputs=input_labels,
        outputs=(
            list(output_labels) if len(output_labels) > 0 else scraped_output_labels
        ),
        unpack_mode=unpack_mode,
    )


def _get_output_labels(
    func: FunctionType, unpack_mode: atomic_model.UnpackMode
) -> list[str]:
    if unpack_mode == atomic_model.UnpackMode.NONE:
        return _parse_return_label_without_unpacking(func)
    elif unpack_mode == atomic_model.UnpackMode.TUPLE:
        return _parse_tuple_return_labels(func)
    elif unpack_mode == atomic_model.UnpackMode.DATACLASS:
        return _parse_dataclass_return_labels(func)
    raise TypeError(
        f"Invalid unpack mode: {unpack_mode}. Possible values are "
        f"{', '.join(atomic_model.UnpackMode.__members__.values())}"
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
        return [label_helpers.default_output_label(0)]

    return_hint = hints.get("return")
    if return_hint is None:
        return [label_helpers.default_output_label(0)]

    # Extract label from the outermost Annotated wrapper
    label = label_helpers.extract_label_from_annotated(return_hint)
    return [label] if label is not None else [label_helpers.default_output_label(0)]


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
    func_node = ast_helpers.get_function_definition(ast_tree)
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
            else label_helpers.default_output_label(i)
        )
        for i, label in enumerate(return_labels[0])
    )

    # Override with annotation-based labels where available
    annotated = label_helpers.get_annotated_output_labels(func)
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
        return_labels.append(label_helpers.extract_return_labels(ret))
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
