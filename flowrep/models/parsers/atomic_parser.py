import ast
import dataclasses
import inspect
from collections.abc import Callable
from types import FunctionType
from typing import Annotated, get_args, get_origin, get_type_hints

from flowrep.models.nodes import atomic_model
from flowrep.models.parsers import label_helpers, parser_helpers
from flowrep.models.parsers.label_helpers import default_output_label


def atomic(
    func: FunctionType | str | None = None,
    /,
    *output_labels: str,
    unpack_mode: atomic_model.UnpackMode = atomic_model.UnpackMode.TUPLE,
) -> FunctionType | Callable[[FunctionType], FunctionType]:
    """
    Decorator that attaches a flowrep.model.AtomicNode to the `recipe` attribute of a
    function.

    Can be used as with or without args (to specify output labels) and/or kwargs --
    @atomic or @atomic(..., unpack_mode=...)
    """
    return parser_helpers.parser2decorator(
        func,
        output_labels,
        parser=parse_atomic,
        decorator_name="@atomic",
        parser_kwargs={"unpack_mode": unpack_mode},
    )


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
    func_node = parser_helpers.get_ast_function_node(func)
    return_labels = _extract_combined_return_labels(func_node)
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
    return label_helpers.merge_labels(
        first_choice=annotated,
        fallback=scraped,
        message_prefix="Annotations and scraped return labels mis-match. ",
    )


def _extract_combined_return_labels(
    func_node: ast.FunctionDef,
) -> list[tuple[str, ...]]:
    return_stmts = [n for n in ast.walk(func_node) if isinstance(n, ast.Return)]
    return_labels: list[tuple[str, ...]] = [()] if len(return_stmts) == 0 else []
    for ret in return_stmts:
        return_labels.append(_extract_return_labels(ret))
    return return_labels


def _extract_return_labels(ret: ast.Return) -> tuple[str, ...]:
    if ret.value is None:
        return_labels: tuple[str, ...] = ()
        return return_labels
    elif isinstance(ret.value, ast.Tuple):
        return tuple(
            elt.id if isinstance(elt, ast.Name) else default_output_label(i)
            for i, elt in enumerate(ret.value.elts)
        )
    else:
        return (
            (ret.value.id,)
            if isinstance(ret.value, ast.Name)
            else (default_output_label(0),)
        )


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
