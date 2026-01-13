import ast
import dataclasses
import inspect
import textwrap
from types import FunctionType

from flowrep import model, workflow


def atomic(
    func=None, /, *output_labels, unpack_mode: model.UnpackMode = model.UnpackMode.TUPLE
):
    """
    Decorator that attaches a flowrep.model.AtomicNode to the `recipe` attribute of a
    function.

    Can be used as with or without kwargs -- @atomic or @atomic(unpack_mode=...)
    """

    def decorator(f):
        f.recipe = parse_atomic(f, *output_labels, unpack_mode=unpack_mode)
        return f

    # If func is provided and is actually a function, apply decorator directly
    if func is not None and callable(func):
        return decorator(func)

    # Otherwise, func and output_labels contain the arguments, return decorator
    # Combine func into output_labels if it's not None
    all_output_labels = (func,) + output_labels if func is not None else output_labels

    def decorator_with_args(f):
        f.recipe = parse_atomic(f, *all_output_labels, unpack_mode=unpack_mode)
        return f

    return decorator_with_args


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


def default_output_label(i: int):
    return f"output_{i}"


def _get_output_labels(func: FunctionType, unpack_mode: model.UnpackMode) -> list[str]:
    if unpack_mode == model.UnpackMode.NONE:
        return ["output_0"]
    elif unpack_mode == model.UnpackMode.TUPLE:
        return _parse_tuple_return_labels(func)
    elif unpack_mode == model.UnpackMode.DATACLASS:
        return _parse_dataclass_return_labels(func)
    raise TypeError(
        f"Invalid unpack mode: {unpack_mode}. Possible values are "
        f"{', '.join(model.UnpackMode.__members__.values())}"
    )


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
    return list(
        (
            label
            if all(other_branch[i] == label for other_branch in return_labels)
            else default_output_label(i)
        )
        for i, label in enumerate(return_labels[0])
    )


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
    return_annotation = sig.return_annotation

    if return_annotation is inspect.Parameter.empty or not dataclasses.is_dataclass(
        return_annotation
    ):
        raise ValueError(
            f"Dataclass unpack mode requires a return type annotation that is a "
            f"dataclass, but got {return_annotation}"
        )

    return [field.name for field in dataclasses.fields(return_annotation)]
