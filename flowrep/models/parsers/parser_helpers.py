import ast
import inspect
import textwrap
from collections.abc import Callable
from types import FunctionType
from typing import Any, cast

from flowrep import workflow


def parser2decorator(
    func: FunctionType | str | None,
    output_labels: tuple[str, ...],
    *,
    parser: Callable[..., Any],
    decorator_name: str,
    parser_kwargs: dict[str, Any] | None = None,
) -> FunctionType | Callable[[FunctionType], FunctionType]:
    parser_kwargs = parser_kwargs or {}

    if isinstance(func, FunctionType):
        # Direct decoration: @workflow / @atomic
        parsed_labels: tuple[str, ...] = ()
        target_func = func
    elif func is not None and not isinstance(func, str):
        raise TypeError(
            f"{decorator_name} can only decorate functions, got {type(func).__name__}"
        )
    else:
        # Called with args: @decorator(...) or @decorator("label", ...)
        parsed_labels = (func,) + output_labels if func is not None else output_labels
        target_func = None

    def decorator(f: FunctionType) -> FunctionType:
        ensure_function(f, decorator_name)
        f.flowrep_recipe = parser(f, *parsed_labels, **parser_kwargs)  # type: ignore[attr-defined]
        return f

    return decorator(target_func) if target_func else decorator


def ensure_function(f: Any, decorator_name: str) -> None:
    if not isinstance(f, FunctionType):
        raise TypeError(
            f"{decorator_name} can only decorate functions, got {type(f).__name__}"
        )


def get_function_definition(tree: ast.Module) -> ast.FunctionDef:
    if len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef):
        return tree.body[0]
    raise ValueError(
        f"Expected ast to receive a single function defintion, but got "
        f"{workflow._function_to_ast_dict(tree.body)}"
    )


def get_source_code(func: FunctionType) -> str:
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
    return source_code


def get_ast_function_node(func: FunctionType) -> ast.FunctionDef:
    return get_function_definition(ast.parse(get_source_code(func)))


def skip_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    return (
        body[1:]
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        )
        else body
    )


def resolve_symbols_to_strings(
    node: (
        ast.expr | None
    ),  # Expecting a Name or Tuple[Name], and will otherwise TypeError
) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    elif isinstance(node, ast.Tuple) and all(
        isinstance(elt, ast.Name) for elt in node.elts
    ):
        return [cast(ast.Name, elt).id for elt in node.elts]
    else:
        raise TypeError(
            f"Expected to receive a symbol or tuple of symbols from ast.Name or "
            f"ast.Tuple, but could not parse this from {type(node)}."
        )
