import ast
import inspect
import textwrap
from types import FunctionType
from typing import Any

from flowrep import workflow


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
