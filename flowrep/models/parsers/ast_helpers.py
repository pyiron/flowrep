import ast
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
