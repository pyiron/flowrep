from __future__ import annotations

import ast
from collections.abc import Collection
from types import FunctionType

from flowrep.models.parsers import parser_protocol, scope_helpers


def walk_func_def(
    body_walker: parser_protocol.BodyWalker,
    tree: ast.FunctionDef,
    func: FunctionType,
    inputs: list[str],
    output_labels: Collection[str],
):
    body_walker.inputs = inputs
    scope = scope_helpers.get_scope(func)

    found_return = False
    for body in skip_docstring(tree.body):
        if isinstance(body, ast.Assign | ast.AnnAssign):
            body_walker.handle_assign(body, scope)
        elif isinstance(body, ast.For):
            body_walker.handle_for(body, scope, parsing_function_def=True)
        elif isinstance(body, ast.While | ast.If | ast.Try):
            raise NotImplementedError(
                f"Support for control flow statement {type(body)} is forthcoming."
            )
        elif isinstance(body, ast.Return):
            if found_return:
                raise ValueError(
                    "Workflow python definitions must have exactly one return."
                )
            found_return = True
            # Sets state: outputs, output_edges
            body_walker.handle_return(body, func, output_labels)
        else:
            raise TypeError(
                f"Workflow python definitions can only interpret assignments, a subset "
                f"of flow control (for/while/if/try) and a return, but ast found "
                f"{type(body)} {body.value if hasattr(body, 'value') else ''}"
            )

    if not found_return:
        raise ValueError("Workflow python definitions must have a return statement.")


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
