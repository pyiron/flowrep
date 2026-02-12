from __future__ import annotations

import ast
from collections.abc import Collection
from types import FunctionType

from flowrep.models.parsers import object_scope, parser_protocol


def walk_func_def(
    body_walker: parser_protocol.BodyWalker,
    tree: ast.FunctionDef,
    func: FunctionType,
    output_labels: Collection[str],
):
    scope = object_scope.get_scope(func)

    found_return = False
    for body in skip_docstring(tree.body):
        if isinstance(body, ast.Return):
            if found_return:
                raise ValueError(
                    "Workflow python definitions must have exactly one return."
                )
            found_return = True
            body_walker.handle_return(body, func, output_labels)
        else:
            body_walker.visit(body, scope)

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


# Note: There is no FuncDefParser class, because if we are parsing a function
# definition, the state object is already a WorkflowParser -- unlike parsing a control
# flow, no new _additional_ state builder is required.
