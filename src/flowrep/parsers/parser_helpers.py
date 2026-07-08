from __future__ import annotations

import ast
import dataclasses
import inspect
import textwrap
from collections.abc import Callable
from types import FunctionType
from typing import Any, cast

from flowrep import base_models
from flowrep.parsers import constant_parser, symbol_scope
from flowrep.prospective import helper_models, union_types


class SourceCodeUnavailableError(ValueError): ...


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
        f"Expected ast to receive a single function definition, but got a body of "
        f"{[type(t) for t in tree.body]}"
    )


def get_source_code(func: FunctionType) -> str:
    if func.__name__ == "<lambda>":
        raise SourceCodeUnavailableError(
            "Cannot parse return labels for lambda functions. "
            "Use a named function with @atomic decorator."
        )

    try:
        source_code = textwrap.dedent(inspect.getsource(func))
    except (OSError, TypeError) as e:
        raise SourceCodeUnavailableError(
            f"Cannot parse return labels for {func.__qualname__}: "
            f"source code unavailable (lambdas, dynamically defined functions, "
            f"and compiled code are not supported)"
        ) from e
    return source_code


def get_available_source_code(func: FunctionType) -> str | None:
    try:
        return get_source_code(func)
    except SourceCodeUnavailableError:
        return None


@dataclasses.dataclass(frozen=True)
class SignatureInfo:
    names: list[str]
    have_defaults: list[str]
    have_restricted_kinds: dict[str, base_models.RestrictedParamKind]

    @classmethod
    def of(cls, func: FunctionType) -> SignatureInfo:
        sig = inspect.signature(func)
        return SignatureInfo(
            names=list(sig.parameters.keys()),
            have_defaults=[
                label
                for label, param in sig.parameters.items()
                if param.default is not inspect.Parameter.empty
            ],
            have_restricted_kinds={
                label: rk
                for label, param in sig.parameters.items()
                if (rk := base_models.RestrictedParamKind.from_param_kind(param.kind))
                is not None
            },
        )


def get_ast_function_node(func: FunctionType) -> ast.FunctionDef:
    return get_function_definition(ast.parse(get_source_code(func)))


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


def consume_call_arguments(
    scope: symbol_scope.SymbolScope,
    ast_call: ast.Call,
    child: helper_models.LabeledRecipe,
    nodes: union_types.Recipes,
) -> None:
    """Record all argument->port consumptions for a node-creating call.

    ``ast.Name`` arguments consume an existing symbol; any other argument must be a
    Python literal, which is injected as a ``ConstantRecipe`` source node into
    *nodes* and wired to the consuming port.
    """

    def _consume(arg_node: ast.expr, consumer_port: str) -> None:
        if isinstance(arg_node, ast.Name):
            scope.consume(arg_node.id, child.label, consumer_port)
            return
        is_literal, value = constant_parser.try_parse_constant(arg_node)
        if not is_literal:
            raise TypeError(
                f"Workflow python definitions can only interpret function calls with "
                f"symbolic input or literal constants; for input '{consumer_port}' of "
                f"node '{child.label}' found un-parseable "
                f"{type(arg_node).__name__}."
            )
        constant_parser.inject_constant(nodes, scope, value, child.label, consumer_port)

    for i, arg in enumerate(ast_call.args):
        _consume(arg, child.node.inputs[i])
    for kw in ast_call.keywords:
        if not isinstance(kw.arg, str):  # pragma: no cover
            raise TypeError(
                "How did you get here? A `None` value should be possible for "
                "**kwargs, but variadics should have been excluded before "
                "this. Please raise a GitHub issue."
            )
        _consume(kw.value, kw.arg)
