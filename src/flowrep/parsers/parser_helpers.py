from __future__ import annotations

import ast
import dataclasses
import inspect
import textwrap
from collections.abc import Callable, Iterable
from types import FunctionType
from typing import Any, TypeVar, cast

from flowrep import base_models, edge_models
from flowrep.parsers import (
    attribute_parser,
    constant_parser,
    label_helpers,
    symbol_scope,
)
from flowrep.prospective import constant_recipe, helper_models, union_types


class SourceCodeUnavailableError(ValueError): ...


_D = TypeVar("_D")


def apply_label_decorator(
    func: _D | str | None,
    output_labels: tuple[str, ...],
    *,
    wrap: Callable[[_D, tuple[str, ...]], _D],
    decorator_name: str,
    allowed_types: tuple[type, ...],
) -> _D | Callable[[_D], _D]:
    if not isinstance(func, str | None):
        func = _normalize_target(func, decorator_name)

    if isinstance(func, allowed_types):
        return wrap(cast("_D", func), ())
    elif func is None:
        labels = output_labels
    elif isinstance(func, str):
        labels = (func, *output_labels)
    else:
        _ensure_allowed(func, allowed_types, decorator_name)

    def deferred(f: _D) -> _D:
        f = _normalize_target(f, decorator_name)
        _ensure_allowed(f, allowed_types, decorator_name)
        return wrap(f, labels)

    return deferred


def _normalize_target(f: Any, decorator_name: str) -> Any:
    if isinstance(f, classmethod):
        raise TypeError(
            f"{decorator_name} cannot decorate a classmethod: its `cls` is bound "
            f"away at access time and won't match the parsed inputs. Apply "
            f"@classmethod outside {decorator_name}, or use @staticmethod."
        )
    if isinstance(f, staticmethod):
        raise TypeError(f"{decorator_name} should be placed beneath `@staticmethod`.")
    return f


def _ensure_allowed(
    f: Any, allowed_types: tuple[type, ...], decorator_name: str
) -> None:
    if not isinstance(f, allowed_types):
        allowed = " or ".join(t.__name__ for t in allowed_types)
        raise TypeError(
            f"{decorator_name!r} can only decorate {allowed!r}, got {type(f).__name__!r}"
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
    def of(cls, func: FunctionType | type) -> SignatureInfo:
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
    *,
    condition_bindings: dict[str, constant_recipe.ConstantRecipe] | None = None,
    reserved_ports: set[str] | None = None,
    hoisted: dict[ast.expr, edge_models.SourceHandle] | None = None,
) -> None:
    """Record all argument->port consumptions for a node-creating call.

    ``ast.Name`` arguments consume an existing symbol; any other argument must be a
    Python literal. In the default (non-condition) mode a literal is injected as a
    ``ConstantRecipe`` source node into *nodes* and wired to the consuming port. In
    condition mode -- selected by passing a *condition_bindings* dict -- a literal is
    instead routed to a synthetic flow-control input port and recorded in
    *condition_bindings* (``{synthetic_port: ConstantRecipe}``) for the enclosing
    walker to satisfy with a constant peer. A flow-control node has no room to host a
    constant peer inside it, so the peer lives one level up. *reserved_ports* carries
    synthetic port names already allocated for this flow-control chain so that
    multiple literals -- across an if/elif chain -- get distinct, deterministic ports.

    Data-attribute arguments are injected ahead of the call by
    ``attribute_parser.hoist_call_arguments`` and arrive here in *hoisted*, mapping
    the argument's AST node to the ``SourceHandle`` of its outermost getattr node. In
    condition mode there is no room for a peer getattr node inside a flow-control
    condition, so a data-attribute argument is rejected instead.
    """
    reserved = set() if reserved_ports is None else reserved_ports
    already_hoisted = {} if hoisted is None else hoisted

    def _consume(arg_node: ast.expr, consumer_port: str) -> None:
        if (handle := already_hoisted.get(arg_node)) is not None:
            scope.consume_source(handle, child.label, consumer_port)
            return
        if isinstance(arg_node, ast.Name):
            scope.consume(arg_node.id, child.label, consumer_port)
            return
        if attribute_parser.is_data_attribute(arg_node, scope):
            raise ValueError(
                f"Attribute access on workflow data is not supported in flow-control "
                f"condition arguments; for input '{consumer_port}' of condition node "
                f"'{child.label}' found '{ast.unparse(arg_node)}'. Bind it to a "
                f"symbol before the flow control and pass that symbol instead."
            )
        is_literal, value = constant_parser.try_parse_constant(arg_node)
        if not is_literal:
            raise TypeError(
                f"Workflow python definitions can only interpret function calls with "
                f"symbolic input or literal constants; for input '{consumer_port}' of "
                f"node '{child.label}' found un-parseable "
                f"{type(arg_node).__name__}."
            )
        if condition_bindings is not None:
            _bind_condition_constant(
                scope, child, consumer_port, value, condition_bindings, reserved
            )
            return
        constant_parser.inject_constant(nodes, scope, value, child.label, consumer_port)

    for i, arg in enumerate(ast_call.args):
        _consume(arg, child.recipe.inputs[i])
    for kw in ast_call.keywords:
        if not isinstance(kw.arg, str):  # pragma: no cover
            raise TypeError(
                "How did you get here? A `None` value should be possible for "
                "**kwargs, but variadics should have been excluded before "
                "this. Please raise a GitHub issue."
            )
        _consume(kw.value, kw.arg)


def _bind_condition_constant(
    scope: symbol_scope.SymbolScope,
    child: helper_models.LabeledRecipe,
    consumer_port: str,
    value: Any,
    condition_bindings: dict[str, constant_recipe.ConstantRecipe],
    reserved_ports: set[str],
) -> None:
    """Expose a literal condition argument as a synthetic flow-input port.

    The synthetic port name is unique across the condition's real argument symbols
    (``scope.inputs``) and every synthetic port already allocated for this
    flow-control chain (*reserved_ports*), so it is a deterministic function of the
    source and round-trips exactly. The ``ConstantRecipe`` is built eagerly (so a
    non-JSON literal such as a tuple raises ``ConstantParseError`` with call-site
    context here, matching non-condition timing) and handed up for the enclosing
    walker to attach as a peer.
    """
    synthetic_port = label_helpers.unique_suffix(
        constant_recipe.ConstantRecipe.std_label, set(scope.inputs) | reserved_ports
    )
    reserved_ports.add(synthetic_port)
    condition_bindings[synthetic_port] = constant_parser.make_constant(
        value,
        f"Condition argument for input '{consumer_port}' of node '{child.label}'",
    )
    scope.consume_input_source(
        edge_models.InputSource(port=synthetic_port), child.label, consumer_port
    )


def reject_input_alias_outputs(
    body_symbol_map: symbol_scope.SymbolScope,
    candidate_outputs: Iterable[str],
    control_kind: str,
) -> None:
    """Raise if a flow-control body tries to surface an input-alias as an output.

    A symbol aliased directly to a workflow input keeps an ``InputSource`` rather
    than a node ``SourceHandle``. Such a symbol cannot be represented as a
    ``{control_kind}`` output, so we reject it with a clear error instead of
    emitting an invalid recipe.
    """
    offending = sorted(
        s for s in candidate_outputs if body_symbol_map.is_input_alias(s)
    )
    if offending:
        raise ValueError(
            f"A {control_kind} body assigns workflow input(s) directly to symbol(s) "
            f"{offending} that must become {control_kind} outputs. Route the value "
            f"through a node instead of aliasing an input."
        )
