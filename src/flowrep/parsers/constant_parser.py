import ast
from typing import Any

import pydantic

from flowrep import edge_models
from flowrep.parsers import label_helpers, symbol_scope
from flowrep.prospective import constant_recipe, union_types


class ConstantParseError(ValueError):
    """Raised when a literal call argument cannot become a constant node."""


def try_parse_constant(arg: ast.expr) -> tuple[bool, Any]:
    """Return ``(True, value)`` if *arg* is a Python literal, else ``(False, None)``.

    Uses ``ast.literal_eval``, so it accepts scalars, lists, dicts, and negatives,
    and returns ``(False, None)`` for names, calls, lambdas, comprehensions, etc.
    """
    try:
        return True, ast.literal_eval(arg)
    except (ValueError, SyntaxError, TypeError):
        return False, None


def make_constant(value: Any, context: str) -> constant_recipe.ConstantRecipe:
    """Build a ``ConstantRecipe`` for *value*, re-raising the model's
    ``ValidationError`` as a ``ConstantParseError`` prefixed with *context*.

    The ``ConstantRecipe`` model is the single source of truth for the JSON/finite
    invariant; this only adds call-site context to the error.
    """
    try:
        return constant_recipe.ConstantRecipe(constant=value)
    except pydantic.ValidationError as error:
        raise ConstantParseError(
            f"{context} is not a JSON-serializable constant: {value!r}"
        ) from error


def inject_constant(
    nodes: union_types.Recipes,
    scope: symbol_scope.SymbolScope,
    value: Any,
    consumer_label: str,
    consumer_port: str,
) -> None:
    """Create a ``ConstantRecipe`` for *value*, register it in *nodes*, and wire its
    output into ``(consumer_label, consumer_port)`` as a peer edge.

    The ``ConstantRecipe`` model enforces the JSON invariant; a non-JSON literal
    (e.g. a tuple) surfaces here as a ``ValidationError`` and is re-raised with the
    consuming node/port for context.
    """
    constant_label = constant_recipe.ConstantRecipe.std_label
    label = label_helpers.unique_suffix(constant_label, nodes)
    recipe = make_constant(
        value,
        f"Argument for input '{consumer_port}' of node '{consumer_label}'",
    )
    nodes[label] = recipe
    scope.consume_source(
        edge_models.SourceHandle(node=label, port=constant_label),
        consumer_label,
        consumer_port,
    )
