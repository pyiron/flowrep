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
    label = label_helpers.unique_suffix("constant", nodes)
    try:
        recipe = constant_recipe.ConstantRecipe(constant=value)
    except pydantic.ValidationError as error:
        raise ConstantParseError(
            f"Argument for input '{consumer_port}' of node '{consumer_label}' is not "
            f"a JSON-serializable constant: {value!r}"
        ) from error
    nodes[label] = recipe
    scope.consume_source(
        edge_models.SourceHandle(node=label, port="constant"),
        consumer_label,
        consumer_port,
    )
