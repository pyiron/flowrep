import ast
import builtins
import inspect
from types import FunctionType


class ScopeProxy:
    """
    Make the __dict__-like scope dot-accessible without duplicating the dictionary
    like types.SimpleNamespace would.
    """

    def __init__(self, d: dict):
        self._d = d

    def __getattr__(self, name: str):
        try:
            return self._d[name]
        except KeyError:
            raise AttributeError(name) from None


def get_scope(func: FunctionType) -> ScopeProxy:
    return ScopeProxy(inspect.getmodule(func).__dict__ | vars(builtins))


def resolve_symbol_to_object(
    node: ast.expr,  # Expecting a Name or Attribute here, and will otherwise TypeError
    scope: ScopeProxy | object,
    _chain: list[str] | None = None,
) -> object:
    """ """
    _chain = _chain or []
    error_suffix = f" while attempting to resolve the symbol chain '{'.'.join(_chain)}'"
    if isinstance(node, ast.Name):
        attr = node.id
        try:
            obj = getattr(scope, attr)
            for attr in _chain:
                obj = getattr(obj, attr)
            return obj
        except AttributeError as e:
            raise ValueError(f"Could not find attribute '{attr}' {error_suffix}") from e
    elif isinstance(node, ast.Attribute):
        return resolve_symbol_to_object(node.value, scope, [node.attr] + _chain)
    else:
        raise TypeError(
            f"Cannot resolve symbol {node} {error_suffix}. "
            f"Expected an ast.Name or chain of ast.Attribute and ast.Name, but got "
            f"{node}."
        )
