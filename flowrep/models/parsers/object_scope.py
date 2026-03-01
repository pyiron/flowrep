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


def resolve_attribute_to_object(attribute: str, scope: ScopeProxy | object) -> object:
    """
    Resolve a dot-separated attribute string to the actual object it references in the
    given scope. For example, if attribute is "os.path.join", this function will
    return the actual join function from the os.path module.

    Args:
        attribute: A dot-separated string representing the attribute to resolve.
        scope: The scope in which to resolve the attribute. This can be a ScopeProxy
            or any object that supports attribute access.

    Returns:
        The object that the attribute resolves to in the given scope.
    """
    obj = None
    try:
        for attr in attribute.split("."):
            obj = getattr(obj or scope, attr)
        return obj
    except AttributeError as e:
        raise ValueError(f"Could not find attribute '{attr}' of {attribute}") from e


def resolve_symbol_to_object(
    node: ast.expr,  # Expecting a Name or Attribute here, and will otherwise TypeError
    scope: ScopeProxy | object,
    _chain: list[str] | None = None,
) -> object:
    """
    Recursively resolve a symbol in the form of an ast.Name or ast.Attribute to the
    actual object it references in the given scope. The _chain parameter is used
    internally to keep track of the attribute chain being resolved, and should not
    be provided by the caller.

    Args:
        node: An ast.expr representing the symbol to resolve. Expected to be an
            ast.Name or ast.Attribute.
        scope: The scope in which to resolve the symbol. This can be a ScopeProxy
            or any object that supports attribute access.

    Returns:
        The object that the symbol resolves to in the given scope.
    """
    _chain = _chain or []
    if isinstance(node, ast.Name):
        return resolve_attribute_to_object(".".join([node.id] + _chain), scope)
    elif isinstance(node, ast.Attribute):
        return resolve_symbol_to_object(node.value, scope, [node.attr] + _chain)
    else:
        raise TypeError(
            f"Cannot resolve symbol {node} while building the symbol chain "
            f"'{'.'.join(_chain)}'. Expected an ast.Name or chain of ast.Attribute "
            f"and ast.Name, but got {node}."
        )
