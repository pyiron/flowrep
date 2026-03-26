from __future__ import annotations

import ast
import builtins
import inspect
import sys
from collections.abc import Callable, MutableMapping
from typing import Any


class _EmptyValue: ...


class ScopeProxy(MutableMapping[str, object]):
    """
    A mutable mapping to connect symbols to python objects.

    By default, does not allow re-registration of existing symbols to new values.
    """

    def __init__(
        self,
        d: MutableMapping[str, object] | None = None,
        allow_overwrite: bool = False,
    ):
        self._d = {} if d is None else {k: v for k, v in d.items()}
        self.allow_overwrite = allow_overwrite

    def __getitem__(self, name: str):
        return self._d[name]

    def __setitem__(self, name: str, value: object):
        if not self.allow_overwrite:
            old_value = self._d.get(name, _EmptyValue)
            if old_value is not _EmptyValue and value is not old_value:
                raise ValueError(
                    f"Variable {name} already exists as {old_value!r} in this "
                    f"scope. It cannot be reassigned to a new value of {value!r} "
                    f"while allow_overwrite is False."
                )
            self._d[name] = value
        else:
            self._d[name] = value

    def __delitem__(self, name: str):
        self._d.__delitem__(name)

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __getattr__(self, name: str):
        try:
            return self.__getitem__(name)
        except KeyError:
            raise AttributeError(name) from None

    def __str__(self):
        return str(self._d)

    def register(self, name: str, obj: object) -> None:
        """
        Add a name → object binding to this scope.

        Used by the parser when encountering ``import`` statements inside
        function bodies.  The binding is visible to all subsequent symbol
        resolutions within this scope (or any scope that shares the same
        backing namespace).
        """
        self[name] = obj

    def fork(self) -> ScopeProxy:
        """
        Create a shallow copy of this scope.

        Modifications to the fork (e.g. registering new imports) do *not*
        affect the parent.  Used when walking conditional branches so that
        branch-local imports don't leak into sibling branches.
        """
        return ScopeProxy(self, allow_overwrite=self.allow_overwrite)


def get_scope(func: Callable[..., Any] | type[Any]) -> ScopeProxy:
    module = inspect.getmodule(func)
    if module is None:
        module_name = getattr(func, "__module__", None)
        if module_name is not None:
            module = sys.modules.get(module_name)
    if module is None:
        raise ValueError(
            f"Cannot determine the module for {func!r}. "
            "inspect.getmodule() returned None and no resolvable __module__ "
            "attribute was found."
        )
    return ScopeProxy(module.__dict__ | vars(builtins))


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
