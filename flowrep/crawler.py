import ast
import inspect
import sys
import types
from typing import Any

from pyiron_snippets import versions


def function_id(func) -> versions.VersionInfo:
    return versions.VersionInfo.of(func)


class CallCollector(ast.NodeVisitor):
    def __init__(self):
        self.calls = []

    def visit_Call(self, node):
        self.calls.append(node.func)
        self.generic_visit(node)


def build_global_namespace(func) -> dict[str, object]:
    namespace = dict(func.__globals__)

    if func.__closure__:
        freevars = func.__code__.co_freevars
        for var, cell in zip(freevars, func.__closure__):
            namespace[var] = cell.cell_contents

    return namespace


def resolve_ast_node(node: ast.AST, namespace: dict[str, object]) -> Any:
    """
    Resolve an AST node to its corresponding object in the given namespace.

    Args:
        node (ast.AST): The AST node to resolve.
        namespace (dict[str, object]): The namespace to use for resolution.

    Returns:
        Any: The resolved object, or None if it cannot be resolved.
    """
    if isinstance(node, ast.Name):
        return namespace.get(node.id)

    if isinstance(node, ast.Attribute):
        base = resolve_ast_node(node.value, namespace)
        if base is None:
            return None
        return getattr(base, node.attr, None)

    return None


def extract_called_functions(func: types.FunctionType) -> set[types.FunctionType]:
    """
    Extract all functions called by the given function.

    Args:
        func (types.FunctionType): The function to analyze.

    Returns:
        Set[types.FunctionType]: A set of functions that are called by the given function.
    """
    source = inspect.getsource(func)

    tree = ast.parse(source)
    collector = CallCollector()
    collector.visit(tree)

    namespace = build_global_namespace(func)
    resolved = set()

    for call_node in collector.calls:
        obj = resolve_ast_node(call_node, namespace)
        if callable(obj):
            resolved.add(obj)

    return resolved


def analyze_function_dependencies(root_func: types.FunctionType) -> tuple[
    set[types.FunctionType],  # local functions
    set[FunctionID],  # non-local functions
]:
    """
    Recursively analyze function dependencies.

    Args:
        root_func (types.FunctionType): The root function to analyze.

    Returns:
        Tuple[Set[types.FunctionType], Set[FunctionID]]: A tuple containing:
            - A set of local functions (defined in the same codebase).
            - A set of external function IDs (from other modules or libraries).
    """
    visited: set[FunctionID] = set()
    local_functions: set[types.FunctionType] = set()
    external_functions: set[FunctionID] = set()

    def walk(func):
        fid = function_id(func)
        if fid.fully_qualified_name in visited:
            return
        visited.add(fid.fully_qualified_name)

        for called in extract_called_functions(func):
            cid = function_id(called)

            if cid.version is None:
                local_functions.add(called)
                walk(called)
            else:
                external_functions.add(cid)

    walk(root_func)
    return local_functions, external_functions
