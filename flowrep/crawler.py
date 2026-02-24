import ast
import inspect
import types

from pyiron_snippets import versions

from flowrep.models.parsers import object_scope


class CallCollector(ast.NodeVisitor):
    def __init__(self):
        self.calls: list[ast.expr] = []

    def visit_Call(self, node: ast.Call) -> None:
        self.calls.append(node.func)
        self.generic_visit(node)


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

    namespace = object_scope.get_scope(func)
    resolved = set()

    for call_node in collector.calls:
        obj = object_scope.resolve_symbol_to_object(call_node, namespace)
        if callable(obj):
            resolved.add(obj)

    return resolved


def analyze_function_dependencies(root_func: types.FunctionType) -> tuple[
    set[types.FunctionType],  # local functions
    set[versions.VersionInfo],  # non-local functions
]:
    """
    Recursively analyze function dependencies.

    Args:
        root_func (types.FunctionType): The root function to analyze.

    Returns:
        Tuple[Set[types.FunctionType], Set[VersionInfo]]: A tuple containing:
            - A set of local functions (defined in the same codebase).
            - A set of external function IDs (from other modules or libraries).
    """
    visited: set[versions.VersionInfo] = set()
    local_functions: set[types.FunctionType] = set()
    external_functions: set[versions.VersionInfo] = set()

    def walk(func):
        fid = versions.VersionInfo.of(func)
        if fid.fully_qualified_name in visited:
            return
        visited.add(fid.fully_qualified_name)

        for called in extract_called_functions(func):
            cid = versions.VersionInfo.of(called)

            if cid.version is None:
                local_functions.add(called)
                walk(called)
            else:
                external_functions.add(cid)

    walk(root_func)
    return local_functions, external_functions
