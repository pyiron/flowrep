import ast
import inspect
import types
from collections.abc import Callable

from pyiron_snippets import versions

from flowrep.models.parsers import object_scope, parser_helpers

CallDependencies = dict[versions.VersionInfo, list[Callable]]


def get_call_dependencies(
    func: types.FunctionType,
    version_scraping: versions.VersionScrapingMap | None = None,
    _call_dependencies: CallDependencies | None = None,
    _visited: set[str] | None = None,
) -> CallDependencies:
    """
    Recursively collect all callable dependencies of *func* via AST introspection.

    Each dependency is keyed by its :class:`~pyiron_snippets.versions.VersionInfo`
    and maps to the list of concrete callables sharing that identity.  The search
    is depth-first: for every resolved callee that is a
    :class:`~types.FunctionType` (i.e. has inspectable source), the function
    recurses into the callee's own scope.

    Args:
        func: The function whose call-graph to analyse.
        version_scraping (VersionScrapingMap | None): Since some modules may store
            their version in other ways, this provides an optional map between module
            names and callables to leverage for extracting that module's version.
        _call_dependencies: Accumulator for recursive calls — do not pass manually.
        _visited: Fully-qualified names already traversed — do not pass manually.

    Returns:
        A mapping from :class:`VersionInfo` to the callables found under that
        identity across the entire (sub-)tree.
    """
    call_dependencies: CallDependencies = _call_dependencies or {}
    visited: set[str] = _visited or set()

    func_fqn = versions.VersionInfo.of(func).fully_qualified_name
    if func_fqn in visited:
        return call_dependencies
    visited.add(func_fqn)

    scope = object_scope.get_scope(func)
    tree = parser_helpers.get_ast_function_node(func)
    collector = CallCollector()
    collector.visit(tree)

    for call in collector.calls:
        try:
            caller = object_scope.resolve_symbol_to_object(call, scope)
        except (ValueError, TypeError):
            continue

        if not callable(caller):
            continue

        info = versions.VersionInfo.of(caller, version_scraping=version_scraping)
        call_dependencies.setdefault(info, []).append(caller)

        # Depth-first search on dependencies — only possible when we have source
        if isinstance(caller, types.FunctionType):
            get_call_dependencies(caller, version_scraping, call_dependencies, visited)

    return call_dependencies


def split_by_version_availability(
    call_dependencies: CallDependencies,
) -> tuple[CallDependencies, CallDependencies]:
    """
    Partition *call_dependencies* by whether a version string is available.

    Args:
        call_dependencies: The dependency map to partition.

    Returns:
        A ``(has_version, no_version)`` tuple of :data:`CallDependencies` dicts.
    """
    has_version: CallDependencies = {}
    no_version: CallDependencies = {}
    for info, dependents in call_dependencies.items():
        if info.version is None:
            no_version[info] = dependents
        else:
            has_version[info] = dependents

    return has_version, no_version


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
