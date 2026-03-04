import ast
import types
from collections.abc import Callable

from pyiron_snippets import versions

from flowrep.models.parsers import import_parser, object_scope, parser_helpers

CallDependencies = dict[versions.VersionInfo, Callable]


def get_call_dependencies(
    func: types.FunctionType,
    version_scraping: versions.VersionScrapingMap | None = None,
    _call_dependencies: CallDependencies | None = None,
    _visited: set[str] | None = None,
) -> CallDependencies:
    """
    Recursively collect all callable dependencies of *func* via AST introspection.

    Each dependency is keyed by its :class:`~pyiron_snippets.versions.VersionInfo`
    and maps to the callables instance with that identity.  The search is depth-first:
    for every resolved callee that is a :class:`~types.FunctionType` (i.e. has
    inspectable source), the function recurses into the callee's own scope.

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

    tree = parser_helpers.get_ast_function_node(func)
    collector = CallCollector()
    collector.visit(tree)
    local_modules = import_parser.build_scope(collector.imports, collector.import_froms)
    scope = object_scope.get_scope(func, extra_modules=local_modules)

    for call in collector.calls:
        try:
            caller = object_scope.resolve_symbol_to_object(call, scope)
        except (ValueError, TypeError):
            continue

        if not callable(caller):  # pragma: no cover
            # Under remotely normal circumstances, this should be unreachable
            raise TypeError(
                f"Caller {caller} is not callable, yet was generated from the list of "
                f"ast.Call calls, in particular {call}. We're expecting these to "
                f"actually connect to callables. Please raise a GitHub issue if you "
                f"think this is not a mistake."
            )

        info = versions.VersionInfo.of(caller, version_scraping=version_scraping)
        # In principle, we open ourselves to overwriting an existing dependency here,
        # but it would need to somehow have exactly the same version info (including
        # qualname) yet be a different object.
        # This ought not happen by accident, and in case it somehow does happen on
        # purpose (it probably shouldn't), we just silently keep the more recent one.

        call_dependencies[info] = caller

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
    for info, dependency in call_dependencies.items():
        if info.version is None:
            no_version[info] = dependency
        else:
            has_version[info] = dependency

    return has_version, no_version


class CallCollector(ast.NodeVisitor):
    def __init__(self):
        self.calls: list[ast.expr] = []
        self.imports: list[ast.Import] = []
        self.import_froms: list[ast.ImportFrom] = []

    def visit_Call(self, node: ast.Call) -> None:
        self.calls.append(node.func)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        self.imports.append(node)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.import_froms.append(node)
        self.generic_visit(node)
