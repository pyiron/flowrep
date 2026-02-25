import ast
import builtins
import types
from collections.abc import Callable

from pyiron_snippets import versions

from flowrep.models.parsers import object_scope, parser_helpers

CallDependencies = dict[versions.VersionInfo, Callable]


def _get_collector(func: types.FunctionType) -> CallCollector:
    tree = parser_helpers.get_ast_function_node(func)
    collector = CallCollector()
    collector.visit(tree)
    return collector


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
    return get_dependencies(func, version_scraping, _call_dependencies, _visited)[0]

def get_dependencies(
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
    variables = []
    visited: set[str] = _visited or set()

    func_fqn = versions.VersionInfo.of(func).fully_qualified_name
    if func_fqn in visited:
        return call_dependencies, variables
    visited.add(func_fqn)

    scope = object_scope.get_scope(func)
    collector = _get_collector(func)
    items = collector.items.difference(set(dir(builtins)))

    for item in items:
        try:
            obj = object_scope.resolve_attribute_to_object(item, scope)
        except (ValueError, TypeError):
            continue

        if callable(obj):  # pragma: no cover
            info = versions.VersionInfo.of(obj, version_scraping=version_scraping)
            call_dependencies[info] = obj

            # Depth-first search on dependencies — only possible when we have source
            if isinstance(obj, types.FunctionType):
                get_call_dependencies(obj, version_scraping, call_dependencies, visited)
        else:
            variables.append(obj)

    return call_dependencies, variables


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
        self.items: set[str] = set()
        self.local_vars: set[str] = set()

    def _append_item(self, node: ast.expr) -> None:
        item = ast.unparse(node)
        if item.split(".")[0] not in self.local_vars:
            self.items.add(item.split("(")[0])

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Collect function arguments as local variables
        for arg in node.args.args:
            self.local_vars.add(arg.arg)
            # Check for type hints in arguments
            if arg.annotation and isinstance(arg.annotation, ast.Name):
                self._append_item(arg.annotation)

        # Check for type hints in the return type
        if node.returns and isinstance(node.returns, ast.Name):
            self._append_item(node.returns)

        # Visit the body of the function
        self.generic_visit(node)

        # Clear local variables after leaving the function scope
        self.local_vars.clear()

    def visit_Assign(self, node: ast.Assign) -> None:
        # Handle multiple assignments and unpacking
        for target in node.targets:
            self._process_assignment_target(target)
        self.generic_visit(node)

    def _process_assignment_target(self, target):
        # Recursively process assignment targets to handle unpacking
        if isinstance(target, ast.Attribute):
            if target.id not in self.local_vars:
                self._append_item(target)
        elif isinstance(target, ast.Name):
            # Add the variable name to local_vars
            self.local_vars.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            # Handle tuple or list unpacking (e.g., x, y = ...)
            for element in target.elts:
                self._process_assignment_target(element)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        # Handle annotated assignments (e.g., x: CustomType = 42)
        if isinstance(node.target, ast.Name):
            self.local_vars.add(node.target.id)
        if node.annotation and isinstance(node.annotation, ast.Name):
            self._append_item(node.annotation)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        # Collect all variables that are not locally defined
        if node.id not in self.local_vars:
            self._append_item(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Collect attributes that are not locally defined
        self._append_item(node)

    def visit_For(self, node: ast.For) -> None:
        # Handle loop variables as local variables
        self._process_assignment_target(node.target)
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        # Handle variables defined in with statements (e.g., with open(...) as f)
        for item in node.items:
            if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                self.local_vars.add(item.optional_vars.id)
        self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        # Handle variables defined in list comprehensions
        for generator in node.generators:
            self._process_assignment_target(generator.target)
        self.generic_visit(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        # Handle variables defined in dict comprehensions
        for generator in node.generators:
            self._process_assignment_target(generator.target)
        self.generic_visit(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        # Handle variables defined in set comprehensions
        for generator in node.generators:
            self._process_assignment_target(generator.target)
        self.generic_visit(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        # Handle variables defined in generator expressions
        for generator in node.generators:
            self._process_assignment_target(generator.target)
        self.generic_visit(node)
