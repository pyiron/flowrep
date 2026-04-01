from __future__ import annotations

import ast
import builtins
import inspect
import textwrap
from collections.abc import Callable
from typing import Any

from pyiron_snippets import versions

from flowrep.parsers import object_scope

CallDependencies = dict[versions.VersionInfo, object]


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


class UndefinedVariableVisitor(ast.NodeVisitor):
    """AST visitor that collects used and locally-defined variable names.

    Local (nested) function definitions inside the analysed function body are
    **not** supported: encountering one raises :exc:`NotImplementedError` so
    that callers fail fast with a clear message instead of silently producing
    wrong dependency results.

    Class definitions at any nesting level are tracked in :attr:`defined_vars`
    so that class names used later in the same scope are not reported as
    undefined symbols.
    """

    def __init__(self):
        self.used_vars: set[str] = set()
        self.defined_vars: set[str] = set()
        self._nesting_depth: int = 0
        self.imports: list[ast.Import] = []
        self.import_froms: list[ast.ImportFrom] = []

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.used_vars.add(node.id)
        elif isinstance(node.ctx, ast.Store):
            self.defined_vars.add(node.id)

    def _visit_function_def(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if self._nesting_depth > 0:
            keyword = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            raise NotImplementedError(
                f"Local function definitions are not supported: "
                f"'{keyword} {node.name}' inside a function body cannot be "
                "analysed for dependencies."
            )
        # Register the function name and all of its parameters so that
        # recursive calls and uses of any argument inside the body are not
        # reported as undefined external symbols.
        self.defined_vars.add(node.name)
        all_args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
        for arg in all_args:
            self.defined_vars.add(arg.arg)
        if node.args.vararg:
            self.defined_vars.add(node.args.vararg.arg)
        if node.args.kwarg:
            self.defined_vars.add(node.args.kwarg.arg)
        self._nesting_depth += 1
        self.generic_visit(node)
        self._nesting_depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function_def(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function_def(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.defined_vars.add(node.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.import_froms.append(node)

    def visit_Import(self, node: ast.Import) -> None:
        self.imports.append(node)


def find_undefined_variables(
    func_or_var: Callable[..., Any] | type[Any],
) -> dict[str, object]:
    """
    Find variables that are used but not defined in the source of *func_or_var*.

    If the source code for *func_or_var* cannot be retrieved or parsed (e.g.,
    for certain built-in objects or when no source is available), this
    function returns an empty set instead of raising an exception.
    """
    try:
        # Prefer actual source code over string representations for both
        # callables and other inspectable objects (e.g. classes, modules).
        raw_source = inspect.getsource(func_or_var)
    except (OSError, TypeError):
        # No reliable source available; treat as having no undefined variables.
        return {}

    source = textwrap.dedent(raw_source)

    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Source could not be parsed as Python code; fail gracefully.
        return {}

    visitor = UndefinedVariableVisitor()
    visitor.visit(tree)
    undefined_vars = (visitor.used_vars - visitor.defined_vars).difference(
        set(dir(builtins))
    )
    scope = object_scope.get_scope(func_or_var)
    return {
        item: object_scope.resolve_attribute_to_object(item, scope)
        for item in undefined_vars
    }


def get_call_dependencies(
    func_or_var: Callable[..., Any] | type[Any],
    version_scraping: versions.VersionScrapingMap | None = None,
    _call_dependencies: CallDependencies | None = None,
    _visited: set[str] | None = None,
) -> CallDependencies:

    call_dependencies: CallDependencies = _call_dependencies or {}
    visited: set[str] = _visited or set()

    func_fqn = versions.VersionInfo.of(func_or_var).fully_qualified_name
    if func_fqn in visited:
        return call_dependencies
    visited.add(func_fqn)

    # Find variables that are used but not defined
    for obj in find_undefined_variables(func_or_var).values():
        info = versions.VersionInfo.of(obj, version_scraping=version_scraping)
        call_dependencies[info] = obj

        if callable(obj) or isinstance(obj, type):
            if info.version is None:
                get_call_dependencies(obj, version_scraping, call_dependencies, visited)
    return call_dependencies
