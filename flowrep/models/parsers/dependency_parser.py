from __future__ import annotations

import ast
import builtins
import inspect
import textwrap
import types
from collections.abc import Callable

from pyiron_snippets import versions

from flowrep.models.parsers import object_scope, parser_helpers

CallDependencies = dict[versions.VersionInfo, Callable]


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
    def __init__(self):
        self.used_vars = set()
        self.defined_vars = set()

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):  # Variable is being used
            self.used_vars.add(node.id)
        elif isinstance(node.ctx, ast.Store):  # Variable is being defined
            self.defined_vars.add(node.id)

    def visit_FunctionDef(self, node):
        # Add function arguments to defined variables
        for arg in node.args.args:
            self.defined_vars.add(arg.arg)
        self.generic_visit(node)


def find_undefined_variables(
    func_or_var: Callable | object,
) -> set[str]:
    if callable(func_or_var):
        source = textwrap.dedent(inspect.getsource(func_or_var))
    else:
        source = str(func_or_var)
    tree = ast.parse(source)

    visitor = UndefinedVariableVisitor()
    visitor.visit(tree)
    undefined_vars = visitor.used_vars - visitor.defined_vars
    return undefined_vars.difference(set(dir(builtins)))


def get_call_dependencies(
    func_or_var: Callable | object,
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
    scope = object_scope.get_scope(func_or_var)
    for item in find_undefined_variables(func_or_var):
        try:
            obj = object_scope.resolve_attribute_to_object(item, scope)
        except (ValueError, TypeError):
            continue
        info = versions.VersionInfo.of(obj, version_scraping=version_scraping)
        call_dependencies[info] = obj

        get_call_dependencies(obj, version_scraping, call_dependencies, visited)
    return call_dependencies
