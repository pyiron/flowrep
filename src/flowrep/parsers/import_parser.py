import ast
import importlib

from flowrep.parsers import object_scope
from flowrep.parsers.object_scope import ScopeProxy


def build_scope(
    imports: list[ast.Import] | None = None,
    import_froms: list[ast.ImportFrom] | None = None,
) -> object_scope.ScopeProxy:
    """
    Build a scope dictionary from a list of `import` and `from ... import ...` statements.

    Args:
        imports (list | None): A list of `ast.Import` nodes.
        import_froms (list | None): A list of `ast.ImportFrom` nodes.

    Returns:
        object_scope.ScopeProxy: A mutable mapping representing the scope with imported
            modules and objects.
    """
    scope = ScopeProxy()

    imports = imports or []
    import_froms = import_froms or []

    # Handle `import` statements
    for imp in imports:
        for alias in imp.names:
            asname = alias.asname or alias.name
            module = importlib.import_module(alias.name)
            scope.register(name=asname, obj=module)

    # Handle `from ... import ...` statements
    for imp_from in import_froms:
        level = imp_from.level
        # Dynamically import the module (absolute or relative)
        if imp_from.module is None or level > 0:
            raise ValueError(
                f"Relative imports are not supported in dependency parsing. "
                f"Encountered importing from {imp_from.module}."
            )
        module = importlib.import_module(imp_from.module)
        for alias in imp_from.names:
            name = alias.name
            asname = alias.asname or name
            obj = getattr(module, name)
            scope.register(name=asname, obj=obj)

    return scope
