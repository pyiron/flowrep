import importlib


def build_scope(imports: list | None = None, import_froms: list | None = None) -> dict:
    """
    Build a scope dictionary from a list of `import` and `from ... import ...` statements.

    Args:
        imports (list | None): A list of `ast.Import` nodes.
        import_froms (list | None): A list of `ast.ImportFrom` nodes.

    Returns:
        dict: A dictionary representing the scope with imported modules and objects.
    """
    scope = {}

    imports = imports or []
    import_froms = import_froms or []

    # Handle `import` statements
    for imp in imports:
        for alias in imp.names:
            module_name = alias.name
            asname = alias.asname or module_name
            try:
                # Dynamically import the module
                module = importlib.import_module(module_name)
                scope[asname] = module
            except ImportError:
                print(f"Warning: Could not import module '{module_name}'")

    # Handle `from ... import ...` statements
    for imp_from in import_froms:
        module_name = imp_from.module
        level = imp_from.level
        try:
            # Dynamically import the module (absolute or relative)
            module = importlib.import_module(
                module_name, package=None if level == 0 else "."
            )
            for alias in imp_from.names:
                name = alias.name
                asname = alias.asname or name
                try:
                    # Get the specific object from the module
                    obj = getattr(module, name)
                    scope[asname] = obj
                except AttributeError:
                    print(f"Warning: Module '{module_name}' has no attribute '{name}'")
        except ImportError:
            print(f"Warning: Could not import module '{module_name}'")

    return scope
