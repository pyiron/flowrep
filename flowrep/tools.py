import ast
import copy
import hashlib
import inspect
import textwrap
from collections import defaultdict
from collections.abc import Callable
from importlib import import_module
from typing import Any


def serialize_functions(data: dict[str, Any]) -> dict[str, Any]:
    """
    Separate functions from the data dictionary and store them in a function
    dictionary. The functions inside the data dictionary will be replaced by
    their name (which would for example make it easier to hash it)

    Args:
        data (dict[str, Any]): The data dictionary containing nodes and
            functions.

    Returns:
        tuple: A tuple containing the modified data dictionary and the
            function dictionary.
    """
    data = copy.deepcopy(data)
    if "nodes" in data:
        for key, node in data["nodes"].items():
            data["nodes"][key] = serialize_functions(node)
    elif "function" in data and not isinstance(data["function"], str):
        data["function"] = get_function_metadata(data["function"])
    if "test" in data and not isinstance(data["test"]["function"], str):
        data["test"]["function"] = get_function_metadata(data["test"]["function"])
    return data


def _get_version_from_module(module_name: str) -> str:
    base_module_name = module_name.split(".")[0]
    base_module = import_module(base_module_name)
    return getattr(base_module, "__version__", "not_defined")


def get_function_metadata(
    cls: Callable | dict[str, str], full_metadata: bool = False
) -> dict[str, str]:
    """
    Get metadata for a given function or class.

    Args:
        cls (Callable | dict[str, str]): The function or class to get metadata for.
        full_metadata (bool): Whether to include full metadata including hash,
            docstring, and name.

    Returns:
        dict[str, str]: A dictionary containing the metadata of the function or class.
    """
    if isinstance(cls, dict) and "module" in cls and "qualname" in cls:
        return cls
    data = {
        "module": cls.__module__,
        "qualname": cls.__qualname__,
    }

    data["version"] = _get_version_from_module(data["module"])
    if not full_metadata:
        return data
    data["hash"] = hash_function(cls)
    data["docstring"] = cls.__doc__ or ""
    data["name"] = cls.__name__
    return data


class RemoveDocstrings(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body = node.body[1:]
        return node

    def visit_AsyncFunctionDef(self, node):
        return self.visit_FunctionDef(node)


class CanonicalASTDumper(ast.NodeVisitor):
    def __init__(self):
        self.parts = []

    def generic_visit(self, node):
        self.parts.append(type(node).__name__)
        for field, value in ast.iter_fields(node):
            self.parts.append(field)
            self._visit_value(value)

    def _visit_value(self, value):
        if isinstance(value, ast.AST):
            self.visit(value)
        elif isinstance(value, list):
            self.parts.append("[")
            for item in value:
                self._visit_value(item)
            self.parts.append("]")
        elif isinstance(value, (str, int, float, complex, bool, type(None))):
            self.parts.append(repr(value))
        else:
            raise TypeError(f"Unsupported AST value: {type(value)!r}")


def _hash_function_with_ast(fn: Callable) -> str:
    """
    Hash a function based on its AST structure, ignoring docstrings and formatting.

    Args:
        fn (Callable): The function to hash.

    Returns:
        str: A hash string representing the function's AST structure.
    """
    name = getattr(fn, "__name__", "unkonwn")
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        source = None

    if source is not None:
        source = textwrap.dedent(source)
        tree = ast.parse(source)

        tree = RemoveDocstrings().visit(tree)
        ast.fix_missing_locations(tree)

        dumper = CanonicalASTDumper()
        dumper.visit(tree)

        payload = "\n".join(dumper.parts).encode("utf-8")
        return name + ":" + hashlib.sha256(payload).hexdigest()
    raise ValueError("Source code not available for AST hashing")


def hash_function(fn: Callable) -> str:
    """
    Hash a function based on its module, qualified name, and version. If the
    function does not have a module or qualified name, it raises a TypeError.

    Args:
        fn (Callable): The function to hash.

    Returns:
        str: A stable hash string representing the function.
    """

    name = getattr(fn, "__name__", "unkonwn")
    if hasattr(fn, "__module__") and hasattr(fn, "__qualname__"):
        version = _get_version_from_module(fn.__module__)
        identity = f"{fn.__module__}:{fn.__qualname__}:{version}"
        return name + ":" + hashlib.sha256(identity.encode("utf-8")).hexdigest()

    raise TypeError(f"{fn!r} is not hashable - wrap it in another function")


def recursive_defaultdict() -> defaultdict:
    """
    Create a recursively nested defaultdict.

    Example:
    >>> d = recursive_defaultdict()
    >>> d['a']['b']['c'] = 1
    >>> print(d)

    Output: 1
    """
    return defaultdict(recursive_defaultdict)


def dict_to_recursive_dd(d: dict | defaultdict) -> defaultdict:
    """Convert a regular dict to a recursively nested defaultdict."""
    if isinstance(d, dict) and not isinstance(d, defaultdict):
        return defaultdict(
            recursive_defaultdict, {k: dict_to_recursive_dd(v) for k, v in d.items()}
        )
    return d


def recursive_dd_to_dict(d: dict | defaultdict) -> dict:
    """Convert a recursively nested defaultdict to a regular dict."""
    if isinstance(d, defaultdict):
        return {k: recursive_dd_to_dict(v) for k, v in d.items()}
    return d
