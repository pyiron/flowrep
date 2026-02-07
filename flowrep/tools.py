import ast
import copy
import hashlib
import inspect
import json
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


def _function_to_ast_dict(node):
    if isinstance(node, ast.AST):
        result = {"_type": type(node).__name__}
        for field, value in ast.iter_fields(node):
            result[field] = _function_to_ast_dict(value)
        return result
    elif isinstance(node, list):
        return [_function_to_ast_dict(item) for item in node]
    else:
        return node


def get_ast_dict(func: Callable) -> dict:
    """Get the AST dictionary representation of a function."""
    source_code = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source_code)
    return _function_to_ast_dict(tree)


def hash_function(fn: Callable) -> str:
    """
    Returns a stable hash for a Python callable.

    - Uses AST-based semantic hashing when source is available
    - Falls back to identity hashing (module + qualname) otherwise

    Args:
        fn (Callable): The function to hash.

    Returns:
        str: A stable hash string representing the function.
    """

    # ---- Primary path: semantic hash ----
    try:
        payload = json.dumps(
            get_ast_dict(fn), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return "ast:" + hashlib.sha256(payload).hexdigest()
    except (OSError, TypeError):
        pass

    # ---- Fallback path: identity hash ----
    if hasattr(fn, "__module__") and hasattr(fn, "__qualname__"):
        version = _get_version_from_module(fn.__module__)
        identity = f"{fn.__module__}:{fn.__qualname__}:{version}"
        return "id:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()

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
