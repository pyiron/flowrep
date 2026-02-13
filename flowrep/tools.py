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


def hash_function(fn: Callable) -> str:
    """
    Hash a function based on its source code or signature.
    
    For regular functions, the hash is based on the dedented source code.
    For other callables (built-ins, methods, etc.), the hash is based on
    the module, qualified name, and signature.

    Args:
        fn (Callable): The function to hash.

    Returns:
        str: A stable hash string in the format "function_name:hash_hex".
        
    Raises:
        OSError: If source code cannot be retrieved for a function.
        TypeError: If the callable doesn't have the required attributes.
    """

    if inspect.isfunction(fn):
        try:
            source_code = inspect.getsource(fn)
        except (OSError, TypeError):
            # Fall back to signature for functions where source is unavailable
            source_code = f"{fn.__module__}:{fn.__qualname__}:{inspect.signature(fn)}"
    else:
        source_code = f"{fn.__module__}:{fn.__qualname__}:{inspect.signature(fn)}"
    source_code = textwrap.dedent(source_code.replace("\r\n", "\n"))
    source_code_hash = hashlib.sha256(source_code.encode("utf-8")).hexdigest()
    name = getattr(fn, "__name__", "unknown")
    return name + ":" + source_code_hash


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
