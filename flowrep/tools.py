import copy
import hashlib
import json
from collections import defaultdict
from collections.abc import Callable
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
    from importlib import import_module

    base_module = import_module(data["module"].split(".")[0])
    data["version"] = (
        base_module.__version__
        if hasattr(base_module, "__version__")
        else "not_defined"
    )
    if not full_metadata:
        return data
    data["hash"] = hash_function(cls)
    data["docstring"] = cls.__doc__ or ""
    data["name"] = cls.__name__
    return data


def hash_function(fn: Callable) -> str:
    """
    Generate a SHA-256 hash for a given function based on its bytecode and
    metadata.

    Args:
        fn (Callable): The function to be hashed.

    Returns:
        str: A SHA-256 hash of the function's bytecode and metadata.
    """
    h = hashlib.sha256()

    code = fn.__code__

    # include bytecode
    h.update(code.co_code)

    # include metadata
    fields_dict = {
        "co_argcount": code.co_argcount,
        "co_posonlyargcount": code.co_posonlyargcount,
        "co_kwonlyargcount": code.co_kwonlyargcount,
        "co_nlocals": code.co_nlocals,
        "co_stacksize": code.co_stacksize,
        "co_flags": code.co_flags,
        "co_consts": code.co_consts,
        "co_names": code.co_names,
        "co_varnames": code.co_varnames,
        "co_freevars": code.co_freevars,
        "co_cellvars": code.co_cellvars,
        "defaults": fn.__defaults__,
        "kwdefaults": fn.__kwdefaults__,
    }

    h.update(json.dumps(fields_dict, sort_keys=True, default=str).encode("utf-8"))

    return h.hexdigest()


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
