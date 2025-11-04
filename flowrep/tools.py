import copy
import hashlib
import json
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import networkx as nx


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


def get_function_metadata(cls: Callable | dict[str, str]) -> dict[str, str]:
    if isinstance(cls, dict) and "module" in cls and "qualname" in cls:
        return cls
    module = cls.__module__
    qualname = cls.__qualname__
    from importlib import import_module

    base_module = import_module(module.split(".")[0])
    version = (
        base_module.__version__
        if hasattr(base_module, "__version__")
        else "not_defined"
    )
    return {
        "module": module,
        "qualname": qualname,
        "version": version,
    }


def get_hashed_node_dict(
    node: str, graph: nx.DiGraph, nodes_dict: dict[str, dict]
) -> dict[str, Any]:
    """
    Get a dictionary representation of a node for hashing purposes and database
    entries. This function extracts the metadata of the node, its inputs, and
    outputs, and returns a dictionary that can be hashed.

    Args:
        node (str): The name of the node to be hashed.
        graph (nx.DiGraph): The directed graph representing the function.
        nodes_dict (dict[str, dict]): A dictionary containing metadata for all nodes.

    Returns:
        dict[str, Any]: A dictionary representation of the node for hashing.

    Raises:
        ValueError: If the node does not have a function or if the data is not flat.
    """
    if "function" not in nodes_dict[node]:
        raise ValueError("Hashing works only on flat data")
    data_dict = {
        "node": get_function_metadata(nodes_dict[node]["function"]),
        "inputs": {},
        "outputs": [tag.split(".")[-1] for tag in graph.successors(node)],
    }
    connected_inputs = []
    for tag in graph.predecessors(node):
        assert "inputs" in tag
        key = tag.split(".")[-1]
        predecessor = list(graph.predecessors(tag))
        assert len(predecessor) == 1
        predecessor = predecessor[0]
        pre_predecessor = list(graph.predecessors(predecessor))
        if len(pre_predecessor) > 0:
            assert len(pre_predecessor) == 1
            value = (
                get_node_hash(pre_predecessor[0], graph, nodes_dict)
                + "@"
                + predecessor.split(".")[-1]
            )
            connected_inputs.append(key)
        else:
            value = predecessor
        data_dict["inputs"][key] = value
    data_dict["node"]["connected_inputs"] = connected_inputs
    return data_dict


def get_node_hash(node: str, graph: nx.DiGraph, nodes_dict: dict[str, dict]) -> str:
    """
    Get a hash of the node's metadata, inputs, and outputs.

    Args:
        node (str): The name of the node to be hashed.
        graph (nx.DiGraph): The directed graph representing the function.
        nodes_dict (dict[str, dict]): A dictionary containing metadata for all nodes.

    Returns:
        str: A SHA-256 hash of the node's metadata, inputs, and outputs.
    """
    data_dict = get_hashed_node_dict(node=node, graph=graph, nodes_dict=nodes_dict)
    return hashlib.sha256(
        json.dumps(data_dict, sort_keys=True).encode("utf-8")
    ).hexdigest()


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
