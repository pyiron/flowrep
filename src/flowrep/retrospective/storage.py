"""
Convenience tools for accessing :cls:`flowrep.datastructures.LiveWorkflow` data stored in
*bagofholding* `H5Bag` objects using "lexical" paths (node names, "inputs"/"outputs",
and port names). The data may be the bag's top-level object, or be kept somewhere inside
it, e.g. as an attribute of some other saved object.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

from packaging import version
from pyiron_snippets import import_alarm

from flowrep import base_models, lexical
from flowrep.retrospective import datastructures, storage_widget

with import_alarm.ImportAlarm(
    "This tool requires the 'bagofholding' package.", raise_exception=True
) as _import_alarm:
    import bagofholding as boh


if TYPE_CHECKING:
    from bagofholding import H5Bag


DEFAULT_STORAGE_ROOT = "object"
"""Where a bag made by saving node data directly keeps that data."""

_NODE_DATA_TYPES = (
    datastructures.AtomicData,
    datastructures.ConstantData,
    datastructures.DagData,
    datastructures.ForEachData,
    datastructures.IfData,
    datastructures.TryData,
    datastructures.WhileData,
)
_LOADABLE_TYPES = (
    *_NODE_DATA_TYPES,
    datastructures.InputDataPort,
    datastructures.OutputDataPort,
)


class LexicalBagBrowser:
    """
    A convenience class for browsing and loading data from
    :cls:`LiveWorkflow` objects serialized in a *bagofholding* :cls:`H5Bag`.

    Lets you access data using the "lexical" paths (i.e. "."-joined paths of node names,
    "inputs/outputs", and port names) instead of the actual H5 path inside the file.
    Node data kept inside some other saved object can be browsed by passing the storage
    path to it as `storage_root`.
    """

    @_import_alarm
    def __init__(
        self,
        bag: H5Bag | str | pathlib.Path,
        storage_root: str = DEFAULT_STORAGE_ROOT,
    ):
        if isinstance(bag, (str, pathlib.Path)):
            self.bag = boh.H5Bag(bag)
        else:
            self.bag = bag
        self.storage_root = _normalize_root(storage_root)
        validate_bag(self.bag, self.storage_root)

    def list_paths(self) -> list[str]:
        """A list of all available lexical content paths."""
        return list_lexical_paths(self.bag, self.storage_root)

    def widget(self) -> storage_widget.LexicalBagTree:
        """A jupyter-notebook widget for graphical browsing"""
        return storage_widget.LexicalBagTree(self)

    def browse(self) -> storage_widget.LexicalBagTree | list[str]:
        """Look at (but don't load and instantiate) the available content."""
        try:
            return self.widget()
        except ImportError:
            return self.list_paths()

    def load(
        self, path: str
    ) -> (
        datastructures.AtomicData
        | datastructures.ConstantData
        | datastructures.DagData
        | datastructures.ForEachData
        | datastructures.IfData
        | datastructures.InputDataPort
        | datastructures.OutputDataPort
        | datastructures.TryData
        | datastructures.WhileData
    ):
        """Load a node or IO port using its lexical path."""
        return load_from_bag(self.bag, path, self.storage_root)


@_import_alarm
def validate_bag(bag: H5Bag, storage_root: str = DEFAULT_STORAGE_ROOT):
    if not isinstance(bag, boh.H5Bag):
        raise TypeError(f"Expected a {boh.H5Bag.__name__!r} object, got {bag!r}")

    _validate_bag_metadata(bag)
    _validate_object_metadata(bag, storage_root)


def _normalize_root(storage_root: str) -> str:
    return storage_root.rstrip("/")


def _validate_bag_metadata(bag: H5Bag):
    bag_info = bag.get_bag_info()

    DEV_VERSION = "0.0.0+unknown"
    VERSION_MIN = version.Version("0.1.5")
    VERSION_MAX = version.Version("0.2.0")

    if bag_info.version == DEV_VERSION:
        return

    try:
        v = version.Version(bag_info.version)
    except version.InvalidVersion as e:
        raise ValueError(f"Unparseable bag version {bag_info.version!r}") from e

    if not (VERSION_MIN <= v < VERSION_MAX):
        raise ValueError(
            f"Bag version {bag_info.version!r} must be >={VERSION_MIN}, <{VERSION_MAX}"
        )


def _validate_object_metadata(bag: H5Bag, storage_root: str = DEFAULT_STORAGE_ROOT):
    root = _normalize_root(storage_root)
    try:
        object_info = bag[root]
    except boh.exceptions.InvalidMetadataError:
        raise ValueError(f"Nothing is stored at {root!r}") from None
    qualnames = tuple(cls.__qualname__ for cls in _NODE_DATA_TYPES)
    if object_info.qualname not in qualnames:
        raise TypeError(
            f"Can only load saved node data (one of {qualnames}), but got "
            f"{object_info.qualname!r} at {root!r}"
        )


def list_lexical_paths(
    bag: boh.H5Bag, storage_root: str = DEFAULT_STORAGE_ROOT
) -> list[str]:
    """
    Look through the bag and return a list of "."-separated lexical paths for nodes and
    ports, starting from the node data stored at *storage_root*.
    """
    paths: list[str] = []
    _collect_lexical_paths(bag, _normalize_root(storage_root), "", paths)
    return paths


def _collect_lexical_paths(
    bag: H5Bag,
    storage_path: str,
    prefix: str,
    paths: list[str],
) -> None:
    for io_type in base_models.IOTypes:
        io_storage = (
            _path_to_input_ports(storage_path)
            if io_type == base_models.IOTypes.INPUTS
            else _path_to_output_ports(storage_path)
        )
        port_names = bag.open_group(io_storage)
        for port in port_names:
            paths.append(
                lexical.port_path(prefix.rstrip(lexical.DELIMITER), io_type, port)
            )

    nodes_storage = _path_to_nodes(storage_path)
    try:
        node_names = bag.open_group(nodes_storage)
    except KeyError:
        return
    for node in node_names:
        node_path = f"{prefix}{node}"
        paths.append(node_path)
        _collect_lexical_paths(bag, f"{nodes_storage}/{node}", f"{node_path}.", paths)


def _path_to_input_ports(path: str) -> str:
    return f"{path}/state/input_ports"


def _path_to_output_ports(path: str) -> str:
    return f"{path}/state/output_ports"


def _path_to_nodes(path: str) -> str:
    return f"{path}/state/nodes"


def load_from_bag(
    bag: H5Bag, lexical_path: str, storage_root: str = DEFAULT_STORAGE_ROOT
) -> (
    datastructures.AtomicData
    | datastructures.ConstantData
    | datastructures.DagData
    | datastructures.ForEachData
    | datastructures.IfData
    | datastructures.InputDataPort
    | datastructures.OutputDataPort
    | datastructures.TryData
    | datastructures.WhileData
):
    """
    Load data from a :cls:`LiveNode` stored in a *bagofholding* by using its lexical
    path.

    Args:
        bag (H5Bag): The bag containing the saved node data.
        lexical_path (str): The dot-separated path of node names, IO references, and/or
            port names.
        storage_root (str): Where in the bag the node data is kept. Defaults to the
            top-level object.

    Returns:
        A retrospective data node or IO data port
    """
    storage_path = _normalize_root(storage_root)
    step = ""
    walked_path = step
    while lexical_path:
        last_step = step
        step, _, lexical_path = lexical_path.partition(".")
        walked_path += f".{step}"
        try:
            storage_path = _extend_path(bag, storage_path, step, last_step)
        except _CannotFindLocationError as e:
            raise ValueError(
                f"Could not find {step!r} at {walked_path.lstrip('.')!r}"
            ) from e

    obj = bag.load(storage_path)

    if step in ("inputs", "outputs"):
        raise ValueError(
            f"Path terminated in {step!r}. Please select an individual port to load "
            f"from among {tuple(obj.keys())}"
        )

    if not isinstance(obj, _LOADABLE_TYPES):
        raise TypeError(
            f"Expected to load one of {tuple(cls.__name__ for cls in _LOADABLE_TYPES)}, "
            f"but got {type(obj).__name__}: {obj!r}"
        )
    return obj


class _CannotFindLocationError(ValueError): ...


def _extend_path(bag: H5Bag, storage_path: str, step: str, last_step: str) -> str:
    extended_path: str
    if last_step in tuple(base_models.IOTypes):
        parent, child = storage_path, step
    elif step == base_models.IOTypes.INPUTS:
        parent, child = _path_to_input_ports(storage_path).rsplit("/", maxsplit=1)
    elif step == base_models.IOTypes.OUTPUTS:
        parent, child = _path_to_output_ports(storage_path).rsplit("/", maxsplit=1)
    else:
        parent, child = _path_to_nodes(storage_path), step
    extended_path = f"{parent}/{child}"

    try:
        children = bag.open_group(parent)
    except KeyError:
        raise _CannotFindLocationError(extended_path) from None

    if child not in children:
        raise _CannotFindLocationError(extended_path)

    return extended_path
