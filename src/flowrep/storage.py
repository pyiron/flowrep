"""
Convenience tools for accessing :cls:`flowrep.live.LiveWorkflow` data stored in
*bagofholding* `H5Bag` objects using "lexical" paths (node names, "inputs"/"outputs",
and port names).
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

from pyiron_snippets import import_alarm

from flowrep import base_models, live, widget

with import_alarm.ImportAlarm(
    "This tool requires the 'bagofholding' package."
) as _import_alarm:
    import bagofholding as boh


if TYPE_CHECKING:
    from bagofholding import H5Bag


@_import_alarm
class LexicalBagBrowser:
    """
    A convenience class for browsing and loading data from
    :cls:`LiveWorkflow` objects serialized in a *bagofholding* :cls:`H5Bag`.

    Lets you access data using the "lexical" paths (i.e. "."-joined paths of node names,
    "inputs/outputs", and port names) instead of the actual H5 path inside the file.
    """

    def __init__(self, bag: H5Bag | str | pathlib.Path):
        if isinstance(bag, (str, pathlib.Path)):
            self.bag = boh.H5Bag(bag)
        else:
            self.bag = bag
        validate_bag(self.bag)

    def list_paths(self) -> list[str]:
        """A list of all available lexical content paths."""
        return list_lexical_paths(self.bag)

    def widget(self):
        """A jupyter-notebook widget for graphical browsing"""
        return widget.LexicalBagTree(self)

    def browse(self):
        """Look at (but don't load and instantiate) the available content."""
        try:
            return self.widget()
        except ImportError:
            return self.list_paths()

    def load(
        self, path: str
    ) -> (
        live.LiveAtomic
        | live.LiveWorkflow
        | live.FlowControl
        | live.InputPort
        | live.OutputPort
    ):
        """Load a node or IO port using its lexical path."""
        return load_from_bag(self.bag, path)


@_import_alarm
def validate_bag(bag: H5Bag):
    if not isinstance(bag, boh.H5Bag):
        raise TypeError(f"Expected a {boh.H5Bag.__name__!r} object, got {bag!r}")

    _validate_bag_metadata(bag)
    _validate_object_metadata(bag)


def _validate_bag_metadata(bag: H5Bag):
    bag_info = bag.get_bag_info()

    valid_versions = {"0.1.5"}
    if bag_info.version not in valid_versions:
        raise ValueError(
            f"{boh.__name__!r} version must be among {valid_versions!r}, but got {bag_info.version!r}"
        )


def _validate_object_metadata(bag: H5Bag):
    object_info = bag["object"]
    if object_info.qualname != live.LiveWorkflow.__qualname__:
        raise TypeError(
            f"Can only load saved live workflows ({live.LiveWorkflow.__qualname__!r} "
            f"type), but got {object_info.qualname!r}"
        )


def list_lexical_paths(bag: boh.H5Bag):
    """
    Look through the bag and return a list of "."-separated lexical paths for nodes and
    ports.
    """
    paths: list[str] = []
    _collect_lexical_paths(bag, "object/", "", paths)
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
        try:
            port_names = bag.open_group(io_storage)
        except KeyError:
            continue
        for port in port_names:
            paths.append(f"{prefix}{io_type}.{port}")

    nodes_storage = _path_to_nodes(storage_path)
    try:
        node_names = bag.open_group(nodes_storage)
    except KeyError:
        return
    for node in node_names:
        lexical = f"{prefix}{node}"
        paths.append(lexical)
        _collect_lexical_paths(bag, f"{nodes_storage}/{node}", f"{lexical}.", paths)


def _path_to_input_ports(path: str):
    return f"{path}/state/input_ports"


def _path_to_output_ports(path: str):
    return f"{path}/state/output_ports"


def _path_to_nodes(path: str):
    return f"{path}/state/nodes"


def load_from_bag(
    bag: H5Bag, lexical_path: str
) -> (
    live.LiveAtomic
    | live.LiveWorkflow
    | live.FlowControl
    | live.InputPort
    | live.OutputPort
):
    """
    Load data from a :cls:`LiveNode` stored in a *bagofholding* by using its lexical
    path.

    Args:
        bag (H5Bag): The bag containing the saved live node.
        lexical_path (str): The dot-separated path of node names, IO references, and/or
            port names.

    Returns:
        A live node or live IO port
    """
    storage_path = "object/"
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

    expected_types = (
        live.LiveAtomic,
        live.LiveWorkflow,
        live.FlowControl,
        live.InputPort,
        live.OutputPort,
    )
    if not isinstance(obj, expected_types):
        raise TypeError(
            f"Expected to load one of {(cls.__name__ for cls in expected_types)}, but "
            f"got {obj}"
        )
    return obj


class _CannotFindLocationError(ValueError): ...


def _extend_path(bag: H5Bag, storage_path: str, step: str, last_step: str) -> str:
    extended_path: str
    if last_step in base_models.IOTypes:
        extended_path = f"{storage_path}/{step}"
    elif step == base_models.IOTypes.INPUTS:
        extended_path = _path_to_input_ports(storage_path)
    elif step == base_models.IOTypes.OUTPUTS:
        extended_path = _path_to_output_ports(storage_path)
    else:
        extended_path = f"{_path_to_nodes(storage_path)}/{step}"

    try:
        if extended_path not in bag:
            raise _CannotFindLocationError(extended_path)
    except ValueError:  # bagofholding.exceptions.InvalidMetadataError
        raise _CannotFindLocationError(extended_path) from None

    return extended_path
