"""
A simple jupyter notebook widget to empower the storage browser.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from pyiron_snippets import import_alarm

from flowrep import base_models

with import_alarm.ImportAlarm("This tool requires 'ipytree'.") as _import_alarm:
    import ipytree

if TYPE_CHECKING:
    import traitlets  # Expected as a dependency of ipytree

    from flowrep import storage


@dataclasses.dataclass
class _NodeMeta:
    lexical_path: str
    storage_path: str
    is_io_group: bool = False
    loaded: bool = False


class LexicalBagTree(ipytree.Tree):
    """Notebook tree widget driven by lexical paths."""

    @_import_alarm
    def __init__(self, browser: storage.LexicalBagBrowser) -> None:
        super().__init__(multiple_selection=False)
        self._browser = browser
        self._bag = browser.bag
        self._node_meta: dict[int, _NodeMeta] = {}
        self.selected_lexical_path: str | None = None
        self.observe(self._on_select, names=["selected_nodes"])

        root = self._make_node(
            label="Workflow",
            storage_path="object/",
            lexical_path="",
            icon="project-diagram",
            opened=True,
        )
        self.add_node(root)

    def load_selected(self) -> object:
        """Load and instantiate the selected object"""
        if self.selected_lexical_path is None:
            raise ValueError("No entry selected")
        return self._browser.load(self.selected_lexical_path)

    def _meta(self, node: ipytree.Node) -> _NodeMeta:
        try:
            return self._node_meta[id(node)]
        except KeyError:
            raise ValueError(
                f"Node {node.name!r} is not tracked by this tree"
            ) from None

    def _make_node(
        self,
        label: str,
        storage_path: str,
        lexical_path: str,
        *,
        icon: str = "file",
        opened: bool = False,
        is_io_group: bool = False,
    ) -> ipytree.Node:
        has_children = is_io_group or self._has_expandable_children(storage_path)

        node = ipytree.Node(
            label,
            [],
            opened=opened,
            icon="folder" if has_children else icon,
            open_icon_style="success" if has_children else "default",
            close_icon_style="danger",
        )
        self._node_meta[id(node)] = _NodeMeta(
            lexical_path=lexical_path,
            storage_path=storage_path,
            is_io_group=is_io_group,
        )

        if has_children:
            if opened:
                self._populate_children(node)
            else:
                node.add_node(ipytree.Node("...", disabled=True))
                node.observe(self._lazy_expand, names=["opened"])

        return node

    def _has_expandable_children(self, storage_path: str) -> bool:
        """True if storage_path has child nodes or IO port groups."""
        for suffix in (
            "/state/nodes",
            "/state/input_ports",
            "/state/output_ports",
        ):
            try:
                if self._bag.open_group(f"{storage_path}{suffix}"):
                    return True
            except KeyError:
                continue
        return False

    def _lazy_expand(self, change: traitlets.Bunch) -> None:
        node = change["owner"]
        meta = self._meta(node)
        if meta.loaded:
            return
        node.nodes = []
        self._populate_children(node)

    def _populate_children(self, node: ipytree.Node) -> None:
        meta = self._meta(node)
        if meta.is_io_group:
            self._add_port_children(node)
        else:
            self._add_node_children(node)
        meta.loaded = True

    def _add_node_children(self, node: ipytree.Node) -> None:
        meta = self._meta(node)
        storage_path = meta.storage_path
        prefix = f"{meta.lexical_path}." if meta.lexical_path else ""

        # IO groups --------------------------------------------------------
        io_map = {
            base_models.IOTypes.INPUTS: "/state/input_ports",
            base_models.IOTypes.OUTPUTS: "/state/output_ports",
        }
        for io_type, suffix in io_map.items():
            io_storage = f"{storage_path}{suffix}"
            ports = self._bag.open_group(io_storage)
            if not ports:
                continue
            io_node = self._make_node(
                label=str(io_type),
                storage_path=io_storage,
                lexical_path=f"{prefix}{io_type}",
                icon="plug",
                is_io_group=True,
            )
            node.add_node(io_node)

        # Child nodes ------------------------------------------------------
        nodes_storage = f"{storage_path}/state/nodes"
        children = self._bag.open_group(nodes_storage)
        for child in children:
            child_node = self._make_node(
                label=child,
                storage_path=f"{nodes_storage}/{child}",
                lexical_path=f"{prefix}{child}",
                icon="cube",
            )
            node.add_node(child_node)

    def _add_port_children(self, node: ipytree.Node) -> None:
        meta = self._meta(node)
        prefix = f"{meta.lexical_path}." if meta.lexical_path else ""
        ports = self._bag.open_group(meta.storage_path)
        for port in ports:
            port_node = self._make_node(
                label=port,
                storage_path=f"{meta.storage_path}/{port}",
                lexical_path=f"{prefix}{port}",
                icon="sign-in-alt",
            )
            node.add_node(port_node)

    def _on_select(self, change: traitlets.Bunch) -> None:
        selected = change["new"]
        if selected:
            meta = self._meta(selected[0])
            self.selected_lexical_path = meta.lexical_path or None
