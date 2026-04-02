"""Tests for the widget module (LexicalBagTree)."""

import os
import tempfile
import unittest

import bagofholding as boh
import ipytree

from flowrep import live, storage, storage_widget, wfms

from flowrep_static import makers

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _save_workflow(path: str, **kwargs: object) -> live.LiveWorkflow:
    recipe = makers.make_simple_workflow_recipe()
    live_wf = wfms.run_recipe(recipe, **kwargs)
    boh.H5Bag.save(live_wf, path)
    return live_wf


class _WidgetTestCase(unittest.TestCase):
    """Base class providing a browser backed by a temporary H5 file."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmpdir.name
        self.addCleanup(self._tmpdir.cleanup)

        path = os.path.join(self.tmpdir, "test.h5")
        _save_workflow(path, a=3, b=4)
        self.browser = storage.LexicalBagBrowser(path)
        self.tree = storage_widget.LexicalBagTree(self.browser)

    def _root_node(self) -> ipytree.Node:
        return self.tree.nodes[0]


# ═══════════════════════════════════════════════════════════════════════════
# Construction
# ═══════════════════════════════════════════════════════════════════════════


class TestTreeConstruction(_WidgetTestCase):
    def test_root_exists(self):
        self.assertEqual(len(self.tree.nodes), 1)

    def test_root_label(self):
        self.assertEqual(self._root_node().name, "Workflow")

    def test_root_metadata_registered(self):
        meta = self.tree._meta(self._root_node())
        self.assertEqual(meta.lexical_path, "")
        self.assertEqual(meta.storage_path, "object/")

    def test_root_is_opened_and_loaded(self):
        meta = self.tree._meta(self._root_node())
        self.assertTrue(meta.loaded)

    def test_root_has_children(self):
        """Root should be pre-populated since it is opened on construction."""
        root = self._root_node()
        # At minimum: inputs group, outputs group, add_0 child
        self.assertGreaterEqual(len(root.nodes), 3)

    def test_node_meta_populated_for_children(self):
        """Every child of root should also be tracked in _node_meta."""
        root = self._root_node()
        for child in root.nodes:
            with self.subTest(label=child.name):
                meta = self.tree._meta(child)
                self.assertIsNotNone(meta)


# ═══════════════════════════════════════════════════════════════════════════
# _meta lookup
# ═══════════════════════════════════════════════════════════════════════════


class TestMetaLookup(_WidgetTestCase):
    def test_known_node(self):
        meta = self.tree._meta(self._root_node())
        self.assertIsInstance(meta, storage_widget._NodeMeta)

    def test_unknown_node_raises(self):
        stranger = ipytree.Node("stranger")
        with self.assertRaisesRegex(ValueError, "not tracked"):
            self.tree._meta(stranger)


# ═══════════════════════════════════════════════════════════════════════════
# load_selected
# ═══════════════════════════════════════════════════════════════════════════


class TestLoadSelected(_WidgetTestCase):
    def test_nothing_selected_raises(self):
        self.assertIsNone(self.tree.selected_lexical_path)
        with self.assertRaisesRegex(ValueError, "No entry selected"):
            self.tree.load_selected()

    def test_after_selection(self):
        """Manually set the selected path, then load."""
        self.tree.selected_lexical_path = "add_0"
        obj = self.tree.load_selected()
        self.assertIsInstance(obj, live.LiveAtomic)


# ═══════════════════════════════════════════════════════════════════════════
# _on_select callback
# ═══════════════════════════════════════════════════════════════════════════


class TestOnSelect(_WidgetTestCase):
    def test_select_root_sets_none(self):
        """Root has lexical_path='', which should normalize to None."""
        root = self._root_node()
        self.tree._on_select({"new": [root]})
        self.assertIsNone(self.tree.selected_lexical_path)

    def test_select_child_node(self):
        root = self._root_node()
        # Find the add_0 child node
        add_node = next(n for n in root.nodes if n.name == "add_0")
        self.tree._on_select({"new": [add_node]})
        self.assertEqual(self.tree.selected_lexical_path, "add_0")

    def test_deselect(self):
        self.tree.selected_lexical_path = "add_0"
        self.tree._on_select({"new": []})
        # Selection should remain unchanged when new selection is empty
        self.assertEqual(self.tree.selected_lexical_path, "add_0")

    def test_untracked_node_raises(self):
        stranger = ipytree.Node("stranger")
        with self.assertRaises(ValueError):
            self.tree._on_select({"new": [stranger]})


# ═══════════════════════════════════════════════════════════════════════════
# Lazy expansion
# ═══════════════════════════════════════════════════════════════════════════


class TestLazyExpand(_WidgetTestCase):
    def _find_collapsed_node(self) -> ipytree.Node:
        """Find a child that has a '...' placeholder (not yet expanded)."""
        root = self._root_node()
        for child in root.nodes:
            meta = self.tree._meta(child)
            if not meta.loaded:
                return child

    def test_collapsed_node_has_placeholder(self):
        node = self._find_collapsed_node()
        self.assertEqual(len(node.nodes), 1)
        self.assertEqual(node.nodes[0].name, "...")

    def test_expand_populates_children(self):
        node = self._find_collapsed_node()
        meta = self.tree._meta(node)
        self.assertFalse(meta.loaded)

        # Simulate opening the node by calling _lazy_expand directly
        self.tree._lazy_expand({"owner": node})

        self.assertTrue(meta.loaded)
        # Placeholder should be gone, real children should be present
        child_names = [c.name for c in node.nodes]
        self.assertNotIn("...", child_names)
        self.assertGreater(len(node.nodes), 0)

    def test_expand_is_idempotent(self):
        node = self._find_collapsed_node()
        self.tree._lazy_expand({"owner": node})
        children_after_first = list(node.nodes)
        self.tree._lazy_expand({"owner": node})
        self.assertEqual(list(node.nodes), children_after_first)

    def test_expanded_children_are_tracked(self):
        node = self._find_collapsed_node()
        self.tree._lazy_expand({"owner": node})
        for child in node.nodes:
            with self.subTest(label=child.name):
                meta = self.tree._meta(child)
                self.assertIsNotNone(meta)


# ═══════════════════════════════════════════════════════════════════════════
# _has_expandable_children
# ═══════════════════════════════════════════════════════════════════════════


class TestHasExpandableChildren(_WidgetTestCase):
    def test_root_has_children(self):
        self.assertTrue(self.tree._has_expandable_children("object/"))

    def test_leaf_port_has_no_children(self):
        self.assertFalse(
            self.tree._has_expandable_children("object/state/input_ports/a")
        )

    def test_nonexistent_path_has_no_children(self):
        self.assertFalse(
            self.tree._has_expandable_children("object/state/nodes/nonexistent")
        )


# ═══════════════════════════════════════════════════════════════════════════
# IO group and port children
# ═══════════════════════════════════════════════════════════════════════════


class TestIOGroupExpansion(_WidgetTestCase):
    def _find_io_group(self, io_type: str) -> ipytree.Node:
        root = self._root_node()
        for child in root.nodes:
            if child.name == io_type:
                return child
        self.fail(f"Could not find {io_type!r} group node")

    def test_inputs_group_exists(self):
        node = self._find_io_group("inputs")
        meta = self.tree._meta(node)
        self.assertTrue(meta.is_io_group)

    def test_outputs_group_exists(self):
        node = self._find_io_group("outputs")
        meta = self.tree._meta(node)
        self.assertTrue(meta.is_io_group)

    def test_inputs_group_expands_to_ports(self):
        node = self._find_io_group("inputs")
        self.tree._lazy_expand({"owner": node})
        port_names = {c.name for c in node.nodes}
        self.assertEqual(port_names, {"a", "b"})

    def test_outputs_group_expands_to_ports(self):
        node = self._find_io_group("outputs")
        self.tree._lazy_expand({"owner": node})
        port_names = {c.name for c in node.nodes}
        self.assertEqual(port_names, {"result"})

    def test_port_lexical_path(self):
        node = self._find_io_group("inputs")
        self.tree._lazy_expand({"owner": node})
        port_a = next(c for c in node.nodes if c.name == "a")
        meta = self.tree._meta(port_a)
        self.assertEqual(meta.lexical_path, "inputs.a")


if __name__ == "__main__":
    unittest.main()
