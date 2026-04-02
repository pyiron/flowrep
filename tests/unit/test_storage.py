"""
Tests for the storage module (bag validation, lexical path listing, loading,
and the LexicalBagBrowser convenience class).
"""

import os
import pathlib
import tempfile
import unittest
from unittest import mock

import bagofholding as boh

from flowrep import live, storage, wfms, widget

from flowrep_static import makers

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _save_workflow(path: str, **kwargs: object) -> live.LiveWorkflow:
    """Run the simple workflow recipe and save the result to *path*."""
    recipe = makers.make_simple_workflow_recipe()
    live_wf = wfms.run_recipe(recipe, **kwargs)
    boh.H5Bag.save(live_wf, path)
    return live_wf


class _BagTestCase(unittest.TestCase):
    """Base class that provides a temporary directory cleaned up after each test."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmpdir.name
        self.addCleanup(self._tmpdir.cleanup)

    def _bag_path(self, name: str = "test.h5") -> str:
        return os.path.join(self.tmpdir, name)


# ═══════════════════════════════════════════════════════════════════════════
# validate_bag / _validate_bag_metadata / _validate_object_metadata
# ═══════════════════════════════════════════════════════════════════════════


class TestValidateBag(_BagTestCase):
    def test_not_h5bag_raises_type_error(self):
        with self.assertRaisesRegex(TypeError, boh.H5Bag.__name__):
            storage.validate_bag("not a bag")

    def test_valid_bag(self):
        path = self._bag_path()
        _save_workflow(path, a=1, b=2)
        bag = boh.H5Bag(path)
        storage.validate_bag(bag)  # should not raise


class TestValidateBagMetadata(_BagTestCase):
    def test_wrong_version(self):
        path = self._bag_path()
        _save_workflow(path, a=1, b=2)
        bag = boh.H5Bag(path)
        fake_info = mock.Mock()
        fake_info.version = "99.99.99"
        with (
            mock.patch.object(bag, "get_bag_info", return_value=fake_info),
            self.assertRaisesRegex(ValueError, "99.99.99"),
        ):
            storage._validate_bag_metadata(bag)


class TestValidateObjectMetadata(_BagTestCase):
    def test_wrong_object_qualname(self):
        """Saving a non-LiveWorkflow object should fail the qualname check."""
        path = self._bag_path()
        boh.H5Bag.save(42, path)
        bag = boh.H5Bag(path)
        with self.assertRaisesRegex(TypeError, live.LiveWorkflow.__qualname__):
            storage._validate_object_metadata(bag)


# ═══════════════════════════════════════════════════════════════════════════
# list_lexical_paths / _collect_lexical_paths
# ═══════════════════════════════════════════════════════════════════════════


class TestListLexicalPaths(_BagTestCase):
    def test_single_child_workflow(self):
        """Paths should include workflow IO, child node, and child IO."""
        path = self._bag_path()
        _save_workflow(path, a=3, b=4)
        bag = boh.H5Bag(path)

        paths = storage.list_lexical_paths(bag)

        expected = {
            # Workflow-level IO
            "inputs.a",
            "inputs.b",
            "outputs.result",
            # Child node
            "add_0",
            # Child IO
            "add_0.inputs.a",
            "add_0.inputs.b",
            "add_0.outputs.output_0",
        }
        self.assertEqual(set(paths), expected)


# ═══════════════════════════════════════════════════════════════════════════
# load_from_bag / _extend_path
# ═══════════════════════════════════════════════════════════════════════════


class TestLoadFromBag(_BagTestCase):
    def setUp(self) -> None:
        super().setUp()
        path = self._bag_path()
        self._live_wf = _save_workflow(path, a=3, b=4)
        self._bag = boh.H5Bag(path)

    def test_load_root_workflow(self):
        """Empty lexical path loads the root LiveWorkflow."""
        obj = storage.load_from_bag(self._bag, "")
        self.assertIsInstance(obj, live.LiveWorkflow)

    def test_load_child_node(self):
        obj = storage.load_from_bag(self._bag, "add_0")
        self.assertIsInstance(obj, live.LiveAtomic)

    def test_load_input_port(self):
        obj = storage.load_from_bag(self._bag, "inputs.a")
        self.assertIsInstance(obj, live.InputPort)

    def test_load_output_port(self):
        obj = storage.load_from_bag(self._bag, "outputs.result")
        self.assertIsInstance(obj, live.OutputPort)

    def test_load_child_input_port(self):
        obj = storage.load_from_bag(self._bag, "add_0.inputs.a")
        self.assertIsInstance(obj, live.InputPort)

    def test_load_child_output_port(self):
        obj = storage.load_from_bag(self._bag, "add_0.outputs.output_0")
        self.assertIsInstance(obj, live.OutputPort)

    def test_load_child_output_port_has_data(self):
        """The saved workflow was run with a=3, b=4, so output should be 7."""
        obj = storage.load_from_bag(self._bag, "add_0.outputs.output_0")
        self.assertIsInstance(obj, live.OutputPort)
        self.assertEqual(obj.value, 7)

    def test_terminated_in_inputs_raises(self):
        with self.assertRaisesRegex(ValueError, "terminated in 'inputs'"):
            storage.load_from_bag(self._bag, "inputs")

    def test_terminated_in_outputs_raises(self):
        with self.assertRaisesRegex(ValueError, "terminated in 'outputs'"):
            storage.load_from_bag(self._bag, "outputs")

    def test_terminated_in_child_inputs_raises(self):
        with self.assertRaisesRegex(ValueError, "terminated in 'inputs'"):
            storage.load_from_bag(self._bag, "add_0.inputs")

    def test_invalid_path_raises(self):
        with self.assertRaisesRegex(ValueError, "nonexistent"):
            storage.load_from_bag(self._bag, "nonexistent")

    def test_invalid_nested_path_includes_walked_path(self):
        with self.assertRaisesRegex(ValueError, r"at 'add_0\.bad_port'"):
            storage.load_from_bag(self._bag, "add_0.bad_port")

    def test_invalid_port_under_io_group(self):
        with self.assertRaisesRegex(ValueError, "no_such_port"):
            storage.load_from_bag(self._bag, "inputs.no_such_port")

    def test_unexpected_loaded_type_raises(self):
        """If bag.load returns something outside the expected union, raise TypeError."""
        with (
            mock.patch.object(self._bag, "load", return_value="a plain string"),
            self.assertRaisesRegex(TypeError, "Expected to load one of"),
        ):
            storage.load_from_bag(self._bag, "add_0")


class TestExtendPath(_BagTestCase):
    def setUp(self) -> None:
        super().setUp()
        path = self._bag_path()
        _save_workflow(path, a=1, b=2)
        self._bag = boh.H5Bag(path)

    def test_step_into_node(self):
        result = storage._extend_path(self._bag, "object", "add_0", "")
        self.assertEqual(result, "object/state/nodes/add_0")

    def test_step_into_inputs(self):
        result = storage._extend_path(self._bag, "object", "inputs", "")
        self.assertEqual(result, "object/state/input_ports")

    def test_step_into_outputs(self):
        result = storage._extend_path(self._bag, "object", "outputs", "")
        self.assertEqual(result, "object/state/output_ports")

    def test_step_into_port_after_io_type(self):
        result = storage._extend_path(
            self._bag, "object/state/input_ports", "a", "inputs"
        )
        self.assertEqual(result, "object/state/input_ports/a")

    def test_missing_path_raises(self):
        with self.assertRaises(storage._CannotFindLocationError):
            storage._extend_path(self._bag, "object", "does_not_exist", "")


# ═══════════════════════════════════════════════════════════════════════════
# LexicalBagBrowser
# ═══════════════════════════════════════════════════════════════════════════


class TestLexicalBagBrowserInit(_BagTestCase):
    def _save(self) -> str:
        path = self._bag_path()
        _save_workflow(path, a=1, b=2)
        return path

    def test_init_with_str_path(self):
        path = self._save()
        browser = storage.LexicalBagBrowser(path)
        self.assertIsInstance(browser.bag, boh.H5Bag)

    def test_init_with_pathlib_path(self):
        path = self._save()
        browser = storage.LexicalBagBrowser(pathlib.Path(path))
        self.assertIsInstance(browser.bag, boh.H5Bag)

    def test_init_with_bag_instance(self):
        path = self._save()
        bag = boh.H5Bag(path)
        browser = storage.LexicalBagBrowser(bag)
        self.assertIs(browser.bag, bag)

    def test_init_validates(self):
        """Constructor should reject a bag with the wrong object type."""
        path = self._bag_path()
        boh.H5Bag.save(42, path)
        with self.assertRaises(TypeError):
            storage.LexicalBagBrowser(path)


class TestLexicalBagBrowserMethods(_BagTestCase):
    def setUp(self) -> None:
        super().setUp()
        path = self._bag_path()
        _save_workflow(path, a=3, b=4)
        self.browser = storage.LexicalBagBrowser(path)

    def test_list_paths(self):
        paths = self.browser.list_paths()
        self.assertIn("add_0", paths)
        self.assertIn("inputs.a", paths)

    def test_load_node(self):
        obj = self.browser.load("add_0")
        self.assertIsInstance(obj, live.LiveAtomic)

    def test_load_port(self):
        obj = self.browser.load("inputs.a")
        self.assertIsInstance(obj, live.InputPort)

    def test_browse_returns_widget(self):
        result = self.browser.browse()
        self.assertIsInstance(result, widget.LexicalBagTree)

    def test_browse_falls_back_to_list(self):
        with mock.patch.object(self.browser, "widget", side_effect=ImportError):
            result = self.browser.browse()
        self.assertIsInstance(result, list)

    def test_widget_returns_tree(self):
        result = self.browser.widget()
        self.assertIsInstance(result, widget.LexicalBagTree)


if __name__ == "__main__":
    unittest.main()
