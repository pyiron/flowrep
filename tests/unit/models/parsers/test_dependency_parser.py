import ast
import textwrap
import unittest
from pyiron_snippets import versions
from unittest.mock import MagicMock, patch

from flowrep.models.parsers import dependency_parser


class TestSplitByVersionAvailability(unittest.TestCase):
    def test_split_by_version_availability(self):
        mock_version_1 = MagicMock(version="1.0.0")
        mock_version_2 = MagicMock(version=None)
        mock_func_1 = MagicMock()
        mock_func_2 = MagicMock()

        call_dependencies = {
            mock_version_1: mock_func_1,
            mock_version_2: mock_func_2,
        }

        has_version, no_version = dependency_parser.split_by_version_availability(
            call_dependencies
        )

        self.assertIn(mock_version_1, has_version)
        self.assertIn(mock_version_2, no_version)
        self.assertNotIn(mock_version_1, no_version)
        self.assertNotIn(mock_version_2, has_version)


class TestUndefinedVariableVisitor(unittest.TestCase):
    def test_undefined_variable_visitor(self):
        source_code = """
        def test_function(a: int, b):
            c = a + b
            return d
        """
        tree = ast.parse(textwrap.dedent(source_code))
        visitor = dependency_parser.UndefinedVariableVisitor()
        visitor.visit(tree)

        self.assertIn("d", visitor.used_vars)
        self.assertIn("int", visitor.used_vars)
        self.assertIn("a", visitor.defined_vars)
        self.assertIn("b", visitor.defined_vars)
        self.assertIn("c", visitor.defined_vars)
        self.assertNotIn("d", visitor.defined_vars)


class TestFindUndefinedVariables(unittest.TestCase):
    def test_find_undefined_variables(self):
        x = 1

        def test_function(a, b):
            c = a + b + x
            return c

        undefined_vars = dependency_parser.find_undefined_variables(test_function)
        self.assertIn("x", undefined_vars)
        self.assertNotIn("a", undefined_vars)
        self.assertNotIn("b", undefined_vars)
        self.assertNotIn("c", undefined_vars)


class TestGetCallDependencies(unittest.TestCase):
    @patch("flowrep.models.parsers.object_scope.get_scope")
    @patch("flowrep.models.parsers.object_scope.resolve_attribute_to_object")
    @patch("pyiron_snippets.versions.VersionInfo.of")
    def test_get_call_dependencies(
        self, mock_version_info_of, mock_resolve_attribute_to_object, mock_get_scope
    ):
        mock_func = MagicMock()
        mock_version_info = MagicMock()
        mock_version_info.fully_qualified_name = "mock_func"
        mock_version_info_of.return_value = mock_version_info
        mock_resolve_attribute_to_object.return_value = mock_func

        mock_scope = MagicMock()
        mock_get_scope.return_value = mock_scope

        with patch(
            "flowrep.models.parsers.dependency_parser.find_undefined_variables"
        ) as mock_find_undefined:
            mock_find_undefined.return_value = {"undefined_var"}
            call_dependencies = dependency_parser.get_call_dependencies(mock_func)

        self.assertIn(mock_version_info, call_dependencies)
        self.assertEqual(call_dependencies[mock_version_info], mock_func)
        mock_get_scope.assert_called_once_with(mock_func)
        mock_resolve_attribute_to_object.assert_called_once_with(
            "undefined_var", mock_scope
        )

    def test_type_hints(self):
        def test_function(a: np.array, b: np.array):
            return a + b

        self.assertDictEqual(
            {versions.VersionInfo.of(np): np},
            dependency_parser.get_call_dependencies(test_function),
        )


if __name__ == "__main__":
    unittest.main()
