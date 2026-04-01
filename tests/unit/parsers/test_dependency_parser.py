import ast
import textwrap
import unittest
from unittest.mock import MagicMock

from flowrep.parsers import dependency_parser


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

    def test_all_argument_kinds_are_defined(self):
        source_code = """
        def test_function(posonly, /, regular, *args, kw_only, **kwargs):
            return posonly + regular + kw_only
        """
        tree = ast.parse(textwrap.dedent(source_code))
        visitor = dependency_parser.UndefinedVariableVisitor()
        visitor.visit(tree)

        for name in ("posonly", "regular", "args", "kw_only", "kwargs"):
            self.assertIn(name, visitor.defined_vars)

    def test_local_function_definition_raises(self):
        source_code = """
        def outer(x):
            def helper(y):
                return y
            return helper(x)
        """
        tree = ast.parse(textwrap.dedent(source_code))
        visitor = dependency_parser.UndefinedVariableVisitor()
        with self.assertRaises(NotImplementedError):
            visitor.visit(tree)

    def test_local_async_function_definition_raises(self):
        source_code = """
        def outer(x):
            async def helper(y):
                return y
            return helper(x)
        """
        tree = ast.parse(textwrap.dedent(source_code))
        visitor = dependency_parser.UndefinedVariableVisitor()
        with self.assertRaises(NotImplementedError):
            visitor.visit(tree)


x = 1


def test_function(a, b):
    c = a + b + x
    return c


class TestFindUndefinedVariables(unittest.TestCase):
    def test_find_undefined_variables(self):
        undefined_vars = dependency_parser.find_undefined_variables(test_function)
        self.assertIn("x", undefined_vars)
        self.assertNotIn("a", undefined_vars)
        self.assertNotIn("b", undefined_vars)
        self.assertNotIn("c", undefined_vars)


if __name__ == "__main__":
    unittest.main()
