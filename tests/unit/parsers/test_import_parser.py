import ast
import os
import unittest

from flowrep.parsers import import_parser


class TestBuildScope(unittest.TestCase):
    def test_empty_scope(self):
        scope = import_parser.build_scope()
        self.assertEqual(len(scope), 0)

    def test_import_registers_module_by_name(self):
        node = ast.parse("import os").body[0]
        scope = import_parser.build_scope(imports=[node])
        self.assertIs(scope["os"], os)

    def test_import_registers_module_by_asname(self):
        node = ast.parse("import os as operating_system").body[0]
        scope = import_parser.build_scope(imports=[node])
        self.assertIs(scope["operating_system"], os)
        self.assertNotIn("os", scope)

    def test_import_from_registers_object_by_name(self):
        node = ast.parse("from os import path").body[0]
        scope = import_parser.build_scope(import_froms=[node])
        self.assertIs(scope["path"], os.path)

    def test_import_from_registers_object_by_asname(self):
        node = ast.parse("from os import path as p").body[0]
        scope = import_parser.build_scope(import_froms=[node])
        self.assertIs(scope["p"], os.path)
        self.assertNotIn("path", scope)

    def test_relative_import_raises(self):
        """level > 0 (relative import) must raise ValueError."""
        node = ast.parse("from .foo import bar").body[0]
        with self.assertRaises(ValueError):
            import_parser.build_scope(import_froms=[node])

    def test_none_module_raises(self):
        """module=None with level=0 must also raise ValueError."""
        node = ast.ImportFrom(module=None, names=[ast.alias(name="something")], level=0)
        with self.assertRaises(ValueError):
            import_parser.build_scope(import_froms=[node])


if __name__ == "__main__":
    unittest.main()
