"""Compiling and building a workflow from inside a module that is *still importing*.

``flowrep2python`` emits fully-qualified child calls (``pkg.mod.add(...)``), and
``RenderedSource.build`` re-parses that emitted source, so the parser has to resolve
``pkg.mod.add`` by walking attributes down from ``pkg``. CPython only sets the
submodule attribute on a parent package once the submodule's body has finished
executing, so a module that compiles one of its own workflows at import time hands the
parser a package that does not yet know about it -- ``getattr(pkg, "mod")`` raises
``AttributeError: cannot access submodule 'mod' of module 'pkg'``.

``sys.modules`` holds the entry from the moment the import begins, which is what
``object_scope.resolve_attribute_to_object`` falls back on.
"""

import importlib
import pathlib
import sys
import tempfile
import textwrap
import unittest

_PACKAGE = "_flowrep_still_importing_pkg"

_MODULE_SOURCE = """
import flowrep


@flowrep.atomic("z")
def add(x, y):
    z = x + y
    return z


@flowrep.workflow("out")
def wf(x, y):
    s = add(x=x, y=y)
    return s


# Compiling here, at module scope, is the whole point of the fixture: the parent
# package has no attribute for this module yet.
recipe = wf.flowrep_recipe.model_copy(update={"reference": None})
built = flowrep.tools.flowrep2python(recipe, function_name="wf").build()
"""


class TestBuildingWhileImporting(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        package = pathlib.Path(self._tmp.name) / _PACKAGE
        package.mkdir()
        (package / "__init__.py").touch()
        (package / "mod.py").write_text(textwrap.dedent(_MODULE_SOURCE))
        sys.path.insert(0, self._tmp.name)
        importlib.invalidate_caches()

    def tearDown(self):
        sys.path.remove(self._tmp.name)
        for name in [n for n in sys.modules if n.split(".")[0] == _PACKAGE]:
            del sys.modules[name]
        self._tmp.cleanup()

    def test_child_reference_resolves_before_the_import_finishes(self):
        module = importlib.import_module(f"{_PACKAGE}.mod")
        self.assertEqual(
            3,
            module.built(x=1, y=2),
            msg="The re-parsed source must reach `add` through the half-imported "
            "package and still describe the same computation.",
        )


if __name__ == "__main__":
    unittest.main()
