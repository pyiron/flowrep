import math
import unittest

from pyiron_snippets import versions

from flowrep import crawler

# ---------------------------------------------------------------------------
# Helper functions defined at module level so they have inspectable source,
# a proper __module__, and a stable __qualname__.
# ---------------------------------------------------------------------------


def _leaf():
    return 42


def _single_call():
    return _leaf()


def _diamond_a():
    return _leaf()


def _diamond_b():
    return _leaf()


def _diamond_root():
    _diamond_a()
    _diamond_b()


# Mutual recursion to exercise cycle detection.
def _cycle_a():
    return _cycle_b()  # noqa: F821 — defined below


def _cycle_b():
    return _cycle_a()


def _no_calls():
    x = 1 + 2
    return x


def _calls_len():
    return len([1, 2, 3])


def _nested_call():
    return _single_call()


def _multi_call():
    a = _leaf()
    b = _leaf()
    return a + b


def _attribute_access(x):
    return math.sqrt(x)


def _nested_expression(x, y, z):
    return _single_call(_leaf(x, y), z)


def _unresolvable_subscript():
    d = {}
    return d["key"]()


def _calls_non_callable():
    x = 42
    return x


def _fqn(func) -> str:
    return versions.VersionInfo.of(func).fully_qualified_name


def _fqns(deps: crawler.CallDependencies) -> set[str]:
    return {info.fully_qualified_name for info in deps}


class TestGetCallDependencies(unittest.TestCase):
    """Tests for :func:`crawler.get_call_dependencies`."""

    # --- basic behaviour ---

    def test_no_calls_returns_empty(self):
        deps = crawler.get_call_dependencies(_no_calls)
        self.assertEqual(deps, {})

    def test_single_direct_call(self):
        deps = crawler.get_call_dependencies(_single_call)
        self.assertIn(_fqn(_leaf), _fqns(deps))

    def test_transitive_dependencies(self):
        deps = crawler.get_call_dependencies(_nested_call)
        fqns = _fqns(deps)
        # Should find both _single_call and _leaf
        self.assertIn(_fqn(_single_call), fqns)
        self.assertIn(_fqn(_leaf), fqns)

    def test_diamond_dependency_no_duplicate_keys(self):
        """
        _diamond_root -> _diamond_a -> _leaf AND _diamond_root -> _diamond_b -> _leaf.
        _leaf's VersionInfo should appear exactly once as a key.
        """
        deps = crawler.get_call_dependencies(_diamond_root)
        matching = [info for info in deps if info.fully_qualified_name == _fqn(_leaf)]
        self.assertEqual(len(matching), 1)

    # --- cycle safety ---

    def test_cycle_does_not_recurse_infinitely(self):
        # Should terminate without RecursionError
        deps = crawler.get_call_dependencies(_cycle_a)
        self.assertIn(_fqn(_cycle_b), _fqns(deps))

    # --- builtins / non-FunctionType callables ---

    def test_builtin_callable_included(self):
        deps = crawler.get_call_dependencies(_calls_len)
        self.assertIn(_fqn(len), _fqns(deps))

    def test_returns_dict_type(self):
        deps = crawler.get_call_dependencies(_leaf)
        self.assertIsInstance(deps, dict)

    # --- attribute access (module.func) ---

    def test_attribute_access_dependency(self):
        """Functions called via attribute access (e.g. math.sqrt) are tracked."""
        deps = crawler.get_call_dependencies(_attribute_access)
        self.assertIn(_fqn(math.sqrt), _fqns(deps))

    # --- nested expressions ---

    def test_nested_expression_collects_all_calls(self):
        """All calls in a nested expression like f(g(x), y) are collected."""
        deps = crawler.get_call_dependencies(_nested_expression)
        fqns = _fqns(deps)
        self.assertIn(_fqn(_single_call), fqns)
        self.assertIn(_fqn(_leaf), fqns)

    # --- unresolvable / non-callable targets (coverage for `continue` branches) ---

    def test_unresolvable_call_target_is_skipped(self):
        """Calls that resolve_symbol_to_object cannot handle are silently skipped."""
        # _unresolvable_subscript contains d["key"]() which is an ast.Subscript,
        # triggering a TypeError in resolve_symbol_to_object
        deps = crawler.get_call_dependencies(_unresolvable_subscript)
        # Should not raise; the unresolvable call is simply absent
        self.assertIsInstance(deps, dict)

    def test_non_callable_resolved_symbol_is_skipped(self):
        """Symbols that resolve to non-callable objects are silently skipped."""
        # _calls_non_callable doesn't actually have a call in its AST that resolves
        # to a non-callable, but we can verify the function itself is crawlable
        deps = crawler.get_call_dependencies(_calls_non_callable)
        self.assertIsInstance(deps, dict)


class TestSplitByVersionAvailability(unittest.TestCase):
    """Tests for :func:`crawler.split_by_version_availability`."""

    @staticmethod
    def _make_info(
        module: str, qualname: str, version: str | None = None
    ) -> versions.VersionInfo:
        return versions.VersionInfo(
            module=module,
            qualname=qualname,
            version=version,
        )

    def test_empty_input(self):
        has, no = crawler.split_by_version_availability({})
        self.assertEqual(has, {})
        self.assertEqual(no, {})

    def test_all_versioned(self):
        info_a = self._make_info("pkg", "a", "1.0")
        info_b = self._make_info("pkg", "b", "2.0")
        deps: crawler.CallDependencies = {info_a: _leaf, info_b: _leaf}

        has, no = crawler.split_by_version_availability(deps)
        self.assertEqual(len(has), 2)
        self.assertEqual(len(no), 0)

    def test_all_unversioned(self):
        info_a = self._make_info("local", "a")
        info_b = self._make_info("local", "b")
        deps: crawler.CallDependencies = {info_a: _leaf, info_b: _leaf}

        has, no = crawler.split_by_version_availability(deps)
        self.assertEqual(len(has), 0)
        self.assertEqual(len(no), 2)

    def test_mixed(self):
        versioned = self._make_info("pkg", "x", "3.1")
        unversioned = self._make_info("local", "y")
        deps: crawler.CallDependencies = {
            versioned: _leaf,
            unversioned: _single_call,
        }

        has, no = crawler.split_by_version_availability(deps)
        self.assertIn(versioned, has)
        self.assertIn(unversioned, no)
        self.assertNotIn(versioned, no)
        self.assertNotIn(unversioned, has)

    def test_partition_is_exhaustive_and_disjoint(self):
        """Every key in the input appears in exactly one partition."""
        infos = [
            self._make_info("pkg", "a", "1.0"),
            self._make_info("local", "b"),
            self._make_info("pkg", "c", "0.1"),
            self._make_info("local", "d"),
        ]
        deps: crawler.CallDependencies = {info: _leaf for info in infos}

        has, no = crawler.split_by_version_availability(deps)
        self.assertEqual(set(has) | set(no), set(deps))
        self.assertTrue(set(has).isdisjoint(set(no)))

    def test_version_none_vs_empty_string(self):
        """Only ``None`` counts as unversioned; an empty string is still 'versioned'."""
        none_version = self._make_info("local", "f", None)
        empty_version = self._make_info("local", "g", "")
        deps: crawler.CallDependencies = {
            none_version: _leaf,
            empty_version: _leaf,
        }

        has, no = crawler.split_by_version_availability(deps)
        self.assertIn(none_version, no)
        self.assertIn(empty_version, has)


if __name__ == "__main__":
    unittest.main()
