"""
Test that all Python code examples in README.md execute correctly.

Strategy:
    1. Extract all ```python fenced code blocks from README.md
    2. Parse doctest-style ``>>> `` / ``... `` lines into executable Python
    3. Convert ``>>> expression`` / ``expected_output`` pairs into assertions
    4. Concatenate everything into a single temporary ``.py`` file
    5. Import that file as a module via importlib

Writing to a real file is essential: flowrep's ``@workflow`` and ``@atomic``
decorators call ``inspect.getsource``, which requires a backing ``.py`` file.
Registering the module in ``sys.modules`` is essential: the built-in WfMS
resolves function references by fully-qualified name at run-time.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

# Adjust the number of .parent calls to reach the repo root from this file's
# location.  E.g. if this file lives at ``tests/unit/test_readme.py``, use
# ``.parent.parent.parent``.
_README = Path(__file__).resolve().parent.parent.parent / "docs" / "README.md"
_MODULE_NAME = "_readme_examples"


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _extract_python_blocks(text: str) -> list[str]:
    """Return the content of every ```python fenced code block."""
    return re.findall(r"```python\n(.*?)```", text, re.DOTALL)


def _is_expression(code: str) -> bool:
    """Return ``True`` if *code* can be compiled as an expression."""
    try:
        compile(code, "<test>", "eval")
        return True
    except SyntaxError:
        return False


def _block_to_executable(block: str) -> str:
    """Convert a doctest-style code block into an executable script fragment.

    ``>>> `` lines are code, ``... `` lines are continuations.  A non-blank,
    non-prefixed line following an expression is treated as expected output and
    turned into an ``assert``.
    """
    lines = block.rstrip("\n").split("\n")
    out: list[str] = []
    pending: str | None = None
    counter = 0

    def _flush_as_code() -> None:
        nonlocal pending
        if pending is not None:
            out.append(pending)
            pending = None

    def _flush_with_assert(expected: str) -> None:
        nonlocal pending, counter
        if pending is not None and _is_expression(pending):
            var = f"_rc{counter}"
            exp = f"_ex{counter}"
            counter += 1
            out.append(f"{var} = {pending}")
            out.append(f"{exp} = {expected!r}")
            out.append(
                f"assert repr({var}) == {exp}, "
                f'f"Expected {{{exp}!r}}, got {{repr({var})}}"'
            )
        elif pending is not None:
            # Statement followed by a spurious output line — just emit the code
            out.append(pending)
        pending = None

    for line in lines:
        if line.startswith(">>> "):
            _flush_as_code()
            pending = line[4:]
        elif line.startswith("... "):
            if pending is not None:
                pending += "\n" + line[4:]
            else:
                out.append(line[4:])
        elif line.strip() == "":
            _flush_as_code()
            out.append("")
        else:
            _flush_with_assert(line.strip())

    _flush_as_code()
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


class TestReadmeExamples(unittest.TestCase):
    """Every ``python`` code block in README.md must execute without error."""

    def test_examples_execute(self):
        self.assertTrue(_README.exists(), f"README not found at {_README}")
        md = _README.read_text(encoding="utf-8")
        blocks = _extract_python_blocks(md)
        self.assertTrue(blocks, "No ```python blocks found in README.md")

        script = "\n\n".join(_block_to_executable(b) for b in blocks)

        # Write to a real file so inspect.getsource works inside decorators
        fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="readme_")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(script)

            spec = importlib.util.spec_from_file_location(_MODULE_NAME, tmp_path)
            assert spec is not None and spec.loader is not None
            mod = importlib.util.module_from_spec(spec)

            # The module must be importable by name so the WfMS can resolve
            # function references like ``_readme_examples.add`` at run-time.
            sys.modules[_MODULE_NAME] = mod
            spec.loader.exec_module(mod)
        finally:
            sys.modules.pop(_MODULE_NAME, None)
            os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main()
