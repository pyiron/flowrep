"""Render Python defaults and annotations into self-contained source text.

Each helper produces candidate source text and then verifies that the text
reconstructs the original object (render-then-verify). On any failure it returns
``None`` so the caller can fall back to binding the live object into a namespace.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import types
import typing
from typing import Any

from pyiron_snippets import versions

_ANNOTATED_ORIGIN_SENTINEL = typing.get_origin(typing.Annotated[int, "x"])


def render_annotation(ann: Any, imports: set[str]) -> str | None:
    """Return verified source text for an annotation, or ``None``.

    Mutates ``imports`` with ``"import {module}"`` lines for every non-builtin
    module the text references. Returns ``None`` (without leaving partial imports
    behind) when any part of the annotation cannot be rendered or the rendered
    text fails to evaluate back to an equal object.
    """
    local_imports: set[str] = set()
    text = _render(ann, local_imports)
    if text is None:
        return None
    if text == "None":  # None / NoneType both legal and normalized by get_type_hints
        return "None"
    if not _verifies(text, ann, local_imports):  # pragma: no cover - defensive
        return None
    imports.update(local_imports)
    return text


def _render(ann: Any, imports: set[str]) -> str | None:
    if ann is None or ann is type(None):
        return "None"

    origin = typing.get_origin(ann)
    if origin is not None:
        return _render_subscripted(ann, origin, imports)

    if isinstance(ann, type):
        return _render_plain_type(ann, imports)
    return None


def _render_subscripted(ann: Any, origin: Any, imports: set[str]) -> str | None:
    args = typing.get_args(ann)

    # Union / Optional: render members joined by ` | `. typing.Union[...] has
    # origin typing.Union; the PEP 604 `int | str` form has origin
    # types.UnionType. Both compare equal after eval, so render both as `A | B`.
    if origin is typing.Union or origin is types.UnionType:
        rendered = [_render(a, imports) for a in args]
        if any(part is None for part in rendered):
            return None
        return " | ".join(part for part in rendered if part is not None)

    # Annotated[T, *meta]: base via _render, metadata via render_default first
    # (literals) then _render (types).
    if origin is _ANNOTATED_ORIGIN_SENTINEL:
        base = _render(args[0], imports)
        if base is None:
            return None
        meta_texts = []
        for meta in args[1:]:
            meta_text = render_default(meta)
            if meta_text is None:
                meta_text = _render(meta, imports)
            if meta_text is None:
                return None
            meta_texts.append(meta_text)
        return f"typing.Annotated[{', '.join([base, *meta_texts])}]"

    # Literal[...]: every arg must be a literal value.
    if origin is typing.Literal:
        arg_texts = [render_default(a) for a in args]
        if any(part is None for part in arg_texts):
            return None
        return f"typing.Literal[{', '.join(part for part in arg_texts if part)}]"

    # Generic alias over a real origin type (list[int], dict[str, int], ...).
    if isinstance(origin, type):
        origin_text = _render_plain_type(origin, imports)
        if origin_text is None:
            return None
        arg_texts = [_render(a, imports) for a in args]
        if any(part is None for part in arg_texts):
            return None
        return f"{origin_text}[{', '.join(part for part in arg_texts if part)}]"

    return None


def _render_plain_type(ann: type, imports: set[str]) -> str | None:
    info = versions.VersionInfo.of(ann)
    if info.is_local or info.is_lambda or info.in_main:
        return None
    if info.module != "builtins":
        imports.add(f"import {info.module}")
    return info.findable_at


def _verifies(text: str, ann: Any, imports: set[str]) -> bool:
    namespace: dict[str, Any] = {"__builtins__": builtins}
    namespace["typing"] = importlib.import_module("typing")
    for line in imports:
        top = line.removeprefix("import ").split(".")[0]
        namespace[top] = importlib.import_module(top)
    try:
        evaluated = eval(text, namespace)  # noqa: S307 - controlled codegen text
    except Exception:  # noqa: BLE001  # pragma: no cover - defensive verify net
        return False
    return evaluated == ann or evaluated is ann


def render_default(value: Any) -> str | None:
    """Return source text for a *literal* default, or ``None``.

    The text is accepted only when ``ast.literal_eval`` reproduces a value equal
    to (and of the same type as) the input. This covers ints, floats, strs,
    bytes, bools, ``None``, and nested tuple/list/dict/set of these, and rejects
    sentinels, enums, ``nan``, ``frozenset`` (no literal syntax), and arbitrary
    instances. No imports are required for any accepted value.
    """
    try:
        text = repr(value)
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None
    if type(parsed) is type(value) and parsed == value:
        return text
    return None
