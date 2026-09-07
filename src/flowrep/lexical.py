"""
Shared helpers for "lexical" paths -- ``"."``-joined node labels, IO type
segments, and port names.

:cls:`flowrep.base_models.RESERVED_NAMES` forbids nodes and ports from being
named ``inputs`` or ``outputs``, so a lexical path is unambiguous.
"""

from __future__ import annotations

from flowrep import base_models

DELIMITER = "."


def join(*segments: str) -> str:
    """Join path segments, ignoring empty ones (e.g. the root's empty path)."""
    return DELIMITER.join(segment for segment in segments if segment)


def split(path: str) -> tuple[str, ...]:
    """Split a lexical path into its segments; the empty path has none."""
    return tuple(path.split(DELIMITER)) if path else ()


def port_path(node_path: str, io_type: base_models.IOTypes, port: str) -> str:
    """The lexical path of a port on the node at ``node_path``."""
    return join(node_path, str(io_type), port)
