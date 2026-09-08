"""
Atomic recipes that materialize fan-out and fan-in as real nodes.

A for-node scatters one input collection across many body instances and gathers their
outputs back into one list. The edge models describe strictly 1:1 connections --
:data:`~flowrep.edge_models.OutputEdges` admits exactly one source per output target --
so a WfMS records that fan-out and fan-in honestly by building nodes for it, rather
than leaving it implicit and unrecordable.

These mirror ``pyiron_workflow.transformers`` deliberately, down to the port labels and
function names, so that records produced by different WfMS describe the same shape.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from pyiron_snippets import versions

from flowrep import base_models
from flowrep.prospective import atomic_recipe


class Transform1toN:
    """
    Scatter one iterable input across ``n`` output ports.

    Attributes:
        input_label: The name of the sole input port.

    Args:
        n: How many output ports to scatter into. Must be at least 1.
    """

    input_label: ClassVar[base_models.Label] = "items"

    @staticmethod
    def output_label(i: int) -> base_models.Label:
        return f"output_{i}"

    @staticmethod
    def iterable_to_outputs(items, /):
        return tuple(items)

    @staticmethod
    def iterable_to_output(items, /):
        return items[0]

    def __init__(self, n: int):
        if n < 1:
            raise ValueError(f"Cannot scatter into {n} outputs; need at least 1.")
        self.n = n

    @property
    def _function(self) -> Callable[[Any], Any]:
        """
        A recipe declaring exactly one output receives the whole return value, so a
        1-wide scatter must return the element rather than a 1-tuple.
        """
        return self.iterable_to_output if self.n == 1 else self.iterable_to_outputs

    @property
    def recipe(self) -> atomic_recipe.AtomicRecipe:
        return atomic_recipe.AtomicRecipe(
            reference=base_models.PythonReference(
                info=versions.VersionInfo.of(self._function),
                restricted_input_kinds={
                    self.input_label: base_models.RestrictedParamKind.POSITIONAL_ONLY
                },
            ),
            inputs=[self.input_label],
            outputs=[self.output_label(i) for i in range(self.n)],
        )


class TransformNto1:
    """
    Gather ``n`` input ports into one list output.

    Attributes:
        output_label: The name of the sole output port.

    Args:
        n: How many input ports to gather. Zero is legal and yields an empty list,
            which is what an empty iterated input needs.
    """

    output_label: ClassVar[base_models.Label] = "output_0"

    @staticmethod
    def input_label(i: int) -> base_models.Label:
        return f"item_{i}"

    @staticmethod
    def inputs_to_list(*items):
        return list(items)

    def __init__(self, n: int):
        if n < 0:
            raise ValueError(f"Cannot gather {n} inputs; need at least 0.")
        self.n = n

    @property
    def recipe(self) -> atomic_recipe.AtomicRecipe:
        return atomic_recipe.AtomicRecipe(
            reference=base_models.PythonReference(
                info=versions.VersionInfo.of(self.inputs_to_list),
                restricted_input_kinds={
                    self.input_label(i): base_models.RestrictedParamKind.POSITIONAL_ONLY
                    for i in range(self.n)
                },
            ),
            inputs=[self.input_label(i) for i in range(self.n)],
            outputs=[self.output_label],
        )
