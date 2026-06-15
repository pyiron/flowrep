"""
Flowrep recipes represent a class-view of how data can be processed via a workflow.

In this module, we provide a prototypical data structure for retrospective,
instance-view workflows, which can be mutated as they are executed to be enriched with
data.

Intended to be a common export- and communication-format for WfMS of instance-views.

Unlike the recipes, no goal is made to provide easy serialization, and these data
structures natively hold complex python objects.
"""

from __future__ import annotations

import abc
import dataclasses
import inspect
import types
from collections.abc import Callable, MutableMapping
from typing import Any, Generic, Self, TypeVar, get_args, get_origin, get_type_hints

from pyiron_snippets import retrieve, singleton

from flowrep import base_models, edge_models
from flowrep.prospective import (
    atomic_recipe,
    for_recipe,
    if_recipe,
    try_recipe,
    union_types,
    while_recipe,
    workflow_recipe,
)


class NotData(metaclass=singleton.Singleton):
    """
    This class exists purely to initialize data channel values where no default value
    is provided; it lets the channel know that it has _no data in it_ and thus should
    not identify as ready.
    """

    @classmethod
    def __repr__(cls):
        # We use the class directly (not instances of it) where there is not yet data
        # So give it a decent repr, even as just a class
        return "NOT_DATA"

    def __reduce__(self):
        return "NOT_DATA"

    def __bool__(self):
        return False


NOT_DATA = NotData()


@dataclasses.dataclass(frozen=False)
class _DataPort:
    value: object | NotData = NOT_DATA
    annotation: Any | None = None


@dataclasses.dataclass(frozen=False)
class InputDataPort(_DataPort):
    default: object | NotData = NOT_DATA

    def get_data(self) -> object | NotData:
        """A shortcut for falling back on the default"""
        return self.default if self.value is NOT_DATA else self.value


@dataclasses.dataclass(frozen=False)
class OutputDataPort(_DataPort): ...


InputDataPorts = MutableMapping[base_models.Label, InputDataPort]
OutputDataPorts = MutableMapping[base_models.Label, OutputDataPort]

RecipeType = TypeVar("RecipeType", bound=base_models.NodeRecipe)


@dataclasses.dataclass(frozen=False)
class NodeData(Generic[RecipeType], abc.ABC):
    recipe: RecipeType
    input_ports: InputDataPorts
    output_ports: OutputDataPorts

    @classmethod
    @abc.abstractmethod
    def from_recipe(cls, recipe: RecipeType) -> Self: ...


def recipe2data(
    recipe: union_types.RecipeDiscrimination, allow_variadic_inputs: bool = True
) -> NodeData:
    """
    Convert a prospective flowrep recipe object to a retrospective flowrep data object.
    The data object is "empty" in that none of the actual data values will be filled;
    This is is an empty-vessel creator to be filled, e.g. by a WfMS run of the recipe.
    """
    match recipe:
        case atomic_recipe.AtomicRecipe():
            return AtomicData.from_recipe(
                recipe, allow_variadic_inputs=allow_variadic_inputs
            )
        case for_recipe.ForEachRecipe():
            return ForEachData.from_recipe(recipe)
        case if_recipe.IfRecipe():
            return IfData.from_recipe(recipe)
        case try_recipe.TryRecipe():
            return TryData.from_recipe(recipe)
        case while_recipe.WhileRecipe():
            return WhileData.from_recipe(recipe)
        case workflow_recipe.WorkflowRecipe():
            return DagData.from_recipe(
                recipe, allow_variadic_inputs=allow_variadic_inputs
            )
        case _:
            raise TypeError(f"Unrecognized recipe type {recipe}")


@dataclasses.dataclass(frozen=False)
class AtomicData(NodeData[atomic_recipe.AtomicRecipe]):
    function: Callable

    @classmethod
    def from_recipe(
        cls, recipe: atomic_recipe.AtomicRecipe, allow_variadic_inputs: bool = True
    ) -> AtomicData:
        function, input_ports, output_ports = _parse_function(
            recipe.reference.info.fully_qualified_name,
            recipe.inputs,
            recipe.outputs,
            recipe.unpack_mode,
            allow_variadic_inputs=allow_variadic_inputs,
        )
        return AtomicData(
            recipe=recipe,
            input_ports=dict(input_ports),
            output_ports=dict(output_ports),
            function=function,
        )


@dataclasses.dataclass(frozen=False)
class CompositeData(NodeData, Generic[RecipeType], abc.ABC):
    nodes: MutableMapping[base_models.Label, NodeData]
    input_edges: edge_models.InputEdges
    edges: edge_models.Edges
    output_edges: edge_models.OutputEdges


@dataclasses.dataclass(frozen=False)
class DagData(CompositeData[workflow_recipe.WorkflowRecipe]):
    @classmethod
    def from_recipe(
        cls, recipe: workflow_recipe.WorkflowRecipe, allow_variadic_inputs: bool = True
    ) -> DagData:
        if recipe.reference:
            function, input_ports, output_ports = _parse_function(
                recipe.reference.info.fully_qualified_name,
                recipe.inputs,
                recipe.outputs,
                allow_variadic_inputs=allow_variadic_inputs,
            )
        else:
            input_ports = {label: InputDataPort() for label in recipe.inputs}
            output_ports = {label: OutputDataPort() for label in recipe.outputs}
        nodes = {
            label: recipe2data(child, allow_variadic_inputs=allow_variadic_inputs)
            for label, child in recipe.nodes.items()
        }
        return DagData(
            recipe=recipe,
            input_ports=dict(input_ports),
            output_ports=dict(output_ports),
            nodes=dict(nodes),
            input_edges=dict(recipe.input_edges),
            edges=dict(recipe.edges),
            output_edges=dict(recipe.output_edges),
        )

    # TODO: add/remove_node/edge/input/output methods, each guarded that they are
    #   unavailable if the underlying recipe has a reference, and otherwise mutatiting
    #   the underlying recipe at the same time


@dataclasses.dataclass(frozen=False)
class FlowControlData(CompositeData, Generic[RecipeType]):

    @classmethod
    def from_recipe(
        cls,
        recipe: RecipeType,
    ) -> Self:
        """
        Flow control nodes are composite with dynamic bodies; WfMS must populate the
        nodes and edges at runtime according to recipe execution.
        """
        return cls(
            recipe=recipe,
            input_ports=dict({label: InputDataPort() for label in recipe.inputs}),
            output_ports=dict({label: OutputDataPort() for label in recipe.outputs}),
            nodes=dict(),
            input_edges={},
            edges={},
            output_edges={},
        )


@dataclasses.dataclass(frozen=False)
class ForEachData(FlowControlData[for_recipe.ForEachRecipe]): ...


@dataclasses.dataclass(frozen=False)
class IfData(FlowControlData[if_recipe.IfRecipe]): ...


@dataclasses.dataclass(frozen=False)
class TryData(FlowControlData[try_recipe.TryRecipe]): ...


@dataclasses.dataclass(frozen=False)
class WhileData(FlowControlData[while_recipe.WhileRecipe]): ...


def _parse_function(
    fully_qualified_name: str,
    inputs: list[str],
    outputs: list[str],
    unpack_mode: atomic_recipe.UnpackMode = atomic_recipe.UnpackMode.TUPLE,
    allow_variadic_inputs: bool = True,
) -> tuple[
    types.FunctionType,
    dict[base_models.Label, InputDataPort],
    dict[base_models.Label, OutputDataPort],
]:
    function = retrieve.import_from_string(fully_qualified_name)
    try:
        hints = get_type_hints(function, include_extras=True)
    except NameError as e:
        raise NameError(
            f"While parsing {fully_qualified_name!r} for recipe inputs {inputs} and "
            f"outputs {outputs}, could not find the symbol for at least one "
            f"annotation. This is likely due to forward referenced annotations. Please "
            f"cross reference the underlying name error ({str(e)!r}) and the function "
            f"being parsed, and locally make the necessary imports before re-parsing."
        ) from e
    sig = inspect.signature(function)

    variadics_in_sig = {
        name
        for name, p in sig.parameters.items()
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }
    accept_extra_inputs = allow_variadic_inputs and bool(variadics_in_sig)

    # --- input ports ---
    if colliding := variadics_in_sig.intersection(inputs):
        raise ValueError(
            f"Recipe inputs {colliding} collide with variadic parameter(s) of "
            f"{fully_qualified_name!r}; variadic parameter names cannot be used as "
            f"port names."
        )

    available = set(sig.parameters)
    missing = set(inputs) - available
    if missing and not accept_extra_inputs:
        raise ValueError(
            f"Requested inputs {missing} not found in signature of {fully_qualified_name!r}"
        )

    input_ports: dict[str, InputDataPort] = {}
    for name in inputs:
        param = sig.parameters.get(name)
        input_port = InputDataPort()
        if param is not None:
            input_port.annotation = hints.get(name, None)
            input_port.default = (
                param.default
                if param.default is not inspect.Parameter.empty
                else NOT_DATA
            )
        input_ports[name] = input_port

    # --- output ports ---
    return_annotation = hints.get("return", None)
    if unpack_mode == atomic_recipe.UnpackMode.NONE:
        output_ports = _parse_return_without_unpacking(return_annotation, outputs)
    elif unpack_mode == atomic_recipe.UnpackMode.TUPLE:
        output_ports = _parse_return_tuple(return_annotation, outputs)
    elif unpack_mode == atomic_recipe.UnpackMode.DATACLASS:
        output_ports = _parse_return_dataclass(return_annotation, outputs)

    return function, input_ports, output_ports


def _parse_return_without_unpacking(
    return_annotation, outputs: list[str]
) -> dict[str, OutputDataPort]:
    if len(outputs) != 1:  # pragma: no cover
        raise ValueError(
            f"Without return unpacking, only one output is allowed, but got {outputs}. "
            f"This should have been caught by the underlying recipe validation. Please "
            f"raise a GitHub issue reporting how you got here!"
        )
    return {outputs[0]: OutputDataPort(annotation=return_annotation)}


def _parse_return_tuple(
    return_annotation, outputs: list[str]
) -> dict[str, OutputDataPort]:
    output_ports: dict[str, OutputDataPort]
    if len(outputs) > 1:
        origin = get_origin(return_annotation)
        args = get_args(return_annotation)

        if return_annotation is not None:
            unpacking_hint = (
                f"To collect the entire tuple in a single port use "
                f"{atomic_recipe.UnpackMode.NONE} unpacking mode."
            )

            if origin is not tuple:
                raise ValueError(
                    f"Multiple outputs {outputs} requested but return annotation "
                    f"{return_annotation!r} is not splittable -- only tuple return "
                    f"hints are splittable. {unpacking_hint}"
                )
            if len(args) != len(outputs):
                raise ValueError(
                    f"Output labels {outputs} (n={len(outputs)}) do not match "
                    f"length of return annotation {return_annotation} (n={len(args)}). "
                    f"Tuple return hint unpacking requires one hint element per output."
                    f" {unpacking_hint}"
                )
            output_ports = {
                label: OutputDataPort(annotation=annotation)
                for label, annotation in zip(outputs, args, strict=True)
            }
        else:
            output_ports = {label: OutputDataPort() for label in outputs}
    else:
        output_ports = {outputs[0]: OutputDataPort(annotation=return_annotation)}
    return output_ports


def _parse_return_dataclass(
    return_annotation, outputs: list[str]
) -> dict[str, OutputDataPort]:
    if not dataclasses.is_dataclass(return_annotation):  # pragma: no cover
        raise TypeError(
            f"Return annotation {return_annotation!r} is not a dataclass. This should "
            f"have been caught by the underlying recipe validation. Please raise a "
            f"GitHub issue reporting how you got here!"
        )

    fields = dataclasses.fields(return_annotation)
    if len(outputs) != len(fields):  # pragma: no cover
        raise ValueError(
            f"Return dataclass {return_annotation!r} has {len(fields)} fields, "
            f"{[f.name for f in fields]}, but {len(outputs)} outputs, {outputs} were "
            f"requested. This should have been caught by the underlying recipe "
            f"validation. Please raise a GitHub issue reporting how you got here!"
        )

    return {
        label: OutputDataPort(
            annotation=(field.type if field.type is not dataclasses.MISSING else None),
        )
        for label, field in zip(outputs, fields, strict=True)
    }
