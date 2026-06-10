from __future__ import annotations

import re
from collections.abc import Callable
from typing import TypeAlias, cast

from pyiron_snippets import versions

from flowrep import base_models, edge_models, subgraph_validation
from flowrep.compiler import flow_control, function
from flowrep.prospective import (
    atomic_recipe,
    union_types,
    workflow_recipe,
)

InResolverType: TypeAlias = Callable[[str], str]
_PortTypeAlias: TypeAlias = (
    edge_models.SourceHandle
    | edge_models.TargetHandle
    | edge_models.InputSource
    | edge_models.OutputTarget
)
_ResolverType: TypeAlias = Callable[[_PortTypeAlias], str]


def _module_and_path(info: versions.VersionInfo) -> tuple[str, str]:
    """Return (module_to_import, dotted_call_path) for a referenced node."""
    if info.is_local:
        raise ValueError(
            f"Cannot emit a call to a function defined in a local scope: "
            f"{info.fully_qualified_name}"
        )
    return info.module, info.fully_qualified_name


def render_call(
    call_path: str, node: union_types.RecipeDiscrimination, in_resolver: InResolverType
) -> str:
    """Render a call expression. `in_resolver(port)` returns a symbol or raises."""
    if reference := getattr(node, "reference", None):
        restricted = reference.restricted_input_kinds
    else:
        restricted = {}
    positional: list[str] = []
    keyword: list[str] = []
    skipped_positional_only: list[str] = []
    for port in node.inputs:
        is_positional_only = (
            restricted.get(port) == base_models.RestrictedParamKind.POSITIONAL_ONLY
        )
        try:
            sym = in_resolver(port)
        except KeyError:  # unsourced defaulted input -> rely on the default
            if is_positional_only:
                skipped_positional_only.append(port)
            continue
        if is_positional_only:
            if skipped_positional_only:
                raise ValueError(
                    f"Cannot render a call to {call_path!r}: positional-only "
                    f"parameter(s) {skipped_positional_only} are unsourced but the "
                    f"later positional-only parameter {port!r} is sourced. There is "
                    f"no valid positional-call form, because positional-only "
                    f"parameters cannot be supplied by keyword."
                )
            positional.append(sym)
        else:
            keyword.append(f"{port}={sym}")
    return f"{call_path}({', '.join(positional + keyword)})"


def node_call_path(
    node: union_types.RecipeDiscrimination,
    label: str,
    emitter: function.Emitter,
    alloc: function.NameAllocator,
) -> str | None:
    """Resolve the call path for a node that emits as a single call expression.

    Returns the dotted call path for a referenced node (adding its import) or the
    local function name for a reference-free ``WorkflowRecipe`` (emitting a nested
    def). Returns ``None`` for flow controls, which cannot be expressed as a single
    call.
    """
    if isinstance(node, atomic_recipe.AtomicRecipe):
        return _set_call_path_from_info(node.reference.info, emitter.module_imports)
    elif isinstance(node, workflow_recipe.WorkflowRecipe):
        if reference := getattr(node, "reference", None):
            return _set_call_path_from_info(reference.info, emitter.module_imports)
        return function.emit_nested_workflow_node(node, label, emitter, alloc)
    elif isinstance(node, flow_control.FLOW_CONTROL_TYPES):
        return None
    else:  # pragma: no cover - all concrete flow control types are handled above
        raise ValueError(f"Unexpected node type: {type(node).__name__}")


def _set_call_path_from_info(
    info: versions.VersionInfo, module_imports: set[str]
) -> str:
    module, call_path = _module_and_path(info)
    module_imports.add(module)
    return call_path


def _topological_nodes(
    recipe: workflow_recipe.WorkflowRecipe,
) -> list[str]:
    """Node labels in dependency order. Preserves insertion order when already valid."""
    return subgraph_validation.topological_sort(
        recipe.nodes,
        [(source.node, target.node) for target, source in recipe.edges.items()],
        cycle_message="Recipe nodes contain a cycle; cannot serialise.",
    )


def _flow_control_input_requirements(
    recipe: workflow_recipe.WorkflowRecipe,
) -> dict[tuple[str, str], str]:
    """Required source-symbol names imposed by flow-control node inputs.

    A flow-control node's input port name is derived by the forward parser from
    the *enclosing symbol* feeding it. To round-trip those port names, every
    flow-control input fed by a *sibling* node must force that sibling's output to
    be named after the port. Inputs fed directly from a workflow input (an
    :class:`InputSource`) need no requirement -- the parameter already carries the
    matching name. Referenced and nested-workflow nodes impose no requirement,
    since their calls pass arguments by keyword regardless of the symbol names.
    """

    requirements: dict[tuple[str, str], str] = {}
    for label, node in recipe.nodes.items():
        if not isinstance(node, flow_control.FLOW_CONTROL_TYPES):
            continue
        for port in node.inputs:
            target = edge_models.TargetHandle(node=label, port=port)
            source = recipe.edges.get(target)
            if isinstance(source, edge_models.SourceHandle):
                requirements[(source.node, source.port)] = port
    return requirements


def _allocate_outputs(
    node: union_types.RecipeDiscrimination,
    label: str,
    produced: dict[tuple[str, str], str],
    required_by_handle: dict[tuple[str, str], str],
    alloc: function.NameAllocator,
) -> list[str]:
    """Allocate output symbols for a node, respecting any required names."""
    lhs_syms = []
    for port in node.outputs:
        handle = (label, port)
        name = required_by_handle.get(handle) or alloc.fresh(
            _output_name_suggestion(label, port, len(node.outputs))
        )
        produced[handle] = name
        lhs_syms.append(name)
    return lhs_syms


def _mk_resolver(
    label: str,
    recipe: workflow_recipe.WorkflowRecipe,
    resolve: _ResolverType,
) -> InResolverType:
    """
    Build an in_resolver closure for a given node label in the recipe.

    Raises:
        KeyError: if the node is not a target of input or peer edges.
    """

    def in_resolver(port: str) -> str:
        target = edge_models.TargetHandle(node=label, port=port)
        if target in recipe.input_edges:
            return resolve(recipe.input_edges[target])
        if target in recipe.edges:
            return resolve(recipe.edges[target])
        raise KeyError(f"No input or peer edge for {target.serialize()}")

    return in_resolver


def emit_workflow_body(
    recipe: workflow_recipe.WorkflowRecipe,
    in_syms: dict[str, str],
    required_out_syms: dict[str, str],
    emitter: function.Emitter,
    alloc: function.NameAllocator,
) -> tuple[list[str], dict[str, str]]:
    produced: dict[tuple[str, str], str] = {}
    lines: list[str] = []

    def resolve(source) -> str:
        if isinstance(source, edge_models.InputSource):
            return in_syms[source.port]
        return produced[(source.node, source.port)]

    # Map (node, port) -> required symbol name for outputs the parent pins.
    required_by_handle: dict[tuple[str, str], str] = {}
    for out_port, sym in required_out_syms.items():
        src = recipe.output_edges[edge_models.OutputTarget(port=out_port)]
        if isinstance(src, edge_models.SourceHandle):
            handle = (src.node, src.port)
            if handle in required_by_handle and required_by_handle[handle] != sym:
                raise ValueError(
                    f"Output port '{out_port}' and another output both source from "
                    f"{src.serialize()} but need different names "
                    f"({required_by_handle[handle]} vs {sym}); this cannot be emitted "
                    f"as an assignment."
                )
            required_by_handle[handle] = sym

    # Flow-control nodes derive their input port names from the enclosing symbols
    # feeding them, so each such source must be named after the port for the port
    # names (and while-loop reassignments) to round-trip.
    for handle, name in _flow_control_input_requirements(recipe).items():
        if (  # pragma: no cover - twin of the output-edge guard above; only a
            # hand-built recipe (one a parser never emits) can name a source for
            # both a flow-control input and an output differently. See
            # TestGuardsAndEdgeCases.test_alias_conflict_raises for the same class.
            handle in required_by_handle
            and required_by_handle[handle] != name
        ):
            raise ValueError(
                f"Source {handle} must be named both '{required_by_handle[handle]}' "
                f"and '{name}'; this cannot be emitted as an assignment."
            )
        required_by_handle[handle] = name

    for label in _topological_nodes(recipe):
        node = recipe.nodes[label]

        call_path = node_call_path(node, label, emitter, alloc)
        if call_path is not None:
            lhs_syms = _allocate_outputs(
                node, label, produced, required_by_handle, alloc
            )
            in_resolver = _mk_resolver(label, recipe, resolve)
            lines.append(
                f"{', '.join(lhs_syms)} = {render_call(call_path, node, in_resolver)}"
            )
        else:  # None branch indicates flow controller node
            in_resolver = _mk_resolver(label, recipe, resolve)
            lines.extend(
                flow_control.emit_flow_control(
                    cast(flow_control.FlowControlRecipeAlias, node),
                    label,
                    in_resolver,
                    produced,
                    required_by_handle,
                    emitter,
                    alloc,
                )
            )

    out_syms: dict[str, str] = {}
    for out_port in recipe.outputs:
        out_syms[out_port] = resolve(
            recipe.output_edges[edge_models.OutputTarget(port=out_port)]
        )
    return lines, out_syms


def emit_body(
    recipe: union_types.RecipeDiscrimination,
    label: str,
    in_syms: dict[str, str],
    required: dict[str, str],
    emitter: function.Emitter,
    alloc: function.NameAllocator,
) -> tuple[list[str], dict[str, str]]:
    """Emit a branch/loop body that may be a workflow subgraph or a single node.

    A reference-free ``WorkflowRecipe`` is inlined as a subgraph (as before); any
    other node (atomic, or a referenced workflow) is emitted as one assignment.
    """
    if isinstance(recipe, workflow_recipe.WorkflowRecipe):
        return emit_workflow_body(recipe, in_syms, required, emitter, alloc)
    if isinstance(recipe, flow_control.FLOW_CONTROL_TYPES):
        return flow_control.emit_flow_control_body(
            recipe, label, in_syms, required, emitter, alloc
        )
    return _emit_single_node_body(recipe, label, in_syms, required, alloc, emitter)


def _emit_single_node_body(
    node: atomic_recipe.AtomicRecipe,
    label: str,
    in_syms: dict[str, str],
    required: dict[str, str],
    alloc: function.NameAllocator,
    emitter: function.Emitter,
) -> tuple[list[str], dict[str, str]]:
    """Emit a single atomic node as one assignment line.

    Output ports are pinned to ``required`` symbols where given, else allocated
    fresh using a label-based hint. Inputs absent from ``in_syms`` (unwired
    defaults) are omitted from the call, exactly as ``_render_call`` already
    does for ``None`` resolutions.
    """

    def in_resolver(port: str) -> str:
        return in_syms[port]

    out_syms: dict[str, str] = {}
    lhs_syms: list[str] = []
    for port in node.outputs:
        name = required.get(port) or alloc.fresh(
            _output_name_suggestion(label, port, len(node.outputs))
        )
        out_syms[port] = name
        lhs_syms.append(name)

    call_path = _set_call_path_from_info(node.reference.info, emitter.module_imports)
    line = f"{', '.join(lhs_syms)} = {render_call(call_path, node, in_resolver)}"
    return [line], out_syms


def label_base(label: str) -> str:
    """Strip the trailing '_N' numeric suffix that label_helpers.unique_suffix appended.

    When re-parsing, the workflow_parser calls
    ``unique_suffix(function.__name__, existing_labels)`` to re-derive the node
    label.  If the nested def's ``__name__`` is the *base* (pre-suffix) of the
    original label, unique_suffix regenerates the same label on round-trip.

    Examples:
        'my_add_0'  → 'my_add'
        'for_each_2' → 'for_each'
        'plain'     → 'plain'  (no numeric suffix; returned as-is)
    """
    m = re.fullmatch(r"^(.+)_(\d+)$", label)
    return m.group(1) if m else label


def _output_name_suggestion(label: str, port: str, n_outputs: int) -> str:
    """Symbol-name hint for a call assignment, derived from the node label.

    Single-output nodes use the label base; multi-output nodes disambiguate per
    port. Pinned (required) names still take precedence at the call site.
    """
    base = label_base(label)
    return base if n_outputs == 1 else f"{base}_{port}"
