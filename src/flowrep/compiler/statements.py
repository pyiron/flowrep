from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, TypeAlias, cast

from pyiron_snippets import versions

from flowrep import base_models, edge_models, subgraph_validation
from flowrep.compiler import flow_control, function, sugar
from flowrep.prospective import (
    atomic_recipe,
    constant_recipe,
    std,
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


def _identity_materialization(value: Any, emitter: function.Emitter) -> str:
    """Render an ``identity`` call wrapping a constant that cannot be inlined.

    A bare ``name = <literal>`` statement no longer re-parses (constants are only
    valid as call arguments), so a constant feeding a workflow output or a bare
    flow-control body is emitted as ``identity(x=<literal>)``. The
    ``flowrep.prospective.std`` import is hoisted via ``_set_call_path_from_info``.
    On re-parse this canonicalises to a std ``identity`` atomic node fed by an
    inlined constant argument, which then round-trips exactly.
    """
    call_path = _set_call_path_from_info(
        std.identity.flowrep_recipe.reference.info,  # type: ignore[attr-defined]
        emitter.module_imports,
    )
    return f"{call_path}(x={value!r})"


def _topological_nodes(
    recipe: workflow_recipe.WorkflowRecipe,
) -> list[str]:
    """Node labels in dependency order. Preserves insertion order when already valid."""
    return subgraph_validation.topological_sort(
        recipe.nodes,
        [(source.node, target.node) for target, source in recipe.edges.items()],
        cycle_message="Recipe nodes contain a cycle; cannot serialise.",
    )


def flow_control_input_requirements(
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

    # Constants feeding a workflow output cannot be inlined (a bare `return <literal>`
    # is not re-parseable, and a bare `sym = <literal>` statement is no longer
    # re-parseable either), so they are wrapped as `sym = identity(x=<literal>)` below
    # and resolved to that symbol everywhere.
    materialized_constants = {
        src.node
        for src in recipe.output_edges.values()
        if isinstance(src, edge_models.SourceHandle)
        and isinstance(recipe.nodes.get(src.node), constant_recipe.ConstantRecipe)
    }

    def resolve(source) -> str:
        if isinstance(source, edge_models.InputSource):
            return in_syms[source.port]
        source_node = recipe.nodes[source.node]
        if (
            isinstance(source_node, constant_recipe.ConstantRecipe)
            and source.node not in materialized_constants
        ):
            return repr(source_node.constant)
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
    # names (and while-loop reassignments) to round-trip. A constant peer feeding a
    # flow-control input is normal now (a literal condition argument injects
    # a constant peer routed through a synthetic flow-control input port). Such a
    # peer is never also a workflow-output source, so its required_by_handle entry
    # stays inert and it is inlined in the topo loop below -- which is why the
    # conflict branch immediately below remains unreachable in parser-produced
    # recipes.
    for handle, name in flow_control_input_requirements(recipe).items():
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

    def _obj_source(
        label: str,
    ) -> edge_models.SourceHandle | edge_models.InputSource | None:
        target = edge_models.TargetHandle(node=label, port=sugar.OBJ_PORT)
        return recipe.input_edges.get(target) or recipe.edges.get(target)

    def _obj_is_symbolic(label: str) -> bool:
        """True if the getattr's object resolves to a symbol we can write '.' after.

        An object fed by an *inlined* constant would sugar to nonsense (``5 .a``), so
        such a node keeps the plain-call emission. The parser cannot produce one.
        """
        source = _obj_source(label)
        if source is None:
            return False
        if isinstance(source, edge_models.InputSource):
            return True
        peer = recipe.nodes[source.node]
        if isinstance(peer, constant_recipe.ConstantRecipe):
            return source.node in materialized_constants
        return True

    # A recognised getattr node is emitted as attribute syntax -- `sym = obj.attr`
    # -- always as its own statement, never inlined into a consumer's argument list.
    #
    # Inlining would be prettier (`f(dc.a)`, `dc.a.val`) but it is not safe. The
    # parser injects a getattr node together with a `ConstantRecipe` peer carrying the
    # attribute name, and constant labels come from one shared `constant_N` counter
    # over the whole workflow. Inlining moves a getattr's creation to its consumer's
    # statement, which reorders that name-constant relative to every *other* constant
    # in the workflow (e.g. a literal call argument) and permutes the counter. The
    # recipe would still execute identically, but it would not re-parse to an equal
    # one. Emitting one statement per getattr, in topological order, reproduces the
    # parser's injection order exactly, so the round trip is exact.
    #
    # Materialising also gives the efficiency guarantee for free: one node, one
    # statement, one symbol -- a getattr consumed by several nodes stays a single
    # node on re-parse rather than being duplicated into each consumer.
    sugared: dict[str, str] = {
        label: attr_name
        for label, node in recipe.nodes.items()
        if sugar.is_std_getattr(node)
        and (attr_name := sugar.attribute_name(label, recipe)) is not None
        and _obj_is_symbolic(label)
    }

    for label in _topological_nodes(recipe):
        node = recipe.nodes[label]
        if isinstance(node, constant_recipe.ConstantRecipe):
            if label not in materialized_constants:
                continue  # inline at the consumer call site via `resolve`
            constant_label = constant_recipe.ConstantRecipe.std_label
            name = required_by_handle.get((label, constant_label)) or alloc.fresh(
                _output_name_suggestion(label, constant_label, 1)
            )
            produced[(label, constant_label)] = name
            lines.append(
                f"{name} = {_identity_materialization(node.constant, emitter)}"
            )
            continue

        if label in sugared:
            name = required_by_handle.get((label, sugar.ATTR_PORT)) or alloc.fresh(
                _output_name_suggestion(label, sugar.ATTR_PORT, 1)
            )
            produced[(label, sugar.ATTR_PORT)] = name
            obj = resolve(_obj_source(label))
            lines.append(f"{name} = {obj}.{sugared[label]}")
            continue

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
    if isinstance(recipe, constant_recipe.ConstantRecipe):
        constant_label = constant_recipe.ConstantRecipe.std_label
        name = required.get(constant_label) or alloc.fresh(
            _output_name_suggestion(label, constant_label, 1)
        )
        return (
            [f"{name} = {_identity_materialization(recipe.constant, emitter)}"],
            {constant_label: name},
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
