from __future__ import annotations

from typing import TypeAlias

from pyiron_snippets import versions

from flowrep import edge_models
from flowrep.compiler import function, statements
from flowrep.nodes import (
    for_recipe,
    if_recipe,
    try_recipe,
    while_recipe,
)

# Recipe types that emit as a flow-control statement rather than a single call.
_FLOW_CONTROL_TYPES = (
    for_recipe.ForEachRecipe,
    if_recipe.IfRecipe,
    try_recipe.TryRecipe,
    while_recipe.WhileRecipe,
)

_FlowControlRecipeAlias: TypeAlias = (
    for_recipe.ForEachRecipe
    | if_recipe.IfRecipe
    | try_recipe.TryRecipe
    | while_recipe.WhileRecipe
)
_BranchingRecipeAlias: TypeAlias = if_recipe.IfRecipe | try_recipe.TryRecipe
_ConditionalRecipeAlias: TypeAlias = if_recipe.IfRecipe | while_recipe.WhileRecipe


def _guard_loop_variable_shadowing(
    node: for_recipe.ForEachRecipe, produced: dict[tuple[str, str], str]
) -> None:
    """Raise if a for-loop variable would shadow an enclosing produced symbol.

    Loop variable names are pinned to body-port names and cannot be renamed without
    breaking round-trip, so a same-named enclosing symbol (which only arises from a
    pin, since the allocator reserves loop variables) cannot be emitted safely.
    """
    shadowed = sorted(set(node.iterated_ports) & set(produced.values()))
    if shadowed:
        raise ValueError(
            f"For-loop variable(s) {shadowed} collide with an enclosing symbol of "
            f"the same name. The loop variable is pinned to the body port name and "
            f"cannot be renamed, so this recipe cannot be emitted as Python."
        )


def _emit_flow_control(
    node: _FlowControlRecipeAlias,
    label: str,
    in_resolver: statements._InResolverType,
    produced: dict[tuple[str, str], str],
    required_by_handle: dict[tuple[str, str], str],
    emitter: function._Emitter,
    alloc: function._NameAllocator,
) -> list[str]:
    """Dispatch a flow-control node to the appropriate emitter."""
    if isinstance(node, for_recipe.ForEachRecipe):
        _guard_loop_variable_shadowing(node, produced)
    required = {
        port: required_by_handle[(label, port)]
        for port in node.outputs
        if (label, port) in required_by_handle
    }
    # Allocate output symbols for all outputs, using required names where available.
    out_syms: dict[str, str] = {}
    for port in node.outputs:
        name = required.get(port) or alloc.fresh(port)
        produced[(label, port)] = name
        out_syms[port] = name

    if isinstance(node, for_recipe.ForEachRecipe):
        return _emit_for_each(node, in_resolver, out_syms, emitter, alloc)
    if isinstance(node, if_recipe.IfRecipe):
        return _emit_if(node, in_resolver, out_syms, emitter, alloc)
    if isinstance(node, while_recipe.WhileRecipe):
        # While outputs must use the port name (== loop variable name), not a fresh name.
        while_out_syms = {port: required.get(port, port) for port in node.outputs}
        for port, name in while_out_syms.items():
            produced[(label, port)] = name
        return _emit_while(node, in_resolver, while_out_syms, emitter, alloc)
    if isinstance(node, try_recipe.TryRecipe):
        return _emit_try(node, in_resolver, out_syms, emitter, alloc)
    raise NotImplementedError(  # pragma: no cover - all four flow-control types are
        # handled above; nothing else reaches this dispatch.
        f"Flow control {type(node).__name__} not yet supported."
    )


def _emit_for_each(
    node: for_recipe.ForEachRecipe,
    in_resolver: statements._InResolverType,
    out_syms: dict[str, str],
    emitter: function._Emitter,
    alloc: function._NameAllocator,
) -> list[str]:
    """Emit a for-each loop as Python source lines."""
    # 1. Accumulator declarations, named by output port.
    lines = [f"{out_syms[port]} = []" for port in node.outputs]

    body_label = node.body_node.label
    body = node.body_node.node

    def collection_symbol(body_port: str) -> str:
        """Return the enclosing-scope symbol that feeds a body port."""
        src = node.input_edges[
            edge_models.TargetHandle(node=body_label, port=body_port)
        ]
        return in_resolver(src.port)

    # 2. Build iteration headers.
    headers: list[str] = []
    for body_port in node.nested_ports:
        headers.append(f"for {body_port} in {collection_symbol(body_port)}:")
    if node.zipped_ports:
        vars_ = ", ".join(node.zipped_ports)
        cols = ", ".join(collection_symbol(p) for p in node.zipped_ports)
        headers.append(f"for {vars_} in zip({cols}):")

    # 3. Body input symbols: iterated ports use the loop variable; broadcast use outer.
    body_in_syms: dict[str, str] = {}
    for body_port in body.inputs:
        if body_port in node.iterated_ports:
            body_in_syms[body_port] = body_port  # iteration variable (same name)
        else:
            target = edge_models.TargetHandle(node=body_label, port=body_port)
            if target in node.input_edges:  # unwired defaulted inputs are omitted
                body_in_syms[body_port] = collection_symbol(body_port)

    # Pin each body output to a symbol matching its port name, so the re-parser
    # reconstructs the same port names on round-trip.
    body_required = {port: port for port in body.outputs}
    body_lines, body_out = statements._emit_body(
        body, body_label, body_in_syms, body_required, emitter, alloc
    )

    # 4. Append each body output to its accumulator.
    append_lines: list[str] = []
    for port in node.outputs:
        src = node.output_edges[edge_models.OutputTarget(port=port)]
        if isinstance(src, edge_models.SourceHandle):
            appended = body_out[src.port]
        else:  # InputSource: a transferred (scattered) input collected per iteration.
            # src.port is the for-node input port; map it back to the iterated body
            # port whose loop variable holds that value each iteration.
            body_port = next(
                target.port
                for target, source in node.input_edges.items()
                if source.port == src.port and target.port in node.iterated_ports
            )
            appended = body_in_syms[body_port]
        append_lines.append(f"{out_syms[port]}.append({appended})")

    # 5. Assemble with nested indentation.
    # Each line will be indented by 4 spaces (one level) by FunctionBuilder.render.
    # Internal indentation is relative to that one level.
    indent = "    " * len(headers)
    rendered: list[str] = []
    for depth, header in enumerate(headers):
        rendered.append("    " * depth + header)
    for line in body_lines + append_lines:
        rendered.append(indent + line)
    return lines + rendered


def _render_condition(
    case_condition,
    node: _ConditionalRecipeAlias,
    in_resolver: statements._InResolverType,
    emitter: function._Emitter,
    alloc: function._NameAllocator,
) -> str:
    """Render a condition call as an inline expression (no assignment)."""
    cond_node = case_condition.node
    cond_label = case_condition.label
    call_path = statements._node_call_path(cond_node, cond_label, emitter, alloc)
    if call_path is None:
        raise NotImplementedError(
            f"Condition node '{cond_label}' is a {type(cond_node).__name__}, but "
            f"reverse-rendering requires a condition to be a single callable. A "
            f"flow-control recipe has no Python condition-expression form, so it "
            f"cannot be emitted as the condition of an if/while."
        )

    def cond_resolver(port: str) -> str:
        target = edge_models.TargetHandle(node=cond_label, port=port)
        return in_resolver(node.input_edges[target].port)

    return statements._render_call(call_path, cond_node, cond_resolver)


def _emit_branch(
    branch_label: str,
    branch_recipe,
    node: _BranchingRecipeAlias,
    in_resolver: statements._InResolverType,
    out_syms: dict[str, str],
    emitter: function._Emitter,
    alloc: function._NameAllocator,
) -> list[str]:
    """Inline a case/else branch body, pinning its outputs to the shared symbols."""
    body_in_syms: dict[str, str] = {}
    for port in branch_recipe.inputs:
        target = edge_models.TargetHandle(node=branch_label, port=port)
        if target in node.input_edges:  # unwired defaulted inputs are omitted
            body_in_syms[port] = in_resolver(node.input_edges[target].port)
    # Pin this branch's outputs to the shared names. The flow-control node's
    # prospective_output_edges map each flow output port to the (branch, branch
    # output port) that feeds it, so we key `required` by the *branch's* output
    # port name (which, for a bare atomic branch, differs from the flow output
    # port -- e.g. "output_0" vs "m"). For a workflow-wrapped branch the two
    # names coincide. Every prospective output target is one of the node's outputs
    # (validated on the recipe), so it is always present in out_syms.
    required: dict[str, str] = {}
    for flow_target, sources in node.prospective_output_edges.items():
        for source in sources:
            if source.node == branch_label:
                required[source.port] = out_syms[flow_target.port]
    body_lines, _ = statements._emit_body(
        branch_recipe, branch_label, body_in_syms, required, emitter, alloc
    )
    return body_lines or ["pass"]


def _emit_if(
    node: if_recipe.IfRecipe,
    in_resolver: statements._InResolverType,
    out_syms: dict[str, str],
    emitter: function._Emitter,
    alloc: function._NameAllocator,
) -> list[str]:
    """Emit an if/elif/else block as Python source lines."""
    lines: list[str] = []
    for idx, case in enumerate(node.cases):
        cond_expr = _render_condition(case.condition, node, in_resolver, emitter, alloc)
        keyword = "if" if idx == 0 else "elif"
        lines.append(f"{keyword} {cond_expr}:")
        body = _emit_branch(
            case.body.label,
            case.body.node,
            node,
            in_resolver,
            out_syms,
            emitter,
            alloc,
        )
        lines.extend("    " + line for line in body)
    if node.else_case is not None:
        lines.append("else:")
        body = _emit_branch(
            node.else_case.label,
            node.else_case.node,
            node,
            in_resolver,
            out_syms,
            emitter,
            alloc,
        )
        lines.extend("    " + line for line in body)
    return lines


def _exception_name(info: versions.VersionInfo, module_imports: set[str]) -> str:
    """Return the Python name to use for an exception type in an except clause.

    For builtins (module == 'builtins'), returns the bare qualname and adds no
    import.  For anything else, adds the module to the module_imports
    set and returns a fully qualified importable string.
    """
    if info.module != "builtins":
        module_imports.add(info.module)
    return info.findable_at


def _emit_try(
    node: try_recipe.TryRecipe,
    in_resolver: statements._InResolverType,
    out_syms: dict[str, str],
    emitter: function._Emitter,
    alloc: function._NameAllocator,
) -> list[str]:
    """Emit a try/except block as Python source lines."""
    # try body
    try_lines = _emit_branch(
        node.try_node.label,
        node.try_node.node,
        node,
        in_resolver,
        out_syms,
        emitter,
        alloc,
    )
    lines = ["try:"]
    lines.extend("    " + line for line in (try_lines or ["pass"]))

    for case in node.exception_cases:
        names = [_exception_name(v, emitter.module_imports) for v in case.exceptions]
        if len(names) == 1:
            header = f"except {names[0]}:"
        else:
            header = f"except ({', '.join(names)}):"
        lines.append(header)
        body = _emit_branch(
            case.body.label,
            case.body.node,
            node,
            in_resolver,
            out_syms,
            emitter,
            alloc,
        )
        lines.extend("    " + line for line in (body or ["pass"]))
    return lines


def _emit_while(
    node: while_recipe.WhileRecipe,
    in_resolver: statements._InResolverType,
    out_syms: dict[str, str],
    emitter: function._Emitter,
    alloc: function._NameAllocator,
) -> list[str]:
    """Emit a while loop as Python source lines."""
    case = node.case
    cond_expr = _render_condition(case.condition, node, in_resolver, emitter, alloc)

    body_label = case.body.label
    body = case.body.node

    # Map body inputs to enclosing-scope symbols (unwired defaults are omitted).
    body_in_syms: dict[str, str] = {}
    for port in body.inputs:
        target = edge_models.TargetHandle(node=body_label, port=port)
        if target in node.input_edges:
            body_in_syms[port] = in_resolver(node.input_edges[target].port)

    # Pin each body output to the loop variable it feeds (output_edges map the
    # body's output ports to the while-node output ports == loop variables), so the
    # variable is reassigned in-place each iteration and the loop terminates.
    required = {
        source.port: out_syms[target.port]
        for target, source in node.output_edges.items()
    }
    body_lines, _ = statements._emit_body(
        body, body_label, body_in_syms, required, emitter, alloc
    )

    lines = [f"while {cond_expr}:"]
    lines.extend("    " + line for line in (body_lines or ["pass"]))
    return lines


def _emit_flow_control_body(
    node: _FlowControlRecipeAlias,
    label: str,
    in_syms: dict[str, str],
    required: dict[str, str],
    emitter: function._Emitter,
    alloc: function._NameAllocator,
) -> tuple[list[str], dict[str, str]]:
    """Inline a flow-control recipe that sits directly as a branch/loop body.

    Bridges ``_emit_body``'s ``(in_syms, required)`` view to the
    ``(in_resolver, produced, required_by_handle)`` view of :func:`_emit_flow_control`,
    then maps the produced output symbols back to ``out_syms`` for the caller.
    """

    def in_resolver(port: str) -> str:
        return in_syms[port]

    produced: dict[tuple[str, str], str] = {}
    required_by_handle = {(label, port): sym for port, sym in required.items()}
    lines = _emit_flow_control(
        node, label, in_resolver, produced, required_by_handle, emitter, alloc
    )
    out_syms = {port: produced[(label, port)] for port in node.outputs}
    return lines, out_syms
