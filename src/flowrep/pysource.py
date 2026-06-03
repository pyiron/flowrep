from __future__ import annotations

import dataclasses
import inspect
import linecache
import re
import sys
import textwrap
import types
import typing
from typing import Annotated, Any, cast, get_args, get_origin

from flowrep import base_models, edge_models, retrospective
from flowrep.nodes import (
    for_recipe,
    if_recipe,
    try_recipe,
    while_recipe,
    workflow_recipe,
)
from flowrep.parsers import label_helpers

# Counter used to make each generated source's linecache key unique.
_GENERATED_COUNTER = 0


def _next_generated_filename(name: str) -> str:
    global _GENERATED_COUNTER
    _GENERATED_COUNTER += 1
    return f"<flowrep_generated_{name}_{_GENERATED_COUNTER}>"


@dataclasses.dataclass
class RenderedSource:
    """Executable Python source plus the live namespace it must be exec'd against."""

    source: str
    namespace: dict[str, Any]
    function_name: str

    def build(self) -> types.FunctionType:
        """exec the source against a copy of the namespace and return the function."""
        fname = _next_generated_filename(self.function_name)
        # Register source with linecache so inspect.getsource works on the result.
        lines = [line + "\n" for line in self.source.splitlines()]
        linecache.cache[fname] = (len(self.source), None, lines, fname)
        # Create a synthetic module and register it in sys.modules BEFORE exec.
        # This ensures inspect.getmodule() resolves to our module for both the
        # nested-def decoration at exec time and the target re-parse by parse_workflow.
        # The module name is derived from the filename (strip angle brackets).
        mod_name = fname.strip("<>")
        module = types.ModuleType(mod_name)
        module.__file__ = fname
        module.__dict__.update({"typing": typing, **self.namespace})
        sys.modules[mod_name] = module
        code = compile(self.source, fname, "exec")
        exec(code, module.__dict__)  # noqa: S102 - controlled codegen, not user input
        return module.__dict__[self.function_name]


class _NameAllocator:
    """Mints unique, valid Python identifiers within a single function namespace."""

    def __init__(self) -> None:
        self._used: set[str] = set()

    def fresh(self, hint: str) -> str:
        base = hint if base_models.is_valid_label(hint) else "_v"
        candidate = base if base not in self._used else None
        if candidate is None:
            candidate = label_helpers.unique_suffix(base, self._used)
        self._used.add(candidate)
        return candidate

    def reserve(self, name: str) -> str:
        self._used.add(name)
        return name


@dataclasses.dataclass
class _Emitter:
    namespace: dict[str, Any] = dataclasses.field(default_factory=dict)
    nested_defs: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class FunctionBuilder:
    name: str
    params: list[str] = dataclasses.field(default_factory=list)
    body_lines: list[str] = dataclasses.field(default_factory=list)
    return_annotation: str | None = None
    return_symbols: list[str] = dataclasses.field(default_factory=list)

    def render(self) -> str:
        sig = ", ".join(self.params)
        # Every generated function is decorated so that exec'ing the source attaches
        # a `.flowrep_recipe`; this also validates the body parses as a workflow.
        header = "@flowrep.workflow\n"
        header += f"def {self.name}({sig})"
        if self.return_annotation is not None:
            header += f" -> {self.return_annotation}"
        header += ":"
        body = self.body_lines or ["pass"]
        ret = ""
        if self.return_symbols:
            ret = "    return " + ", ".join(self.return_symbols) + "\n"
        indented = "".join(f"    {line}\n" for line in body)
        return header + "\n" + indented + ret


def _has_reference(node: Any) -> bool:
    return getattr(node, "reference", None) is not None


def _module_and_path(node: Any) -> tuple[str, str]:
    """Return (module_to_import, dotted_call_path) for a referenced node."""
    info = node.reference.info
    module = info.module
    qualname = info.qualname
    if "<locals>" in qualname:
        raise ValueError(
            f"Cannot emit a call to a function defined in a local scope: "
            f"{module}.{qualname}"
        )
    return module, f"{module}.{qualname}"


def _render_call(call_path: str, node: Any, in_resolver) -> str:
    """Render a call expression. `in_resolver(port)` returns a symbol or None."""
    restricted = node.reference.restricted_input_kinds if _has_reference(node) else {}
    positional: list[str] = []
    keyword: list[str] = []
    for port in node.inputs:
        sym = in_resolver(port)
        if sym is None:  # unsourced defaulted input -> rely on the default
            continue
        if restricted.get(port) == base_models.RestrictedParamKind.POSITIONAL_ONLY:
            positional.append(sym)
        else:
            keyword.append(f"{port}={sym}")
    return f"{call_path}({', '.join(positional + keyword)})"


def _topological_nodes(
    recipe: workflow_recipe.WorkflowRecipe,
) -> list[str]:
    """Node labels in dependency order. Preserves insertion order when already valid."""
    successors: dict[str, list[str]] = {label: [] for label in recipe.nodes}
    indegree: dict[str, int] = {label: 0 for label in recipe.nodes}
    for target, source in recipe.edges.items():
        successors[source.node].append(target.node)
        indegree[target.node] += 1
    # Stable Kahn: prefer original insertion order among ready nodes.
    order = list(recipe.nodes)
    ready = [n for n in order if indegree[n] == 0]
    result: list[str] = []
    while ready:
        ready.sort(key=order.index)
        node = ready.pop(0)
        result.append(node)
        for succ in successors[node]:
            indegree[succ] -= 1
            if indegree[succ] == 0:
                ready.append(succ)
    if len(result) != len(recipe.nodes):
        raise ValueError("Recipe nodes contain a cycle; cannot serialise.")
    return result


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

    flow_control_types = (
        for_recipe.ForEachRecipe,
        if_recipe.IfRecipe,
        try_recipe.TryRecipe,
        while_recipe.WhileRecipe,
    )
    requirements: dict[tuple[str, str], str] = {}
    for label, node in recipe.nodes.items():
        if not isinstance(node, flow_control_types):
            continue
        for port in node.inputs:
            target = edge_models.TargetHandle(node=label, port=port)
            source = recipe.edges.get(target)
            if isinstance(source, edge_models.SourceHandle):
                requirements[(source.node, source.port)] = port
    return requirements


def _output_label_annotation(labels: list[str], type_refs: list[str]) -> str:
    """Build a return annotation that pins output port names via Annotated labels.

    Each output gets ``typing.Annotated[<type_ref>, {"label": <label>}]`` where
    ``type_ref`` is either ``"typing.Any"`` or the name of a namespace-bound type
    object (see :func:`_resolve_output_type_refs`).
    """

    def one(label: str, type_ref: str) -> str:
        return f'typing.Annotated[{type_ref}, {{"label": "{label}"}}]'

    if len(labels) == 1:
        return one(labels[0], type_refs[0])
    inner = ", ".join(
        one(label, ref) for label, ref in zip(labels, type_refs, strict=True)
    )
    return f"tuple[{inner}]"


def _strip_annotated(annotation: Any) -> Any:
    """Unwrap one layer of ``typing.Annotated`` to recover the underlying type."""
    if get_origin(annotation) is Annotated:
        return get_args(annotation)[0]
    return annotation


def _split_return_annotation(return_annotation: Any, n_outputs: int) -> list[Any]:
    """Split a function-level return annotation into one entry per output port.

    Returns a list of length ``n_outputs``; an entry is ``None`` when no usable
    type is available for that port. A single-output workflow uses the whole
    (Annotated-stripped) annotation; a multi-output workflow requires a matching
    ``tuple[...]`` annotation, otherwise every entry is ``None``.
    """
    empty = inspect.Signature.empty
    if return_annotation is empty or return_annotation is None:
        return [None] * n_outputs
    core = _strip_annotated(return_annotation)
    if n_outputs == 1:
        return [_strip_annotated(core)]
    if get_origin(core) is tuple:
        args = get_args(core)
        if len(args) == n_outputs:
            return [_strip_annotated(arg) for arg in args]
    return [None] * n_outputs


def _resolve_output_type_refs(
    signature: inspect.Signature | None,
    outputs: list[str],
    emitter: _Emitter,
) -> list[str]:
    """Return a source-level type reference for each output port.

    Real types are bound into ``emitter.namespace`` as ``_ann_ret_<port>`` and
    referenced by name (arbitrary type objects do not round-trip through repr).
    Ports without a usable type fall back to ``"typing.Any"``.
    """
    return_annotation = (
        signature.return_annotation
        if signature is not None
        else inspect.Signature.empty
    )
    raw = _split_return_annotation(return_annotation, len(outputs))
    refs: list[str] = []
    for port, annotation in zip(outputs, raw, strict=True):
        if annotation is None or annotation is Any:
            refs.append("typing.Any")
        else:
            name = f"_ann_ret_{port}"
            emitter.namespace[name] = annotation
            refs.append(name)
    return refs


def _allocate_outputs(
    node: Any,
    label: str,
    produced: dict[tuple[str, str], str],
    required_by_handle: dict[tuple[str, str], str],
    alloc: _NameAllocator,
) -> list[str]:
    """Allocate output symbols for a node, respecting any required names."""
    lhs_syms = []
    for port in node.outputs:
        handle = (label, port)
        name = required_by_handle.get(handle) or alloc.fresh(port)
        produced[handle] = name
        lhs_syms.append(name)
    return lhs_syms


def _mk_resolver(
    label: str,
    recipe: workflow_recipe.WorkflowRecipe,
    resolve: Any,
):
    """Build an in_resolver closure for a given node label in the recipe."""

    def in_resolver(port: str) -> str | None:
        target = edge_models.TargetHandle(node=label, port=port)
        if target in recipe.input_edges:
            return resolve(recipe.input_edges[target])
        if target in recipe.edges:
            return resolve(recipe.edges[target])
        return None  # unsourced defaulted input

    return in_resolver


def _emit_workflow_body(
    recipe: workflow_recipe.WorkflowRecipe,
    in_syms: dict[str, str],
    required_out_syms: dict[str, str],
    emitter: _Emitter,
    alloc: _NameAllocator,
    imports: set[str],
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

        if _has_reference(node):
            module, call_path = _module_and_path(node)
            imports.add(f"import {module}")
            lhs_syms = _allocate_outputs(
                node, label, produced, required_by_handle, alloc
            )
            in_resolver = _mk_resolver(label, recipe, resolve)
            lines.append(
                f"{', '.join(lhs_syms)} = {_render_call(call_path, node, in_resolver)}"
            )
        elif isinstance(node, workflow_recipe.WorkflowRecipe):
            call_path = _emit_nested_workflow_node(node, label, emitter, alloc)
            lhs_syms = _allocate_outputs(
                node, label, produced, required_by_handle, alloc
            )
            in_resolver = _mk_resolver(label, recipe, resolve)
            lines.append(
                f"{', '.join(lhs_syms)} = {_render_call(call_path, node, in_resolver)}"
            )
        else:
            in_resolver = _mk_resolver(label, recipe, resolve)
            lines.extend(
                _emit_flow_control(
                    node,
                    label,
                    in_resolver,
                    produced,
                    required_by_handle,
                    emitter,
                    alloc,
                    imports,
                )
            )

    out_syms: dict[str, str] = {}
    for out_port in recipe.outputs:
        out_syms[out_port] = resolve(
            recipe.output_edges[edge_models.OutputTarget(port=out_port)]
        )
    return lines, out_syms


def _emit_flow_control(
    node: Any,
    label: str,
    in_resolver: Any,
    produced: dict[tuple[str, str], str],
    required_by_handle: dict[tuple[str, str], str],
    emitter: _Emitter,
    alloc: _NameAllocator,
    imports: set[str],
) -> list[str]:
    """Dispatch a flow-control node to the appropriate emitter."""
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
        return _emit_for_each(node, in_resolver, out_syms, emitter, alloc, imports)
    if isinstance(node, if_recipe.IfRecipe):
        return _emit_if(node, in_resolver, out_syms, emitter, alloc, imports)
    if isinstance(node, while_recipe.WhileRecipe):
        # While outputs must use the port name (== loop variable name), not a fresh name.
        while_out_syms = {port: required.get(port, port) for port in node.outputs}
        for port, name in while_out_syms.items():
            produced[(label, port)] = name
        return _emit_while(node, in_resolver, while_out_syms, emitter, alloc, imports)
    if isinstance(node, try_recipe.TryRecipe):
        return _emit_try(node, in_resolver, out_syms, emitter, alloc, imports)
    raise NotImplementedError(  # pragma: no cover - all four flow-control types are
        # handled above; nothing else reaches this dispatch.
        f"Flow control {type(node).__name__} not yet supported."
    )


def _emit_for_each(
    node: Any,
    in_resolver: Any,
    out_syms: dict[str, str],
    emitter: _Emitter,
    alloc: _NameAllocator,
    imports: set[str],
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
            body_in_syms[body_port] = collection_symbol(body_port)

    # Pin each body output to a symbol matching its port name, so the re-parser
    # reconstructs the same port names on round-trip.
    body_required = {port: port for port in body.outputs}
    body_lines, body_out = _emit_workflow_body(
        body, body_in_syms, body_required, emitter, alloc, imports
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
    case_condition: Any,
    node: Any,
    in_resolver: Any,
    imports: set[str],
) -> str:
    """Render a condition call as an inline expression (no assignment)."""
    cond_node = case_condition.node
    cond_label = case_condition.label
    module, call_path = _module_and_path(cond_node)
    imports.add(f"import {module}")

    def cond_resolver(port: str) -> str | None:
        target = edge_models.TargetHandle(node=cond_label, port=port)
        if target in node.input_edges:
            return in_resolver(node.input_edges[target].port)
        return None

    return _render_call(call_path, cond_node, cond_resolver)


def _emit_branch(
    branch_label: str,
    branch_recipe: Any,
    node: Any,
    in_resolver: Any,
    out_syms: dict[str, str],
    emitter: _Emitter,
    alloc: _NameAllocator,
    imports: set[str],
) -> list[str]:
    """Inline a case/else branch body, pinning its outputs to the shared symbols."""
    body_in_syms: dict[str, str] = {}
    for port in branch_recipe.inputs:
        src = node.input_edges[edge_models.TargetHandle(node=branch_label, port=port)]
        body_in_syms[port] = in_resolver(src.port)
    # Pin only the outputs this branch actually produces to the shared names.
    required = {p: out_syms[p] for p in branch_recipe.outputs if p in out_syms}
    body_lines, _ = _emit_workflow_body(
        branch_recipe, body_in_syms, required, emitter, alloc, imports
    )
    return body_lines or ["pass"]


def _emit_if(
    node: Any,
    in_resolver: Any,
    out_syms: dict[str, str],
    emitter: _Emitter,
    alloc: _NameAllocator,
    imports: set[str],
) -> list[str]:
    """Emit an if/elif/else block as Python source lines."""
    lines: list[str] = []
    for idx, case in enumerate(node.cases):
        cond_expr = _render_condition(case.condition, node, in_resolver, imports)
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
            imports,
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
            imports,
        )
        lines.extend("    " + line for line in body)
    return lines


def _exception_name(version_info: Any, imports: set[str]) -> str:
    """Return the Python name to use for an exception type in an except clause.

    For builtins (module == 'builtins'), returns the bare qualname and adds no
    import.  For anything else, adds ``import {module}`` to the per-function
    imports set and returns ``{module}.{qualname}``.
    """
    module = version_info.module
    qualname = version_info.qualname
    if module == "builtins":
        return qualname
    imports.add(f"import {module}")
    return f"{module}.{qualname}"


def _emit_try(
    node: Any,
    in_resolver: Any,
    out_syms: dict[str, str],
    emitter: _Emitter,
    alloc: _NameAllocator,
    imports: set[str],
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
        imports,
    )
    lines = ["try:"]
    lines.extend("    " + line for line in (try_lines or ["pass"]))

    for case in node.exception_cases:
        names = [_exception_name(v, imports) for v in case.exceptions]
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
            imports,
        )
        lines.extend("    " + line for line in (body or ["pass"]))
    return lines


def _emit_while(
    node: Any,
    in_resolver: Any,
    out_syms: dict[str, str],
    emitter: _Emitter,
    alloc: _NameAllocator,
    imports: set[str],
) -> list[str]:
    """Emit a while loop as Python source lines."""
    case = node.case
    cond_expr = _render_condition(case.condition, node, in_resolver, imports)

    body_label = case.body.label
    body = case.body.node

    # Map body inputs to enclosing-scope symbols (all body ports come from while inputs).
    body_in_syms: dict[str, str] = {}
    for port in body.inputs:
        src = node.input_edges[edge_models.TargetHandle(node=body_label, port=port)]
        body_in_syms[port] = in_resolver(src.port)

    # Pin each body output to its own port name (== loop variable name) so the
    # variable is reassigned in-place each iteration and the loop terminates correctly.
    required = {port: port for port in body.outputs}
    body_lines, _ = _emit_workflow_body(
        body, body_in_syms, required, emitter, alloc, imports
    )

    lines = [f"while {cond_expr}:"]
    lines.extend("    " + line for line in (body_lines or ["pass"]))
    return lines


def _emit_workflow_function(
    recipe: workflow_recipe.WorkflowRecipe,
    name: str,
    emitter: _Emitter,
    signature: inspect.Signature | None = None,
) -> FunctionBuilder:
    alloc = _NameAllocator()
    in_syms = {port: alloc.reserve(port) for port in recipe.inputs}
    params = _render_params(recipe, signature, emitter)
    # Per-function import set: each function body gets its own import lines so that
    # nested defs are self-contained (the parser re-walks each body independently).
    fn_imports: set[str] = set()
    lines, out_syms = _emit_workflow_body(
        recipe, in_syms, {}, emitter, alloc, fn_imports
    )
    return_annotation = None
    if recipe.outputs:
        type_refs = _resolve_output_type_refs(signature, list(recipe.outputs), emitter)
        return_annotation = _output_label_annotation(list(recipe.outputs), type_refs)
    # Prepend sorted imports at the top of the function body. A docstring (the
    # recipe description) must come first of all so the parser's skip_docstring
    # recognises it and `description` round-trips.
    body_lines = sorted(fn_imports) + lines
    if recipe.description is not None:
        body_lines = [repr(recipe.description)] + body_lines
    builder = FunctionBuilder(
        name=name,
        params=params,
        body_lines=body_lines,
        return_annotation=return_annotation,
        return_symbols=[out_syms[p] for p in recipe.outputs],
    )
    return builder


def _label_base(label: str) -> str:
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


def _emit_nested_workflow_node(
    node: workflow_recipe.WorkflowRecipe,
    label: str,
    emitter: _Emitter,
    alloc: _NameAllocator,
) -> str:
    """Emit a nested @flowrep.workflow-decorated def for a reference-free workflow node.

    Returns the local function name (the call path for the enclosing body).
    The nested function is appended to emitter.nested_defs.

    The function is named after the label's base (pre-suffix) form so that
    workflow_parser.parse_workflow re-derives the same node label on round-trip
    via label_helpers.unique_suffix.
    """
    # Use the base of the label (strip _N suffix) so that re-parsing via
    # unique_suffix reconstructs the same original label.
    base = _label_base(label)
    fn_name = alloc.fresh(base)
    builder = _emit_workflow_function(node, fn_name, emitter, signature=None)
    # The @flowrep.workflow decorator is emitted by FunctionBuilder.render itself.
    emitter.nested_defs.append(builder.render())
    return fn_name


def _render_params(
    recipe: workflow_recipe.WorkflowRecipe,
    signature: inspect.Signature | None,
    emitter: _Emitter,
) -> list[str]:
    if signature is None:
        return list(recipe.inputs)

    params: list[str] = []
    seen_positional_only = False
    emitted_keyword_marker = False
    need_pos_only_marker = any(
        signature.parameters.get(p)
        and signature.parameters[p].kind == inspect.Parameter.POSITIONAL_ONLY
        for p in recipe.inputs
    )

    for port in recipe.inputs:
        param = signature.parameters.get(port)
        kind = param.kind if param else inspect.Parameter.POSITIONAL_OR_KEYWORD

        if kind == inspect.Parameter.POSITIONAL_ONLY:
            seen_positional_only = True
        else:
            if seen_positional_only and need_pos_only_marker:
                params.append("/")
                need_pos_only_marker = False  # only insert once
            if kind == inspect.Parameter.KEYWORD_ONLY and not emitted_keyword_marker:
                params.append("*")
                emitted_keyword_marker = True

        piece = port
        has_annotation = (
            param is not None and param.annotation is not inspect.Parameter.empty
        )
        has_default = param is not None and param.default is not inspect.Parameter.empty
        if has_annotation:
            annotation_name = f"_ann_{port}"
            emitter.namespace[annotation_name] = cast(
                inspect.Parameter, param
            ).annotation
            piece += f": {annotation_name}"
        if has_default:
            default_name = f"_default_{port}"
            emitter.namespace[default_name] = cast(inspect.Parameter, param).default
            # PEP 8 / black: spaces around '=' only when the parameter is annotated.
            piece += f" = {default_name}" if has_annotation else f"={default_name}"
        params.append(piece)

    # Edge case: the last param was positional-only and no marker emitted yet.
    if need_pos_only_marker and seen_positional_only and "/" not in params:
        params.append("/")
    return params


def recipe2python(
    function_name: base_models.Label,
    recipe: workflow_recipe.WorkflowRecipe,
    signature: inspect.Signature | None = None,
) -> RenderedSource:
    if recipe.reference is not None:
        raise ValueError(
            f"This recipe already has an underlying Python reference: "
            f"{recipe.reference}"
        )
    emitter = _Emitter()
    target = _emit_workflow_function(recipe, function_name, emitter, signature)
    # Per-node call imports are inlined into each function body by
    # _emit_workflow_function. The module-level preamble provides the deferred-
    # annotation flag (so return annotations are not evaluated at exec time) plus
    # `typing` (for the Annotated return labels) and `flowrep` (for the @workflow
    # decorator emitted on every function).
    preamble = textwrap.dedent("""\
        from __future__ import annotations
        
        import typing
        
        import flowrep
        """)
    nested = "\n".join(emitter.nested_defs)
    func_src = target.render()
    parts = [preamble]
    if nested:
        parts.append(nested)
    parts.append(func_src)
    source = "\n".join(parts) + "\n"
    return RenderedSource(
        source=source, namespace=emitter.namespace, function_name=function_name
    )


def dagdata2python(
    function_name: base_models.Label,
    dagdata: retrospective.DagData,
) -> RenderedSource:
    sig = _build_signature(dagdata.input_ports, dagdata.output_ports)
    # Strip the reference so recipe2python accepts the recipe.
    free_recipe = dagdata.recipe.model_copy(update={"reference": None})
    return recipe2python(function_name, free_recipe, sig)


def _build_signature(
    inputs: retrospective.InputDataPorts,
    outputs: retrospective.OutputDataPorts,
) -> inspect.Signature:
    params = []
    for name, port in inputs.items():
        default = (
            port.default
            if port.default is not retrospective.NOT_DATA
            else inspect.Parameter.empty
        )
        annotation = (
            port.annotation if port.annotation is not None else inspect.Parameter.empty
        )
        params.append(
            inspect.Parameter(
                name,
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=default,
                annotation=annotation,
            )
        )
    return inspect.Signature(
        parameters=params,
        return_annotation=_build_return_annotation(outputs),
    )


def _build_return_annotation(outputs: retrospective.OutputDataPorts) -> Any:
    """Encode per-output-port types as a single function-level return annotation.

    A single port yields its own annotation; multiple ports yield a
    ``tuple[...]`` so :func:`_split_return_annotation` can recover them. Returns
    ``inspect.Signature.empty`` when no port carries a type.
    """
    annotations = [port.annotation for port in outputs.values()]
    if all(annotation is None for annotation in annotations):
        return inspect.Signature.empty
    types = [Any if annotation is None else annotation for annotation in annotations]
    if len(types) == 1:
        return types[0]
    return tuple[tuple(types)]  # type: ignore
