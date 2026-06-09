from __future__ import annotations

import dataclasses
import inspect
import re
import typing
from typing import Any

from flowrep import base_models, retrospective
from flowrep.compiler import annotate, flow_control, statements
from flowrep.nodes import (
    for_recipe,
    if_recipe,
    try_recipe,
    union_types,
    while_recipe,
    workflow_recipe,
)
from flowrep.parsers import label_helpers


class NameAllocator:
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
class Emitter:
    namespace: dict[str, Any] = dataclasses.field(default_factory=dict)
    nested_defs: list[str] = dataclasses.field(default_factory=list)
    module_imports: set[str] = dataclasses.field(default_factory=set)
    # Module-scope names: nested def names, injected namespace symbols, and
    # reserved import bindings. Distinct from the per-function local allocator.
    module_names: NameAllocator = dataclasses.field(default_factory=NameAllocator)
    workflow_decorator: tuple[str, str] = ("not", "useful")

    @property
    def decorator_string(self):
        mod, qualname = self.workflow_decorator
        return f"@{mod}.{qualname}"


@dataclasses.dataclass
class FunctionBuilder:
    name: str
    decorator: str
    params: list[str] = dataclasses.field(default_factory=list)
    body_lines: list[str] = dataclasses.field(default_factory=list)
    return_annotation: str | None = None
    return_symbols: list[str] = dataclasses.field(default_factory=list)
    output_labels: list[str] = dataclasses.field(default_factory=list)

    def render(self) -> str:
        sig = ", ".join(self.params)
        # Every generated function is decorated so that exec'ing the source attaches
        # a `.flowrep_recipe`; this also validates the body parses as a workflow.
        # Output port names are pinned via decorator args; the parser resolves
        # `output_labels` ahead of annotation labels and returned-symbol names, so
        # this round-trips port names without polluting the return annotation.
        if self.output_labels:
            label_args = ", ".join(f'"{label}"' for label in self.output_labels)
            header = f"{self.decorator}({label_args})\n"
        else:
            header = f"{self.decorator}\n"
        header += f"def {self.name}({sig})"
        if self.return_annotation is not None:
            header += f" -> {self.return_annotation}"
        header += ":"
        ret = ""
        if self.return_symbols:
            ret = "    return " + ", ".join(self.return_symbols) + "\n"
        # Only emit `pass` when the body is entirely empty (no lines, no return);
        # a passthrough workflow has no body lines but does have a return, so
        # `pass` would be both redundant and illegal in a @flowrep.workflow function.
        body = self.body_lines or ([] if ret else ["pass"])
        indented = "".join(f"    {line}\n" for line in body)
        return header + "\n" + indented + ret


def referenced_top_level_bindings(
    node: union_types.RecipeDiscrimination,
) -> typing.Iterator[str]:
    """Yield the top-level module binding of every import the recipe will emit.

    Mirrors the structure emission walks: referenced nodes (atomic or referenced
    workflow) emit a dotted call and are not recursed; reference-free workflows
    and flow controls are recursed via ``.nodes`` / ``.prospective_nodes``.
    Builtin exception types are skipped, matching ``_exception_name``.
    """
    reference = getattr(node, "reference", None)
    if reference is not None:
        yield reference.info.module.split(".")[0]
        return
    if isinstance(node, workflow_recipe.WorkflowRecipe):
        for child in node.nodes.values():
            yield from referenced_top_level_bindings(child)
    elif isinstance(node, flow_control.FLOW_CONTROL_TYPES):
        for child in node.prospective_nodes.values():
            yield from referenced_top_level_bindings(child)
        if isinstance(node, try_recipe.TryRecipe):
            for case in node.exception_cases:
                for info in case.exceptions:
                    if info.module != "builtins":
                        yield info.module.split(".")[0]


def _inlined_loop_variables(recipe: workflow_recipe.WorkflowRecipe) -> set[str]:
    """Loop-variable names emitted into this workflow's own scope.

    For-loop variables are pinned to body-port names and bypass the allocator, so
    reserving them prevents allocator-minted symbols from shadowing them. Descends
    through inlined flow-control bodies; stops at reference-free WorkflowRecipe peer
    nodes (emitted as nested defs with their own scope) and referenced nodes.
    """
    names: set[str] = set()
    for node in recipe.nodes.values():
        _collect_loop_variables(node, names)
    return names


def _collect_loop_variables(
    node: union_types.RecipeDiscrimination, names: set[str]
) -> None:
    if isinstance(node, for_recipe.ForEachRecipe):
        names.update(node.iterated_ports)
        _collect_loop_variables_from_body(node.body_node.node, names)
    elif isinstance(node, if_recipe.IfRecipe):
        for body_case in node.cases:
            _collect_loop_variables_from_body(body_case.body.node, names)
        if node.else_case is not None:
            _collect_loop_variables_from_body(node.else_case.node, names)
    elif isinstance(node, try_recipe.TryRecipe):
        _collect_loop_variables_from_body(node.try_node.node, names)
        for exception_case in node.exception_cases:
            _collect_loop_variables_from_body(exception_case.body.node, names)
    elif isinstance(node, while_recipe.WhileRecipe):
        _collect_loop_variables_from_body(node.case.body.node, names)
    # atomic / referenced / reference-free workflow peer node: not inlined here


def _collect_loop_variables_from_body(
    body: union_types.RecipeDiscrimination, names: set[str]
) -> None:
    """Recurse into an inlined body: a WorkflowRecipe's own nodes, or a nested
    flow-control node. Both are emitted into the current scope."""
    if isinstance(body, workflow_recipe.WorkflowRecipe):
        for node in body.nodes.values():
            _collect_loop_variables(node, names)
    elif isinstance(body, flow_control.FLOW_CONTROL_TYPES):
        _collect_loop_variables(body, names)
    # single atomic body: no loop variables


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


def output_name_suggestion(label: str, port: str, n_outputs: int) -> str:
    """Symbol-name hint for a call assignment, derived from the node label.

    Single-output nodes use the label base; multi-output nodes disambiguate per
    port. Pinned (required) names still take precedence at the call site.
    """
    base = _label_base(label)
    return base if n_outputs == 1 else f"{base}_{port}"


def emit_nested_workflow_node(
    node: workflow_recipe.WorkflowRecipe,
    label: str,
    emitter: Emitter,
    alloc: NameAllocator,
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
    # Module-scope: nested defs are emitted at module level, so their names must
    # be unique across the whole module (not just the parent's locals). Reserve
    # the chosen name in the parent allocator too, so the parent can't later mint
    # a local symbol that shadows the def (it is called by bare name in the body).
    fn_name = emitter.module_names.fresh(base)
    alloc.reserve(fn_name)
    builder = emit_workflow_function(node, fn_name, emitter, signature=None)
    # The @flowrep.workflow decorator is emitted by FunctionBuilder.render itself.
    rendered = builder.render()
    if fn_name != base:
        # Node labels are re-derived from the resolved function's __name__
        # (atomic_parser / unique_suffix). When a unique module binding name must
        # differ from the base -- restore __name__ so re-parsing reconstructs the
        # original label. Appended to the nested def's text; nested defs are emitted
        # at module level before the enclosing function, so the assignment executes
        # before the enclosing @workflow decorator parses the body.
        rendered += f"{fn_name}.__name__ = {base!r}\n"
    emitter.nested_defs.append(rendered)
    return fn_name


def _render_params(
    recipe: workflow_recipe.WorkflowRecipe,
    signature: inspect.Signature | None,
    emitter: Emitter,
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
            annotation = typing.cast(inspect.Parameter, param).annotation
            inlined = annotate.render_annotation(annotation, emitter.module_imports)
            if inlined is not None:
                piece += f": {inlined}"
            else:
                annotation_name = emitter.module_names.fresh(f"_ann_{port}")
                emitter.namespace[annotation_name] = annotation
                piece += f": {annotation_name}"
        if has_default:
            default = typing.cast(inspect.Parameter, param).default
            inlined_default = annotate.render_default(default)
            if inlined_default is not None:
                rhs = inlined_default
            else:
                default_name = emitter.module_names.fresh(f"_default_{port}")
                emitter.namespace[default_name] = default
                rhs = default_name
            # PEP 8 / black: spaces around '=' only when the parameter is annotated.
            piece += f" = {rhs}" if has_annotation else f"={rhs}"
        params.append(piece)

    # Edge case: the last param was positional-only and no marker emitted yet.
    if need_pos_only_marker and seen_positional_only and "/" not in params:
        params.append("/")
    return params


def build_signature(
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
    ``tuple[...]``. Returns ``inspect.Signature.empty`` when no port carries a type.
    """
    annotations = [port.annotation for port in outputs.values()]
    if all(annotation is None for annotation in annotations):
        return inspect.Signature.empty
    types = [Any if annotation is None else annotation for annotation in annotations]
    if len(types) == 1:
        return types[0]
    return tuple[tuple(types)]  # type: ignore


def emit_workflow_function(
    recipe: workflow_recipe.WorkflowRecipe,
    name: str,
    emitter: Emitter,
    signature: inspect.Signature | None = None,
) -> FunctionBuilder:
    alloc = NameAllocator()
    in_syms = {port: alloc.reserve(port) for port in recipe.inputs}
    # Loop variables are pinned to body-port names and bypass the allocator; reserve
    # them so node-output symbols never collide with (and get shadowed by) them.
    for loop_variable in _inlined_loop_variables(recipe):
        alloc.reserve(loop_variable)
    params = _render_params(recipe, signature, emitter)
    lines, out_syms = statements.emit_workflow_body(recipe, in_syms, {}, emitter, alloc)
    # Output port names are pinned via the decorator (see FunctionBuilder.render),
    # so the return annotation is free to carry the user's verbatim type. Bind the
    # whole annotation as a live object (arbitrary types do not round-trip through
    # repr) and reference it by name; emit nothing when there is no real annotation.
    return_annotation = None
    if (
        signature is not None
        and signature.return_annotation is not inspect.Signature.empty
    ):
        inlined_return = annotate.render_annotation(
            signature.return_annotation, emitter.module_imports
        )
        if inlined_return is not None:
            return_annotation = inlined_return
        else:
            ann_name = emitter.module_names.fresh("_ann_return")
            emitter.namespace[ann_name] = signature.return_annotation
            return_annotation = ann_name
    # Imports are hoisted to the module preamble via emitter.module_imports, so
    # the body carries only the optional docstring (which must come first so the
    # parser's skip_docstring recognises it and `description` round-trips) and
    # the emitted statements.
    body_lines = lines
    if recipe.description is not None:
        body_lines = [repr(recipe.description)] + body_lines
    builder = FunctionBuilder(
        name=name,
        decorator=emitter.decorator_string,
        params=params,
        body_lines=body_lines,
        return_annotation=return_annotation,
        return_symbols=[out_syms[p] for p in recipe.outputs],
        output_labels=list(recipe.outputs),
    )
    return builder
