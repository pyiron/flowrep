from __future__ import annotations

import ast
import dataclasses
from collections.abc import Callable

from flowrep.models import edge_models, subgraph_validation
from flowrep.models.nodes import helper_models, try_model
from flowrep.models.parsers import (
    object_scope,
    parser_protocol,
    symbol_scope,
)

TRY_BODY_LABEL: str = "try_body"
EXCEPT_BODY_LABEL_PREFIX: str = "except_body"


@dataclasses.dataclass
class _ExceptComponents:
    """Intermediate data collected while processing a single except handler."""

    exceptions: list[str]
    body_walker: parser_protocol.BodyWalker
    body_label: str
    assigned_symbols: list[str]


def parse_try_node(
    tree: ast.Try,
    scope: object_scope.ScopeProxy,
    symbol_map: symbol_scope.SymbolScope,
    walker_factory: Callable[[symbol_scope.SymbolScope], parser_protocol.BodyWalker],
) -> try_model.TryNode:
    """
    Walk a try/except block.

    Args:
        tree: The ``ast.Try`` node.
        scope: Object-level scope for resolving callable references.
        symbol_map: The enclosing :class:`SymbolScope` (used for forking).
        walker_factory: Callable that creates a :class:`BodyWalker` from a
            :class:`SymbolScope`.  Avoids a circular import with
            ``workflow_parser.WorkflowParser``.
    """
    # 0. Fail early for unsupported syntax
    if tree.orelse:
        raise NotImplementedError(
            "Try blocks with else clauses are not supported in our parsing syntax."
        )
    if tree.finalbody:
        raise NotImplementedError(
            "Try blocks with finally clauses are not supported in our parsing "
            "syntax."
        )
    if not tree.handlers:
        raise ValueError("Try node must have at least one except handler.")

    # 1. Parse the try body
    try_scope = symbol_map.fork_scope()
    try_walker = walker_factory(try_scope)
    try_walker.walk(tree.body, scope)
    try_assigned = try_scope.assigned_symbols
    for sym in try_assigned:
        try_scope.produce(sym, sym)

    # 2. Parse each except handler
    except_components: list[_ExceptComponents] = []
    for idx, handler in enumerate(tree.handlers):
        body_label = f"{EXCEPT_BODY_LABEL_PREFIX}_{idx}"

        exceptions = _parse_exception_types(handler, scope)

        except_scope = symbol_map.fork_scope()
        except_walker = walker_factory(except_scope)
        except_walker.walk(handler.body, scope)
        except_assigned = except_scope.assigned_symbols
        for sym in except_assigned:
            except_scope.produce(sym, sym)

        except_components.append(
            _ExceptComponents(
                exceptions=exceptions,
                body_walker=except_walker,
                body_label=body_label,
                assigned_symbols=except_assigned,
            )
        )

    # 3. Wire edges
    inputs, input_edges = _wire_inputs(try_walker, except_components)
    outputs, prospective_output_edges = _wire_outputs(
        try_walker, try_assigned, except_components
    )

    # 4. Build the model
    try_labeled = helper_models.LabeledNode(
        label=TRY_BODY_LABEL,
        node=try_walker.build_model(),
    )

    exception_cases = [
        helper_models.ExceptionCase(
            exceptions=ec.exceptions,
            body=helper_models.LabeledNode(
                label=ec.body_label,
                node=ec.body_walker.build_model(),
            ),
        )
        for ec in except_components
    ]

    return try_model.TryNode(
        inputs=inputs,
        outputs=outputs,
        try_node=try_labeled,
        exception_cases=exception_cases,
        input_edges=input_edges,
        prospective_output_edges=prospective_output_edges,
    )


def _wire_inputs(
    try_walker: parser_protocol.BodyWalker,
    except_components: list[_ExceptComponents],
) -> tuple[list[str], edge_models.InputEdges]:
    """Collect input edges from the try body and all except bodies."""
    inputs: list[str] = []
    input_edges: edge_models.InputEdges = {}

    def _add_input(port: str) -> None:
        if port not in inputs:
            inputs.append(port)

    # Try body inputs
    for port in try_walker.inputs:
        input_edges[edge_models.TargetHandle(node=TRY_BODY_LABEL, port=port)] = (
            edge_models.InputSource(port=port)
        )
        _add_input(port)

    # Except body inputs
    for ec in except_components:
        for port in ec.body_walker.inputs:
            input_edges[edge_models.TargetHandle(node=ec.body_label, port=port)] = (
                edge_models.InputSource(port=port)
            )
            _add_input(port)

    return inputs, input_edges


def _wire_outputs(
    try_walker: parser_protocol.BodyWalker,
    try_assigned: list[str],
    except_components: list[_ExceptComponents],
) -> tuple[list[str], subgraph_validation.ProspectiveOutputEdges]:
    """Collect outputs and prospective output edges from try and except bodies."""
    # Union of assigned symbols across all branches, preserving first-seen order
    outputs: list[str] = []
    seen: set[str] = set()
    for sym in try_assigned:
        if sym not in seen:
            seen.add(sym)
            outputs.append(sym)
    for ec in except_components:
        for sym in ec.assigned_symbols:
            if sym not in seen:
                seen.add(sym)
                outputs.append(sym)

    # Build prospective output edges: each output maps to the list of branch
    # body nodes that can source it.
    prospective_output_edges: subgraph_validation.ProspectiveOutputEdges = {}
    for output_name in outputs:
        target = edge_models.OutputTarget(port=output_name)
        sources: list[edge_models.SourceHandle] = []
        if output_name in try_assigned:
            sources.append(
                edge_models.SourceHandle(node=TRY_BODY_LABEL, port=output_name)
            )
        for ec in except_components:
            if output_name in ec.assigned_symbols:
                sources.append(
                    edge_models.SourceHandle(node=ec.body_label, port=output_name)
                )
        prospective_output_edges[target] = sources

    return outputs, prospective_output_edges


# ======================================================================
# Pure-AST helpers
# ======================================================================


def _parse_exception_types(
    handler: ast.ExceptHandler,
    scope: object_scope.ScopeProxy,
) -> list[str]:
    """
    Resolve the exception type(s) from an except handler to fully qualified names.

    Supports single types (``except ValueError:``) and tuples
    (``except (ValueError, TypeError):``).  Bare ``except:`` and named
    handlers (``except ... as e:``) are not supported.
    """
    if handler.type is None:
        raise ValueError(
            "Bare except clauses are not supported; specify exception type(s) "
            "explicitly."
        )
    if handler.name is not None:
        raise NotImplementedError(
            "Named exception handlers ('except ... as e:') are not yet supported."
        )

    type_nodes = (
        handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    )

    fqns: list[str] = []
    for node in type_nodes:
        exc_class = object_scope.resolve_symbol_to_object(node, scope)
        if not isinstance(exc_class, type) or not issubclass(exc_class, BaseException):
            raise ValueError(
                f"Except handler must catch exception types, but resolved "
                f"{ast.dump(node)} to {exc_class!r}"
            )
        fqns.append(f"{exc_class.__module__}.{exc_class.__qualname__}")
    return fqns
