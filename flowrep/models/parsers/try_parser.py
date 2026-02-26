from __future__ import annotations

import ast

from pyiron_snippets import versions

from flowrep.models.nodes import helper_models, try_model
from flowrep.models.parsers import (
    case_helpers,
    object_scope,
    parser_protocol,
    symbol_scope,
)

TRY_BODY_LABEL: str = "try_body"
EXCEPT_BODY_LABEL_PREFIX: str = "except_body"


def parse_try_node(
    tree: ast.Try,
    scope: object_scope.ScopeProxy,
    symbol_map: symbol_scope.SymbolScope,
    info_factory: versions.VersionInfoFactory,
    walker_factory: parser_protocol.WalkerFactory,
) -> try_model.TryNode:
    """
    Walk a try/except block.

    Args:
        tree: The ``ast.Try`` node.
        scope: Object-level scope for resolving callable references.
        symbol_map: The enclosing :class:`SymbolScope` (used for forking).
        info_factory: Stateful object for collecting version info.
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

    # 1. Parse the try body
    try_branch = case_helpers.walk_branch(
        TRY_BODY_LABEL,
        tree.body,
        symbol_map,
        scope,
        info_factory,
        walker_factory,
    )

    # 2. Parse each except handler
    exception_groups: list[list[str]] = []
    except_branches: list[case_helpers.WalkedBranch] = []
    for idx, handler in enumerate(tree.handlers):
        body_label = f"{EXCEPT_BODY_LABEL_PREFIX}_{idx}"

        exception_groups.append(_parse_exception_types(handler, scope))

        exception_branch = case_helpers.walk_branch(
            body_label,
            handler.body,
            symbol_map,
            scope,
            info_factory,
            walker_factory,
        )
        except_branches.append(exception_branch)

    # 3. Wire edges
    branches = [try_branch] + except_branches
    inputs, input_edges = case_helpers.wire_inputs(branches)
    outputs, prospective_output_edges = case_helpers.wire_outputs(branches)

    exception_cases = [
        helper_models.ExceptionCase(
            exceptions=exceptions,
            body=branch.to_labeled_node(),
        )
        for exceptions, branch in zip(exception_groups, except_branches, strict=True)
    ]

    return try_model.TryNode(
        inputs=inputs,
        outputs=outputs,
        try_node=try_branch.to_labeled_node(),
        exception_cases=exception_cases,
        input_edges=input_edges,
        prospective_output_edges=prospective_output_edges,
    )


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
        exc_info = versions.VersionInfo.of(exc_class)
        if not isinstance(exc_class, type) or not issubclass(exc_class, BaseException):
            raise ValueError(
                f"Except handler must catch exception types, but resolved "
                f"{ast.dump(node)} to {exc_class!r}"
            )
        fqns.append(exc_info.fully_qualified_name)
    return fqns
