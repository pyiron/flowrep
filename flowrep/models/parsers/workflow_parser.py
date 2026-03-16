from __future__ import annotations

import ast
import importlib
from collections.abc import Callable, Collection
from types import FunctionType
from typing import cast

from pyiron_snippets import versions

from flowrep.models import base_models, edge_models
from flowrep.models.nodes import helper_models, union, workflow_model
from flowrep.models.parsers import (
    atomic_parser,
    for_parser,
    if_parser,
    label_helpers,
    object_scope,
    parser_helpers,
    parser_protocol,
    symbol_scope,
    try_parser,
    while_parser,
)


def workflow(
    func: FunctionType | str | None = None,
    /,
    *output_labels: str,
    version_scraping: versions.VersionScrapingMap | None = None,
    forbid_main: bool = False,
    forbid_locals: bool = False,
    require_version: bool = False,
) -> FunctionType | Callable[[FunctionType], FunctionType]:
    """
    Decorator that attaches a :class:`~flowrep.models.nodes.workflow_model.WorkflowNode`
    to the ``flowrep_recipe`` attribute of a function, under constraints that the
    function body is parseable as a workflow recipe.

    The decorated function's module, qualname, and (optionally) package version are
    captured as provenance metadata via
    :meth:`~pyiron_snippets.versions.VersionInfo.of`.

    Can be used with or without arguments.

    Args:
        func: The function to decorate. Passed positionally by Python when the
            decorator is used without parentheses.
        *output_labels: Explicit names for the workflow's output ports. When
            provided, their count must match the number of returned symbols.
        version_scraping: Optional mapping from top-level package names to callables
            that return a version string. Forwarded to
            :meth:`~pyiron_snippets.versions.VersionInfo.of`.
        forbid_main: If ``True``, raise if the function's module is ``__main__``.
        forbid_locals: If ``True``, raise if the function's qualname contains
            ``<locals>``.
        require_version: If ``True``, raise if no version can be determined for
            the function's package.

    Returns:
        The original function with a ``flowrep_recipe`` attribute holding a
        :class:`~flowrep.models.nodes.workflow_model.WorkflowNode`.
    """
    return parser_helpers.parser2decorator(
        func,
        output_labels,
        parser=parse_workflow,
        decorator_name="@workflow",
        parser_kwargs={
            "version_scraping": version_scraping,
            "forbid_main": forbid_main,
            "forbid_locals": forbid_locals,
            "require_version": require_version,
        },
    )


def parse_workflow(
    func: FunctionType,
    *output_labels: str,
    version_scraping: versions.VersionScrapingMap | None = None,
    forbid_main: bool = False,
    forbid_locals: bool = False,
    require_version: bool = False,
):
    """
    Build a :class:`~flowrep.models.nodes.workflow_model.WorkflowNode` by
    statically analysing a Python function's AST.

    The function body is walked statement-by-statement; assignments with calls on
    the right-hand side become atomic (or recursively parsed) child nodes, and
    supported control-flow structures (``for``, ``while``, ``if``, ``try``) are
    converted into the corresponding composite node types. A single ``return``
    statement defines the workflow's output ports.

    Args:
        func: The function to parse into a workflow graph.
        *output_labels: Explicit output port names. When provided, their count must
            match the number of returned symbols.
        version_scraping: Optional version-scraping overrides, forwarded to
            :meth:`~pyiron_snippets.versions.VersionInfo.of`.
        forbid_main: If ``True``, raise if the function's module is ``__main__``.
        forbid_locals: If ``True``, raise if the function's qualname contains
            ``<locals>``.
        require_version: If ``True``, raise if no version can be determined.

    Returns:
        A fully constructed :class:`WorkflowNode`.

    Raises:
        ValueError: If the function has no return, multiple returns, returns
            duplicate symbols, returns workflow inputs directly, or if any
            ``forbid_*`` / ``require_*`` constraint is violated.
        TypeError: If the function body contains unsupported AST statement types.
    """
    info_factory = versions.VersionInfoFactory(
        version_scraping=version_scraping,
        forbid_main=forbid_main,
        forbid_locals=forbid_locals,
        require_version=require_version,
    )
    function_info = info_factory.of(func)
    signature_info = parser_helpers.SignatureInfo.of(func)

    inputs = signature_info.names
    reference = base_models.PythonReference(
        info=function_info,
        inputs_with_defaults=signature_info.have_defaults,
        restricted_input_kinds=signature_info.have_restricted_kinds,
    )
    state = _WorkflowFunctionParser(
        object_scope.get_scope(func),
        symbol_scope.SymbolScope({p: edge_models.InputSource(port=p) for p in inputs}),
        source=reference,
        info_factory=info_factory,
        func=func,
        output_labels=output_labels,
    )
    tree = parser_helpers.get_ast_function_node(func)

    state.walk(skip_docstring(tree.body))

    if not state.found_return:
        raise ValueError("Workflow python definitions must have a return statement.")

    return state.build_model(inputs_override=inputs)


def skip_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    return (
        body[1:]
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        )
        else body
    )


class WorkflowParser(ast.NodeVisitor, parser_protocol.BodyWalker):
    """
    Aggregates state until there is enough data to successfully build the pydantic
    data model.

    Treatment for different ast nodes is under `handle_*` methods, and aim to keep all
    state mutation of _this object_ directly in those methods.

    Other callers reference the handle methods as they walk through some ast tree,
    e.g. to build a top-level workflow from a function definition (`ast.FunctionDef`),
    or to dynamically build a workflow from the body of some control flow.
    """

    def __init__(
        self,
        scope: object_scope.ScopeProxy,
        symbol_map: symbol_scope.SymbolScope,
        info_factory: versions.VersionInfoFactory,
        source: base_models.PythonReference | None = None,
    ):
        self.scope = scope
        self.symbol_map = symbol_map
        self.info_factory = info_factory
        self.nodes: union.Nodes = {}
        self.source = source

    @property
    def inputs(self) -> list[str]:
        return self.symbol_map.inputs

    @property
    def input_edges(self) -> edge_models.InputEdges:
        return self.symbol_map.input_edges

    @property
    def edges(self) -> edge_models.Edges:
        return self.symbol_map.edges

    @property
    def output_edges(self) -> edge_models.OutputEdges:
        return self.symbol_map.output_edges

    @property
    def outputs(self) -> list[str]:
        return self.symbol_map.outputs

    def build_model(
        self,
        inputs_override: list[str] | None = None,
    ) -> workflow_model.WorkflowNode:
        return workflow_model.WorkflowNode(
            inputs=self.inputs if inputs_override is None else inputs_override,
            outputs=self.outputs,
            nodes=self.nodes,
            input_edges=self.input_edges,
            edges=self.edges,
            output_edges=self.output_edges,
            reference=self.source,
        )

    def fork(
        self,
        *,
        new_symbol_map: symbol_scope.SymbolScope,
        new_scope: object_scope.ScopeProxy,
    ) -> WorkflowParser:
        """Create a child walker with optionally replaced symbol map and scope.

        Configuration (version scraping, constraints, etc.) is propagated
        from this walker.  If *new_scope* is ``None``, ``self.scope`` is
        reused (shared, not copied).
        """
        return WorkflowParser(
            scope=new_scope,
            symbol_map=new_symbol_map,
            info_factory=self.info_factory,
        )

    def walk(self, statements: list[ast.stmt]) -> None:
        for statement in statements:
            self.visit(statement)

    def visit_Assign(self, stmt: ast.Assign) -> None:
        self._handle_assign(stmt)

    def visit_AnnAssign(self, stmt: ast.AnnAssign) -> None:
        self._handle_assign(stmt)

    def _handle_assign(self, body: ast.Assign | ast.AnnAssign):
        # Get returned symbols from the left-hand side
        lhs = body.targets[0] if isinstance(body, ast.Assign) else body.target
        new_symbols = parser_helpers.resolve_symbols_to_strings(lhs)

        rhs = body.value
        if isinstance(rhs, ast.Call):
            child = atomic_parser.get_labeled_recipe(
                rhs,
                self.nodes.keys(),
                self.scope,
                self.info_factory,
            )
            self.nodes[child.label] = child.node
            parser_helpers.consume_call_arguments(self.symbol_map, rhs, child)
            self.symbol_map.register(new_symbols, child)
        elif isinstance(rhs, ast.List) and len(rhs.elts) == 0:
            if len(new_symbols) != 1:
                raise ValueError(
                    f"Empty list assignment must target exactly one symbol, "
                    f"got {new_symbols}"
                )
            self.symbol_map.register_accumulator(new_symbols[0])
        else:
            raise ValueError(
                f"Workflow python definitions can only interpret assignments with "
                f"a call or empty list on the right-hand-side, but ast found "
                f"{type(rhs)}"
            )

    def _digest_flow_control(self, label_prefix: str, node: union.NodeType) -> None:
        label = label_helpers.unique_suffix(label_prefix, self.nodes)
        self.nodes[label] = node
        self._connect_node_to_enclosing_scope(label, node)

    def _connect_node_to_enclosing_scope(self, label: str, node: union.NodeType):
        for port in node.inputs:
            self.symbol_map.consume(port, label, port)

        labeled_node = helper_models.LabeledNode(label=label, node=node)
        self.symbol_map.register(new_symbols=node.outputs, child=labeled_node)

    def visit_For(self, tree: ast.For) -> None:
        for_node = for_parser.parse_for_node(self, tree)
        # Accumulators consumed by the for body are no longer available here
        self.symbol_map.declared_accumulators -= set(for_node.outputs)
        self._digest_flow_control("for", for_node)

    def visit_While(self, tree: ast.While) -> None:
        while_node = while_parser.parse_while_node(self, tree)
        self._digest_flow_control("while", while_node)

    def visit_If(self, tree: ast.If) -> None:
        if_node = if_parser.parse_if_node(self, tree)
        self._digest_flow_control("if", if_node)

    def visit_Try(self, tree: ast.Try) -> None:
        try_node = try_parser.parse_try_node(self, tree)
        self._digest_flow_control("try", try_node)

    def visit_Expr(self, stmt: ast.Expr) -> None:
        if is_append_call(stmt.value):
            self._handle_appending_to_accumulator(cast(ast.Call, stmt.value))
        else:
            self.generic_visit(stmt)

    def visit_Import(self, node: ast.Import) -> None:
        """
        Handle ``import foo`` and ``import foo as bar`` statements.

        Resolves the imported module and registers it in the current
        :class:`ScopeProxy` so that subsequent attribute-based calls
        (e.g. ``foo.func(x)``) can be resolved.
        """
        for alias in node.names:
            module = importlib.import_module(alias.name)
            if alias.asname is not None:
                # import numpy as np  →  register "np" → numpy module
                self.scope.register(alias.asname, module)
            else:
                # import os.path  →  register "os" → os module (top-level only)
                top_level_name = alias.name.split(".")[0]
                top_level_module = importlib.import_module(top_level_name)
                self.scope.register(top_level_name, top_level_module)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """
        Handle ``from foo import bar`` and ``from foo import bar as baz``.

        Resolves each imported name and registers it in the current scope.
        """
        if node.module is None or node.level > 0:
            raise ValueError(
                f"Relative imports are not supported in workflow definitions. "
                f"Encountered importing from {node.module}."
            )
        module = importlib.import_module(node.module)
        for alias in node.names:
            obj = getattr(module, alias.name)
            local_name = alias.asname if alias.asname is not None else alias.name
            self.scope.register(local_name, obj)

    def _handle_appending_to_accumulator(self, append_call: ast.Call) -> None:
        used_accumulator = cast(
            ast.Name, cast(ast.Attribute, append_call.func).value
        ).id
        appended_symbol = cast(ast.Name, append_call.args[0]).id
        self.symbol_map.use_accumulator(used_accumulator, appended_symbol)
        appended_source = self.symbol_map[appended_symbol]
        if isinstance(appended_source, edge_models.SourceHandle):
            self.symbol_map.produce(appended_symbol)

    def generic_visit(self, stmt: ast.AST) -> None:
        raise TypeError(
            f"Workflow python definitions can only interpret a subset of assignments, "
            f"and flow controls (for/while/if/try) and (when parsing a function "
            f"definition) a return, but ast found "
            f"{type(stmt)}"
        )


class _WorkflowFunctionParser(WorkflowParser):
    def __init__(
        self,
        scope: object_scope.ScopeProxy,
        symbol_map: symbol_scope.SymbolScope,
        info_factory: versions.VersionInfoFactory,
        *,
        source: base_models.PythonReference | None = None,
        func: FunctionType,
        output_labels: Collection[str],
    ):
        super().__init__(
            scope,
            symbol_map,
            info_factory,
            source=source,
        )
        self._func = func
        self._output_labels = output_labels
        self._found_return = False

    @property
    def found_return(self) -> bool:
        return self._found_return

    def visit_Return(self, stmt: ast.Return) -> None:
        if self._found_return:
            raise ValueError(
                "Workflow python definitions must have exactly one return."
            )
        self._found_return = True
        self.handle_return(stmt, self._func, self._output_labels)

    def handle_return(
        self,
        body: ast.Return,
        func: FunctionType,
        output_labels: Collection[str],
    ) -> None:
        returned_symbols = parser_helpers.resolve_symbols_to_strings(body.value)
        base_models.validate_unique(
            returned_symbols,
            message=f"Workflow python definitions must have unique returns, but "
            f"got duplicates in: {returned_symbols}",
        )

        annotated_returns = label_helpers.get_annotated_output_labels(func)
        scraped_labels = label_helpers.merge_labels(
            first_choice=annotated_returns,
            fallback=returned_symbols,
            message_prefix="Annotation labels and returned symbols mis-match. ",
        )

        if output_labels and len(output_labels) != len(returned_symbols):
            raise ValueError(
                f"When output_labels are specified ({output_labels}), workflow "
                f"python definitions have a matching number of returned symbols "
                f"({returned_symbols})."
            )

        final_ports = list(output_labels) if output_labels else scraped_labels

        for symbol, port in zip(returned_symbols, final_ports, strict=True):
            if symbol not in self.symbol_map:
                raise ValueError(
                    f"Return symbol '{symbol}' is not defined. "
                    f"Available: {list(self.symbol_map)}"
                )

            self.symbol_map.produce(port, symbol)


def is_append_call(node: ast.expr | ast.Expr) -> bool:
    """Check if node is an append call to a known accumulator."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "append"
        and isinstance(node.func.value, ast.Name)
    )
