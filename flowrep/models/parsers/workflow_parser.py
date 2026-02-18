import ast
from collections.abc import Callable, Collection
from types import FunctionType
from typing import Any, cast

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
    while_parser,
)

SpecialHandlers = dict[type[ast.stmt], Callable[[Any, object_scope.ScopeProxy], None]]


def workflow(
    func: FunctionType | str | None = None,
    /,
    *output_labels: str,
) -> FunctionType | Callable[[FunctionType], FunctionType]:
    """
    Decorator that attaches a flowrep.model.WorkflowNode to the `flowrep_recipe`
    attribute of a function, under constraints that the function is parseable as a
    workflow recipe.

    Can be used as with or without args (to specify output labels) -- @workflow or
    @workflow(...)
    """
    return parser_helpers.parser2decorator(
        func,
        output_labels,
        parser=parse_workflow,
        decorator_name="@workflow",
    )


def parse_workflow(
    func: FunctionType,
    *output_labels: str,
):
    inputs = label_helpers.get_input_labels(func)
    state = WorkflowParser(
        symbol_scope.SymbolScope({p: edge_models.InputSource(port=p) for p in inputs})
    )
    tree = parser_helpers.get_ast_function_node(func)

    found_return = False

    def handle_return(stmt: ast.Return, scope: object_scope.ScopeProxy):
        nonlocal found_return
        if found_return:
            raise ValueError(
                "Workflow python definitions must have exactly one return."
            )
        found_return = True
        state.handle_return(stmt, func, output_labels)

    state.walk(
        skip_docstring(tree.body),
        object_scope.get_scope(func),
        special_handlers={ast.Return: handle_return},
    )

    if not found_return:
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


class WorkflowParser(parser_protocol.BodyWalker):
    """
    Aggregates state until there is enough data to successfully build the pydantic
    data model.

    Treatment for different ast nodes is under `handle_*` methods, and aim to keep all
    state mutation of _this object_ directly in those methods.

    Other callers reference the handle methods as they walk through some ast tree,
    e.g. to build a top-level workflow from a function definition (`ast.FunctionDef`),
    or to dynamically build a workflow from the body of some control flow.
    """

    def __init__(self, symbol_map: symbol_scope.SymbolScope):
        self.symbol_map = symbol_map
        self.nodes: union.Nodes = {}

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
        self, inputs_override: list[str] | None = None
    ) -> workflow_model.WorkflowNode:
        return workflow_model.WorkflowNode(
            inputs=self.inputs if inputs_override is None else inputs_override,
            outputs=self.outputs,
            nodes=self.nodes,
            input_edges=self.input_edges,
            edges=self.edges,
            output_edges=self.output_edges,
        )

    def visit(self, stmt: ast.stmt, scope: object_scope.ScopeProxy) -> None:
        if isinstance(stmt, ast.Assign | ast.AnnAssign):
            self.handle_assign(stmt, scope)
        elif isinstance(stmt, ast.For):
            self.handle_for(stmt, scope)
        elif isinstance(stmt, ast.While):
            self.handle_while(stmt, scope)
        elif isinstance(stmt, ast.If):
            self.handle_if(stmt, scope)
        elif isinstance(stmt, ast.Try):
            raise NotImplementedError(
                f"Support for control flow statement {type(stmt)} is forthcoming."
            )
        elif isinstance(stmt, ast.Expr) and is_append_call(stmt.value):
            self.handle_appending_to_accumulator(cast(ast.Call, stmt.value))
        else:
            raise TypeError(
                f"Workflow python definitions can only interpret assignments, a subset "
                f"of flow control (for/while/if/try) and a return, but ast found "
                f"{type(stmt)}"
            )

    def walk(
        self,
        statements: list[ast.stmt],
        scope: object_scope.ScopeProxy,
        *,
        special_handlers: SpecialHandlers | None = None,
    ) -> None:
        for stmt in statements:
            if special_handlers:
                for ast_type, handler in special_handlers.items():
                    if isinstance(stmt, ast_type):
                        handler(stmt, scope)
                        break
                else:
                    self.visit(stmt, scope)
            else:
                self.visit(stmt, scope)

    def handle_assign(
        self, body: ast.Assign | ast.AnnAssign, scope: object_scope.ScopeProxy
    ):
        # Get returned symbols from the left-hand side
        lhs = body.targets[0] if isinstance(body, ast.Assign) else body.target
        new_symbols = parser_helpers.resolve_symbols_to_strings(lhs)

        rhs = body.value
        if isinstance(rhs, ast.Call):
            child = atomic_parser.get_labeled_recipe(rhs, self.nodes.keys(), scope)
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

    def handle_for(
        self,
        tree: ast.For,
        scope: object_scope.ScopeProxy,
    ) -> None:
        fp = for_parser.ForParser()
        fp.build_body(tree, scope, self.symbol_map, WorkflowParser)
        for_node = fp.build_model()

        # Accumulators consumed by the for body are no longer available here
        self.symbol_map.declared_accumulators -= set(for_node.outputs)

        for_label = label_helpers.unique_suffix("for", self.nodes)
        self.nodes[for_label] = for_node

        for port in for_node.inputs:
            self.symbol_map.consume(port, for_label, port)

        labeled_for = helper_models.LabeledNode(label=for_label, node=for_node)
        self.symbol_map.register(new_symbols=for_node.outputs, child=labeled_for)

    def handle_while(
        self,
        tree: ast.While,
        scope: object_scope.ScopeProxy,
    ) -> None:
        wp = while_parser.WhileParser()
        wp.build_body(tree, scope, self.symbol_map, WorkflowParser)
        while_node = wp.build_model()

        while_label = label_helpers.unique_suffix("while", self.nodes)
        self.nodes[while_label] = while_node

        for port in while_node.inputs:
            self.symbol_map.consume(port, while_label, port)

        labeled_while = helper_models.LabeledNode(label=while_label, node=while_node)
        self.symbol_map.register(new_symbols=while_node.outputs, child=labeled_while)

    def handle_if(self, tree: ast.If, scope: object_scope.ScopeProxy) -> None:
        ip = if_parser.IfParser()
        ip.build_body(tree, scope, self.symbol_map, WorkflowParser)
        if_node = ip.build_model()

        if_label = label_helpers.unique_suffix("if", self.nodes)
        self.nodes[if_label] = if_node

        for port in if_node.inputs:
            self.symbol_map.consume(port, if_label, port)

        labeled_if = helper_models.LabeledNode(label=if_label, node=if_node)
        self.symbol_map.register(new_symbols=if_node.outputs, child=labeled_if)

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
            try:
                source = self.symbol_map[symbol]
            except KeyError as e:
                raise ValueError(
                    f"Return symbol '{symbol}' is not defined. "
                    f"Available: {list(self.symbol_map)}"
                ) from e

            if isinstance(source, edge_models.InputSource):
                raise ValueError(
                    f"Workflows expect outputs to be sourced from the subgraph children, "
                    f"but the symbol '{symbol}' appears to be resolved directly from the "
                    f"workflow inputs."
                )
            self.symbol_map.produce(port, symbol)

    def handle_appending_to_accumulator(self, append_call: ast.Call) -> None:
        used_accumulator = cast(
            ast.Name, cast(ast.Attribute, append_call.func).value
        ).id
        appended_symbol = cast(ast.Name, append_call.args[0]).id
        self.symbol_map.use_accumulator(used_accumulator, appended_symbol)
        appended_source = self.symbol_map[appended_symbol]
        if isinstance(appended_source, edge_models.SourceHandle):
            self.symbol_map.produce(appended_symbol, appended_symbol)


def is_append_call(node: ast.expr | ast.Expr) -> bool:
    """Check if node is an append call to a known accumulator."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "append"
        and isinstance(node.func.value, ast.Name)
    )
