import ast
from collections.abc import Callable, Collection, Iterable, Iterator
from types import FunctionType
from typing import cast

from flowrep.models import base_models, edge_models
from flowrep.models.nodes import helper_models, union, workflow_model
from flowrep.models.parsers import (
    atomic_parser,
    for_parser,
    func_def_parser,
    label_helpers,
    parser_helpers,
    parser_protocol,
    scope_helpers,
    symbol_scope,
)


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
    func_def_parser.walk_func_def(state, tree, func, inputs, output_labels)
    return state.build_model()


class WorkflowParser(parser_protocol.BodyWalker):

    def __init__(self, symbol_to_source_map: symbol_scope.SymbolScope):
        # When these are all filled, we are ready to `build_model`
        self.inputs: list[str] = []
        self.nodes: union.Nodes = {}
        self.input_edges: edge_models.InputEdges = {}
        self.edges: edge_models.Edges = {}
        self.output_edges: edge_models.OutputEdges = {}
        self.outputs: list[str] = []

        # These are internal state that doesn't translate to the final model
        self.symbol_to_source_map = symbol_to_source_map
        self._for_loop_accumulators: set[str] = set()

    def build_model(self) -> workflow_model.WorkflowNode:
        return workflow_model.WorkflowNode(
            inputs=self.inputs,
            outputs=self.outputs,
            nodes=self.nodes,
            input_edges=self.input_edges,
            edges=self.edges,
            output_edges=self.output_edges,
        )

    def handle_assign(
        self, body: ast.Assign | ast.AnnAssign, scope: scope_helpers.ScopeProxy
    ):
        # Get returned symbols from the left-hand side
        lhs = body.targets[0] if isinstance(body, ast.Assign) else body.target
        new_symbols = parser_helpers.resolve_symbols_to_strings(lhs)

        rhs = body.value
        if isinstance(rhs, ast.Call):
            # Make a new node from the rhs
            # Modifies state: nodes, input_edges, edges, symbol_to_source_map
            self.handle_assign_call(rhs, new_symbols, scope)
        elif isinstance(rhs, ast.List) and len(rhs.elts) == 0:
            self.handle_assign_empty_list(new_symbols)
        else:
            raise ValueError(
                f"Workflow python definitions can only interpret assignments with "
                f"a call on the right-hand-side, but ast found {type(rhs)}"
            )

    def handle_assign_call(
        self,
        rhs: ast.Call,
        new_symbols: list[str],
        scope: scope_helpers.ScopeProxy,
    ) -> None:
        child = self._get_labeled_recipe(
            ast_call=rhs, existing_names=self.nodes.keys(), scope=scope
        )
        self.nodes[child.label] = child.node  # In-place mutation

        self.symbol_to_source_map.register(new_symbols, child)

        self._add_edges_for_child_inputs(ast_call=rhs, child=child)

    @staticmethod
    def _get_labeled_recipe(
        ast_call: ast.Call,
        existing_names: Iterable[str],
        scope: scope_helpers.ScopeProxy,
    ) -> helper_models.LabeledNode:
        child_call = cast(
            FunctionType, scope_helpers.resolve_symbol_to_object(ast_call.func, scope)
        )
        # Since it is the .func attribute of an ast.Call,
        # the retrieved object had better be a function
        child_recipe = (
            child_call.flowrep_recipe
            if hasattr(child_call, "flowrep_recipe")
            else atomic_parser.parse_atomic(child_call)
        )
        child_name = label_helpers.unique_suffix(child_call.__name__, existing_names)
        return helper_models.LabeledNode(label=child_name, node=child_recipe)

    @staticmethod
    def _get_symbol_sources_from_child_output(
        new_symbols: list[str], child: helper_models.LabeledNode
    ) -> dict[str, edge_models.SourceHandle]:
        """Map new symbols 1:1 to the outputs of a recipe"""
        if len(new_symbols) != len(child.node.outputs):
            raise ValueError(
                f"Cannot map node outputs for '{child.label}', "
                f"{child.node.outputs}, to available symbols: {new_symbols}"
            )

        return {
            symbol: edge_models.SourceHandle(node=child.label, port=port)
            for symbol, port in zip(new_symbols, child.node.outputs, strict=True)
        }

    def _add_edges_for_child_inputs(
        self,
        ast_call: ast.Call,
        child: helper_models.LabeledNode,
    ) -> None:
        for input_symbol, input_port in yield_symbols_passed_to_input_ports(
            ast_call=ast_call, call_node=child
        ):
            try:
                source = self.symbol_to_source_map[input_symbol]
            except KeyError as e:
                raise KeyError(
                    f"Workflow python definitions require node input to be "
                    f"known symbols, but got '{input_symbol}'. available symbols: "
                    f"{self.symbol_to_source_map}"
                ) from e

            target = edge_models.TargetHandle(node=child.label, port=input_port)
            if input_symbol not in self.inputs and isinstance(
                self.symbol_to_source_map[input_symbol], edge_models.InputSource
            ):
                self.inputs.append(input_symbol)
            if isinstance(source, edge_models.SourceHandle):
                self.edges[target] = source  # In-place mutation
            elif isinstance(source, edge_models.InputSource):
                self.input_edges[target] = source  # In-place mutation
            else:  # pragma: no cover
                raise TypeError(
                    f"Unexpected edge source: {type(source)}. This state should be "
                    f"unreachable; please raise a GitHub issue."
                )

    def handle_assign_empty_list(self, new_symbols: list[str]) -> None:
        if len(new_symbols) != 1:
            raise ValueError(
                f"Empty list assignment must target exactly one symbol, "
                f"got {new_symbols}"
            )
        self._for_loop_accumulators.add(new_symbols[0])

    def handle_for(
        self,
        tree: ast.For,
        scope: scope_helpers.ScopeProxy,
        parsing_function_def: bool = False,
    ) -> None:
        # 1. Parse the iteration header — pure AST, no parser state needed
        nested_iters, zipped_iters, body_tree = for_parser.parse_for_iterations(tree)
        all_iters = nested_iters + zipped_iters

        # 2. Fork the scope: replaces iterated-over symbols with loop variables,
        #    all as InputSources from the body's perspective
        child_scope = self.symbol_to_source_map.fork_scope(
            {src: var for var, src in all_iters}
        )

        # 3. Fresh body walker with the forked scope
        body_walker = WorkflowParser(symbol_to_source_map=child_scope)

        # 4. ForParser owns the for-specific wiring; body_walker owns the
        #    general statement dispatch inside the loop body
        fp = for_parser.ForParser(
            body_walker=body_walker,
            accumulators=self._for_loop_accumulators,
        )
        result = fp.build_body(
            body_tree,
            scope=scope,
            nested_iters=nested_iters,
            zipped_iters=zipped_iters,
        )
        # 5. Build the ForNode and integrate it into *this* parser's state
        for_node = fp.build_model()
        for_label = label_helpers.unique_suffix("for", self.nodes)
        self.nodes[for_label] = for_node

        # 6. Consume accumulators that the for-loop fulfilled
        self._for_loop_accumulators -= set(result.used_accumulators)

        # 7. If input is not specified by a function definition, extend it based on
        # symbols consumed in the loop
        for broadcast_symbol in result.broadcast_symbols:
            if broadcast_symbol not in self.inputs:
                if parsing_function_def:
                    raise ValueError(
                        f"A for-loop broadcast the symbol {broadcast_symbol}, but it "
                        f"is not available in the inputs: {self.inputs}"
                    )
                self.inputs.append(broadcast_symbol)

        # 8. Register the for-node's outputs as symbols in *this* scope
        labeled_for = helper_models.LabeledNode(label=for_label, node=for_node)
        self.symbol_to_source_map.register(
            new_symbols=list(result.used_accumulators),
            child=labeled_for,
        )

        # 9. Wire the for-node's inputs back to this parser's edges
        self._connect_child_inputs(
            child_label=for_label,
            broadcast_symbols=result.broadcast_symbols,
            scattered_symbols=result.scattered_symbols,
        )

    def _connect_child_inputs(
        self,
        child_label: str,
        broadcast_symbols: list[str],
        scattered_symbols: list[str],
    ) -> None:
        for broadcast_symbol in broadcast_symbols:
            self.input_edges[
                edge_models.TargetHandle(node=child_label, port=broadcast_symbol)
            ] = edge_models.InputSource(port=broadcast_symbol)

        for source_symbol in scattered_symbols:
            target = edge_models.TargetHandle(node=child_label, port=source_symbol)
            source = self.symbol_to_source_map[source_symbol]
            if isinstance(source, edge_models.InputSource):
                self.input_edges[target] = source
            else:
                self.edges[target] = source

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

        self.outputs = list(output_labels) or scraped_labels

        self.output_edges = {}
        for symbol, port in zip(returned_symbols, self.outputs, strict=True):
            try:
                source = self.symbol_to_source_map[symbol]
            except KeyError as e:
                raise ValueError(
                    f"Return symbol '{symbol}' is not defined. "
                    f"Available: {list(self.symbol_to_source_map)}"
                ) from e

            if isinstance(source, edge_models.InputSource):
                raise ValueError(
                    f"Workflows expect outputs to be sourced from the subgraph children, "
                    f"but the symbol '{symbol}' appears to be resolved directly from the "
                    f"workflow inputs."
                )
            self.output_edges[edge_models.OutputTarget(port=port)] = source

    def handle_appending_to_accumulator(
        self, append_stmt: ast.Expr, accumulators: set[str]
    ) -> tuple[str, str]:
        if is_append_call(append_stmt.value, accumulators):
            append_call = cast(ast.Call, append_stmt.value)
            used_accumulator = cast(
                ast.Name, cast(ast.Attribute, append_call.func).value
            ).id
            appended_symbol = cast(ast.Name, append_call.args[0]).id
            appended_source = self.symbol_to_source_map[appended_symbol]
            if isinstance(appended_source, edge_models.SourceHandle):
                self.outputs.append(appended_symbol)
                self.output_edges[edge_models.OutputTarget(port=appended_symbol)] = (
                    appended_source
                )
            return used_accumulator, appended_symbol
        else:
            raise TypeError(
                f"Inside the context of an ast.For node, the only expression "
                f"currently parseable is an `.append` call to a known "
                f"accumulator symbol. Instead, got a body value {append_stmt.value}."
                f"Currently known accumulators: {accumulators}."
            )


def yield_symbols_passed_to_input_ports(
    ast_call: ast.Call,
    call_node: helper_models.LabeledNode,
) -> Iterator[tuple[str, str]]:
    """Get (argument name, symbol) pairs passed to a node-creating call"""

    def _validate_is_ast_name(node: ast.expr) -> ast.Name:
        if not isinstance(node, ast.Name):
            raise TypeError(
                f"Workflow python definitions can only interpret function "
                f"calls with symbolic input, and thus expected to find an "
                f"ast.Name, but when parsing input for {call_node.label}, found a "
                f"type {type(node)}"
            )
        return node

    for i, arg in enumerate(ast_call.args):
        name_arg = _validate_is_ast_name(arg)
        yield name_arg.id, call_node.node.inputs[i]
    for kw in ast_call.keywords:
        name_value = _validate_is_ast_name(kw.value)
        if not isinstance(kw.arg, str):  # pragma: no cover
            raise TypeError(
                "How did you get here? A `None` value should be possible for "
                "**kwargs, but variadics should have been excluded before "
                "this. Please raise a GitHub issue."
            )
        yield name_value.id, kw.arg


def is_append_call(node: ast.expr | ast.Expr, accumulators: set[str]) -> bool:
    """Check if node is an append call to a known accumulator."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "append"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in accumulators
    )
