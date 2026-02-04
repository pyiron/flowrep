import ast
from collections.abc import Callable, Collection, Iterable, Iterator
from types import FunctionType
from typing import cast

from flowrep.models import base_models, edge_models
from flowrep.models.nodes import helper_models, union, workflow_model
from flowrep.models.parsers import (
    atomic_parser,
    label_helpers,
    parser_helpers,
    scope_helpers,
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
    state = WorkflowParser(inputs=label_helpers.get_input_labels(func))

    tree = parser_helpers.get_ast_function_node(func)
    found_return = False
    for body in tree.body:
        if isinstance(body, ast.Assign | ast.AnnAssign):
            state.handle_assign(func, body)
        elif isinstance(body, ast.Return):
            if found_return:
                raise ValueError(
                    "Workflow python definitions must have exactly one return."
                )
            found_return = True
            # Sets state: outputs, output_edges
            state.handle_return(func, body, output_labels)
        elif isinstance(body, ast.For | ast.While | ast.If | ast.Try):
            raise NotImplementedError(
                f"Support for control flow statement {type(body)} is forthcoming."
            )
        else:
            raise TypeError(
                f"Workflow python definitions can only interpret assignments, a subset "
                f"of flow control (for/while/if/try) and a return, but ast found "
                f"{type(body)}"
            )

    if not found_return:
        raise ValueError("Workflow python definitions must have a return statement.")

    return workflow_model.WorkflowNode(
        inputs=state.inputs,
        outputs=state.outputs,
        nodes=state.nodes,
        input_edges=state.input_edges,
        edges=state.edges,
        output_edges=state.output_edges,
    )


class WorkflowParser:

    def __init__(self, inputs: list[str]):
        self.inputs = inputs
        self.nodes: union.Nodes = {}
        self.input_edges: edge_models.InputEdges = {}
        self.edges: edge_models.Edges = {}
        self.output_edges: edge_models.OutputEdges = {}
        self.outputs: list[str] = []

        self._symbol_to_source_map: dict[
            str, edge_models.InputSource | edge_models.SourceHandle
        ] = {p: edge_models.InputSource(port=p) for p in inputs}

    def enforce_unique_symbols(self, new_symbols: Iterable[str]) -> None:
        if overshadow := set(self._symbol_to_source_map).intersection(new_symbols):
            raise ValueError(
                f"Workflow python definitions must not re-use symbols, but found "
                f"duplicate(s) {overshadow}"
            )

    def handle_assign(self, func: FunctionType, body: ast.Assign | ast.AnnAssign):
        # Get returned symbols from the left-hand side
        lhs = body.targets[0] if isinstance(body, ast.Assign) else body.target
        new_symbols = parser_helpers.resolve_symbols_to_strings(lhs)
        self.enforce_unique_symbols(new_symbols)

        rhs = body.value
        if isinstance(rhs, ast.Call):
            # Make a new node from the rhs
            # Modifies state: nodes, input_edges, edges, symbol_to_source_map
            self.handle_assign_call(rhs, new_symbols, scope_helpers.get_scope(func))
        elif isinstance(rhs, ast.List) and len(rhs.elts) == 0:
            self.handle_assign_empty_list()
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

        self._symbol_to_source_map.update(
            self._get_symbol_sources_from_child_output(
                new_symbols=new_symbols,
                child=child,
            )
        )

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
                source = self._symbol_to_source_map[input_symbol]
            except KeyError as e:
                raise KeyError(
                    f"Workflow python definitions require node input to be "
                    f"known symbols, but got '{input_symbol}'. available symbols: "
                    f"{self._symbol_to_source_map}"
                ) from e

            target = edge_models.TargetHandle(node=child.label, port=input_port)
            if isinstance(source, edge_models.SourceHandle):
                self.edges[target] = source  # In-place mutation
            elif isinstance(source, edge_models.InputSource):
                self.input_edges[target] = source  # In-place mutation
            else:  # pragma: no cover
                raise TypeError(
                    f"Unexpected edge source: {type(source)}. This state should be "
                    f"unreachable; please raise a GitHub issue."
                )

    def handle_assign_empty_list(self) -> None:
        raise NotImplementedError(
            "Assigning empty will probably be lists will probably be used for "
            "coordinating loop aggregators in the future, but is not yet "
            "supported."
        )

    def handle_return(
        self,
        func: FunctionType,
        body: ast.Return,
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
                source = self._symbol_to_source_map[symbol]
            except KeyError as e:
                raise ValueError(
                    f"Return symbol '{symbol}' is not defined. "
                    f"Available: {list(self._symbol_to_source_map)}"
                ) from e

            if isinstance(source, edge_models.InputSource):
                raise ValueError(
                    f"Workflows expect outputs to be sourced from the subgraph children, "
                    f"but the symbol '{symbol}' appears to be resolved directly from the "
                    f"workflow inputs."
                )
            self.output_edges[edge_models.OutputTarget(port=port)] = source


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
