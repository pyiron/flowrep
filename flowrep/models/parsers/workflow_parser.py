import ast
from collections.abc import Callable, Collection, Iterable, Iterator
from types import FunctionType
from typing import cast

from flowrep.models import base_models, edge_models
from flowrep.models.nodes import for_model, helper_models, union, workflow_model
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
    inputs = label_helpers.get_input_labels(func)
    symbol_to_source_map: SymbolSourceMapType = {
        p: edge_models.InputSource(port=p) for p in inputs
    }
    state = WorkflowParser(symbol_to_source_map)
    state.inputs = inputs
    tree = parser_helpers.get_ast_function_node(func)
    state.walk_func_def(tree, func, output_labels)
    return state.build_model()


SymbolSourceMapType = dict[str, edge_models.InputSource | edge_models.SourceHandle]


class WorkflowParser:

    def __init__(self, symbol_to_source_map: SymbolSourceMapType):
        self.inputs: list[str] = []
        self.nodes: union.Nodes = {}
        self.input_edges: edge_models.InputEdges = {}
        self.edges: edge_models.Edges = {}
        self.output_edges: edge_models.OutputEdges = {}
        self.outputs: list[str] = []

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

    def walk_func_def(
        self, tree: ast.FunctionDef, func: FunctionType, output_labels: Collection[str]
    ) -> None:
        scope = scope_helpers.get_scope(func)

        found_return = False
        for body in parser_helpers.skip_docstring(tree.body):
            if isinstance(body, ast.Assign | ast.AnnAssign):
                self.handle_assign(body, scope)
            elif isinstance(body, ast.For):
                self.handle_for(body, scope, parsing_function_def=True)
            elif isinstance(body, ast.While | ast.If | ast.Try):
                raise NotImplementedError(
                    f"Support for control flow statement {type(body)} is forthcoming."
                )
            elif isinstance(body, ast.Return):
                if found_return:
                    raise ValueError(
                        "Workflow python definitions must have exactly one return."
                    )
                found_return = True
                # Sets state: outputs, output_edges
                self.handle_return(func, body, output_labels)
            else:
                raise TypeError(
                    f"Workflow python definitions can only interpret assignments, a subset "
                    f"of flow control (for/while/if/try) and a return, but ast found "
                    f"{type(body)} {body.value if hasattr(body, 'value') else ''}"
                )

        if not found_return:
            raise ValueError(
                "Workflow python definitions must have a return statement."
            )

    def walk_for(
        self, tree: ast.For, scope: scope_helpers.ScopeProxy, accumulators: set[str]
    ) -> dict[str, str]:
        used_accumulators: list[str] = []
        used_accumulator_source_map: dict[str, str] = {}

        for body in tree.body:
            if isinstance(body, ast.Assign | ast.AnnAssign):
                self.handle_assign(body, scope)
            elif isinstance(body, ast.For):
                self.handle_for(body, scope)
            elif isinstance(body, ast.While | ast.If | ast.Try):
                raise NotImplementedError(
                    f"Support for control flow statement {type(body)} is forthcoming."
                )
            elif isinstance(body, ast.Expr):
                used_accumulator, appended_symbol = (
                    self.handle_appending_to_accumulator(body, accumulators)
                )
                used_accumulators.append(used_accumulator)
                used_accumulator_source_map[used_accumulator] = appended_symbol
            else:
                raise TypeError(
                    f"Workflow python definitions can only interpret assignments, a subset "
                    f"of flow control (for/while/if/try) and a return, but ast found "
                    f"{type(body)}"
                )

        if len(used_accumulators) == 0:
            raise ValueError("For loops must use up at least one accumulator symbol.")
        base_models.validate_unique(
            used_accumulators,
            f"Each accumulator may be appended to at most once, but appended "
            f"to: {used_accumulators}",
        )

        return used_accumulator_source_map

    def enforce_unique_symbols(self, new_symbols: Iterable[str]) -> None:
        if overshadow := set(self.symbol_to_source_map).intersection(new_symbols):
            raise ValueError(
                f"Workflow python definitions must not re-use symbols, but found "
                f"duplicate(s) {overshadow}"
            )

    def handle_assign(
        self, body: ast.Assign | ast.AnnAssign, scope: scope_helpers.ScopeProxy
    ):
        # Get returned symbols from the left-hand side
        lhs = body.targets[0] if isinstance(body, ast.Assign) else body.target
        new_symbols = parser_helpers.resolve_symbols_to_strings(lhs)
        self.enforce_unique_symbols(new_symbols)

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

        self.symbol_to_source_map.update(
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
        nested_iters, zipped_iters, for_tree = parse_for_iterations(tree)
        all_iters = nested_iters + zipped_iters

        body_parser = WorkflowParser(
            symbol_to_source_map=remap_to_iterating_symbols(
                self.symbol_to_source_map, all_iters
            )
        )
        used_accumulator_symbol_map = body_parser.walk_for(
            for_tree, scope, self._for_loop_accumulators
        )
        broadcast_body_input_symbols = [
            s
            for s in body_parser.inputs
            if s not in set(used_accumulator_symbol_map.values())
            and s not in {iterating_symbol for iterating_symbol, _ in all_iters}
        ]  # Need to keep it consistently ordered, so don't use a simple set op

        body_label = "body"
        for_body_node = helper_models.LabeledNode(
            label=body_label, node=body_parser.build_model()
        )
        for_inputs = broadcast_body_input_symbols + [pair[1] for pair in all_iters]
        for_outputs = list(used_accumulator_symbol_map)
        broadcast_inputs = {
            edge_models.TargetHandle(node=body_label, port=port): (
                edge_models.InputSource(port=port)
            )
            for port in broadcast_body_input_symbols
        }
        scattered_inputs = {
            edge_models.TargetHandle(
                node=body_label, port=body_port
            ): edge_models.InputSource(port=for_port)
            for body_port, for_port in all_iters
        }
        for_input_edges = broadcast_inputs | scattered_inputs
        nested_ports = [iterated for iterated, _ in nested_iters]
        zipped_ports = [iterated for iterated, _ in zipped_iters]

        for_output_edges = {}
        for_transfer_edges = {}
        for accumulator_symbol, appended_symbol in used_accumulator_symbol_map.items():
            target = edge_models.OutputTarget(port=accumulator_symbol)
            if appended_symbol in body_parser.outputs:
                for_output_edges[target] = edge_models.SourceHandle(
                    node=body_label, port=appended_symbol
                )
            else:
                for_transfer_edges[target] = for_input_edges[
                    edge_models.TargetHandle(node=body_label, port=appended_symbol)
                ]
        for_node = for_model.ForNode(
            body_node=for_body_node,
            inputs=for_inputs,
            outputs=for_outputs,
            input_edges=for_input_edges,
            output_edges=for_output_edges,
            nested_ports=nested_ports,
            zipped_ports=zipped_ports,
            transfer_edges=for_transfer_edges,
        )
        for_label = label_helpers.unique_suffix("for", self.nodes)
        self.nodes[for_label] = for_node
        self._for_loop_accumulators -= set(used_accumulator_symbol_map)

        labeled_for = helper_models.LabeledNode(label=for_label, node=for_node)
        self.symbol_to_source_map.update(
            self._get_symbol_sources_from_child_output(
                new_symbols=list(used_accumulator_symbol_map),
                child=labeled_for,
            )
        )

        for broadcast_symbol in broadcast_body_input_symbols:
            self.input_edges[
                edge_models.TargetHandle(node=for_label, port=broadcast_symbol)
            ] = edge_models.InputSource(port=broadcast_symbol)
            if broadcast_symbol not in self.inputs:
                if parsing_function_def:
                    raise ValueError(
                        f"A for-loop broadcast the symbol {broadcast_symbol}, but it "
                        f"is not available in the inputs: {self.inputs}"
                    )
                self.inputs.append(broadcast_symbol)
        for _, iterated_symbol in all_iters:
            source = self.symbol_to_source_map[iterated_symbol]
            if isinstance(source, edge_models.InputSource):
                self.input_edges[
                    edge_models.TargetHandle(node=for_label, port=iterated_symbol)
                ] = source
            elif isinstance(source, edge_models.SourceHandle):
                self.edges[
                    edge_models.TargetHandle(node=for_label, port=iterated_symbol)
                ] = source

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


def parse_for_iterations(
    for_stmt: ast.For,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], ast.For]:
    """
    Parse for loop iteration structure, handling zip and immediately nested loops.

    Returns (nested_iterations, zipped_iterations) where each is a list of
    (variable_name, source_symbol) tuples.
    """
    nested: list[tuple[str, str]] = []
    zipped: list[tuple[str, str]] = []

    current = for_stmt
    while isinstance(current, ast.For):
        is_zip, pairs = _parse_single_for_header(current)

        if is_zip:
            zipped.extend(pairs)
        else:
            nested.extend(pairs)

        # Check for nested for loop (single statement that's another For)
        if len(current.body) >= 1 and isinstance(current.body[0], ast.For):
            current = current.body[0]
        else:
            break

    return nested, zipped, current


def _parse_single_for_header(
    for_stmt: ast.For,
) -> tuple[bool, list[tuple[str, str]]]:
    """
    Parse a single for loop header.

    Returns (is_zipped, [(var, source), ...]).
    """
    iter_expr = for_stmt.iter
    target = for_stmt.target

    # Check for zip()
    if isinstance(iter_expr, ast.Call) and _is_zip_call(iter_expr):
        if not isinstance(target, ast.Tuple):
            raise ValueError("zip() iteration requires tuple unpacking")

        vars_list = [elt.id for elt in target.elts if isinstance(elt, ast.Name)]
        if len(vars_list) != len(target.elts):
            raise ValueError("zip() iteration targets must be simple names")

        sources = []
        for arg in iter_expr.args:
            if not isinstance(arg, ast.Name):
                raise ValueError("zip() arguments must be simple symbols")
            sources.append(arg.id)

        if len(vars_list) != len(sources):
            raise ValueError(
                f"zip() variable count ({len(vars_list)}) must match "
                f"argument count ({len(sources)})"
            )

        return True, list(zip(vars_list, sources, strict=True))

    # Simple iteration: for x in xs
    if not isinstance(iter_expr, ast.Name):
        raise ValueError(
            "For loop must iterate over a symbol (not an inline expression)"
        )

    if isinstance(target, ast.Name):
        return False, [(target.id, iter_expr.id)]
    elif isinstance(target, ast.Tuple):
        # for a, b in items (tuple unpacking without zip)
        raise ValueError(
            "Tuple unpacking in for loops requires zip(). "
            "Use 'for a, b in zip(as, bs):' instead of 'for a, b in items:'"
        )
    else:
        raise ValueError(f"Unsupported for loop target: {type(target)}")


def _is_zip_call(node: ast.Call) -> bool:
    """Check if a Call node is a call to zip()."""
    return isinstance(node.func, ast.Name) and node.func.id == "zip"


def is_append_call(node: ast.expr | ast.Expr, accumulators: set[str]) -> bool:
    """Check if node is an append call to a known accumulator."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "append"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in accumulators
    )


def remap_to_iterating_symbols(
    symbol_to_source_map, iterated_ports: list[tuple[str, str]]
) -> SymbolSourceMapType:
    """Where a symbol is iterated over, update the map to use the iterating symbol"""
    source_to_loop_var = {src: var for var, src in iterated_ports}
    return {
        (k := source_to_loop_var.get(key, key)): edge_models.InputSource(port=k)
        for key in symbol_to_source_map
    }
