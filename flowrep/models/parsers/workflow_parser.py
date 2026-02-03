import ast
import builtins
import inspect
from collections.abc import Callable, Collection, Iterable, Iterator
from types import FunctionType
from typing import cast

from flowrep.models import base_models, edge_models
from flowrep.models.nodes import helper_models, union, workflow_model
from flowrep.models.parsers import atomic_parser, label_helpers, parser_helpers


def workflow(
    func: FunctionType | str | None = None,
    /,
    *output_labels: str,
) -> FunctionType | Callable[[FunctionType], FunctionType]:
    parsed_labels: tuple[str, ...]
    if isinstance(func, FunctionType):
        # Direct decoration: @workflow
        parsed_labels = ()
        target_func = func
    elif func is not None and not isinstance(func, str):
        raise TypeError(
            f"@workflow can only decorate functions, got {type(func).__name__}"
        )
    else:
        # Called with args: @workflow(...) or @workflow("label", ...)
        parsed_labels = (func,) + output_labels if func is not None else output_labels
        target_func = None

    def decorator(f: FunctionType) -> FunctionType:
        parser_helpers.ensure_function(f, "@workflow")
        f.flowrep_recipe = parse_workflow(f, *parsed_labels)  # type: ignore[attr-defined]
        return f

    return decorator(target_func) if target_func else decorator


def parse_workflow(
    func: FunctionType,
    *output_labels: str,
):
    scope = get_scope(func)
    state = _WorkflowParserState(inputs=label_helpers.get_input_labels(func))

    tree = parser_helpers.get_ast_function_node(func)
    found_return = False
    for body in tree.body:
        if isinstance(body, ast.Assign | ast.AnnAssign):
            # Get returned symbols from the left-hand side
            lhs = body.targets[0] if isinstance(body, ast.Assign) else body.target
            new_symbols = resolve_symbols_to_strings(lhs)
            state.enforce_unique_symbols(new_symbols)

            rhs = body.value
            if isinstance(rhs, ast.Call):
                # Make a new node from the rhs
                # Modifies state: nodes, input_edges, edges, symbol_to_source_map
                state.handle_call_assignment(rhs, new_symbols, scope)
            elif isinstance(rhs, ast.List) and len(rhs.elts) == 0:
                raise NotImplementedError(
                    "Assigning empty will probably be lists will probably be used for "
                    "coordinating loop aggregators in the future, but is not yet "
                    "supported."
                )
            else:
                raise ValueError(
                    f"Workflow python definitions can only interpret assignments with "
                    f"a call on the right-hand-side, but ast found {type(rhs)}"
                )
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


class ScopeProxy:
    """
    Make the __dict__-like scope dot-accessible without duplicating the dictionary
    like types.SimpleNamespace would.
    """

    def __init__(self, d: dict):
        self._d = d

    def __getattr__(self, name: str):
        try:
            return self._d[name]
        except KeyError:
            raise AttributeError(name) from None


class _WorkflowParserState:
    outputs: list[str]
    output_edges: edge_models.OutputEdges

    def __init__(self, inputs: list[str]):
        self.inputs = inputs
        self.nodes: union.Nodes = {}
        self.input_edges: edge_models.InputEdges = {}
        self.edges: edge_models.Edges = {}

        self._symbol_to_source_map: dict[
            str, edge_models.InputSource | edge_models.SourceHandle
        ] = {p: edge_models.InputSource(port=p) for p in inputs}

    def enforce_unique_symbols(self, new_symbols: Iterable[str]) -> None:
        if overshadow := set(self._symbol_to_source_map).intersection(new_symbols):
            raise ValueError(
                f"Workflow python definitions must not re-use symbols, but found "
                f"duplicate(s) {overshadow}"
            )

    def handle_call_assignment(
        self,
        rhs: ast.Call,
        new_symbols: list[str],
        scope: ScopeProxy,
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
        ast_call: ast.Call, existing_names: Iterable[str], scope: ScopeProxy
    ) -> helper_models.LabeledNode:
        child_call = cast(FunctionType, resolve_symbol_to_object(ast_call.func, scope))
        # Since it is the .func attribute of an ast.Call,
        # the retrieved object had better be a function
        child_recipe = (
            child_call.flowrep_recipe
            if hasattr(child_call, "flowrep_recipe")
            else atomic_parser.parse_atomic(child_call)
        )
        child_name = unique_suffix(child_call.__name__, existing_names)
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
            else:
                raise TypeError(f"Unexpected edge source: {type(source)}")

    def handle_return(
        self,
        func: FunctionType,
        body: ast.Return,
        output_labels: Collection[str],
    ) -> None:
        returned_symbols = resolve_symbols_to_strings(body.value)
        base_models.validate_unique(
            returned_symbols,
            message=f"Workflow python definitions must have unique returns, but "
            f"got duplicates in: {returned_symbols}",
        )

        annotated_returns = label_helpers.get_annotated_output_labels(func)
        if annotated_returns is None:
            scraped_labels = returned_symbols
        else:
            if len(annotated_returns) != len(returned_symbols):
                raise ValueError(
                    f"Annotation labels ({annotated_returns}), and returned symbols "
                    f"({returned_symbols}) must have the same length."
                )

            scraped_labels = list(
                ann if ann is not None else ret
                for ann, ret in zip(annotated_returns, returned_symbols, strict=True)
            )

        if output_labels and len(output_labels) != len(returned_symbols):
            raise ValueError(
                f"When output_labels are specified ({output_labels}), workflow "
                f"python definitions have a matching number of returned symbols "
                f"({returned_symbols})."
            )

        self.outputs = list(output_labels) or scraped_labels

        self.output_edges: edge_models.OutputEdges = {}
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


def unique_suffix(name: str, references: Iterable[str]) -> str:
    # This is obviously horribly inefficient, but fix that later
    i = 0
    new_name = f"{name}_{i}"
    while new_name in references:
        i += 1
        new_name = f"{name}_{i}"
    return new_name


def get_scope(func: FunctionType) -> ScopeProxy:
    return ScopeProxy(inspect.getmodule(func).__dict__ | vars(builtins))


def resolve_symbol_to_object(
    node: ast.expr,  # Expecting a Name or Attribute here, and will otherwise TypeError
    scope: ScopeProxy | object,
    _chain: list[str] | None = None,
) -> object:
    """ """
    _chain = _chain or []
    error_suffix = f" while attempting to resolve the symbol chain '{'.'.join(_chain)}'"
    if isinstance(node, ast.Name):
        attr = node.id
        try:
            obj = getattr(scope, attr)
            for attr in _chain:
                obj = getattr(obj, attr)
            return obj
        except AttributeError as e:
            raise ValueError(f"Could not find attribute '{attr}' {error_suffix}") from e
    elif isinstance(node, ast.Attribute):
        return resolve_symbol_to_object(node.value, scope, [node.attr] + _chain)
    else:
        raise TypeError(
            f"Cannot resolve symbol {node} {error_suffix}. "
            f"Expected an ast.Name or chain of ast.Attribute and ast.Name, but got "
            f"{node}."
        )


def resolve_symbols_to_strings(
    node: (
        ast.expr | None
    ),  # Expecting a Name or Tuple[Name], and will otherwise TypeError
) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    elif isinstance(node, ast.Tuple) and all(
        isinstance(elt, ast.Name) for elt in node.elts
    ):
        return [cast(ast.Name, elt).id for elt in node.elts]
    else:
        raise TypeError(
            f"Expected to receive a symbol or tuple of symbols from ast.Name or "
            f"ast.Tuple, but could not parse this from {type(node)}."
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
        if not isinstance(kw.arg, str):
            raise TypeError(
                "How did you get here? A `None` value should be possible for "
                "**kwargs, but variadics should have been excluded before "
                "this. Please raise a GitHub issue."
            )
        yield name_value.id, kw.arg
