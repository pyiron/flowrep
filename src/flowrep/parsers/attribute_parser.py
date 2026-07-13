from __future__ import annotations

import ast

from flowrep import edge_models
from flowrep.parsers import constant_parser, label_helpers, symbol_scope
from flowrep.prospective import std, union_types

_OBJ_PORT = "obj"
_NAME_PORT = "name"
_ATTR_PORT = "attr"


def chain_root(node: ast.expr) -> ast.Name | None:
    """The innermost ``ast.Name`` of a pure Attribute/Name chain, else ``None``."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return node if isinstance(node, ast.Name) else None


def is_data_attribute(node: ast.expr, symbol_map: symbol_scope.SymbolScope) -> bool:
    """True if *node* is attribute access rooted at a symbol known to the graph.

    The symbol map deliberately shadows the object scope: a workflow input named
    ``os`` makes ``os.path`` a data attribute access on that input, not a module
    lookup, exactly as it would be at runtime.
    """
    if not isinstance(node, ast.Attribute):
        return False
    root = chain_root(node)
    return root is not None and root.id in symbol_map


def attribute_name(node: ast.Attribute) -> str:
    """The outermost attribute name, e.g. ``dc.a.val`` -> ``"val"``."""
    return node.attr


def inject_attribute_chain(
    node: ast.expr,
    symbol_map: symbol_scope.SymbolScope,
    nodes: union_types.Recipes,
) -> edge_models.SourceHandle:
    """Add one ``std.getattr_`` node (plus its name constant) per link of *node*.

    Walks the chain root-outward so that node insertion order -- and therefore
    labels and the enclosing scope's input ordering -- are a deterministic
    function of the source. Returns the handle of the outermost link's ``attr``
    output.

    *node* must be a data attribute chain (see :func:`is_data_attribute`); callers
    are expected to gate on that before calling.
    """
    links: list[ast.Attribute] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        links.append(current)
        current = current.value
    links.reverse()

    root = chain_root(node)
    if root is None:  # pragma: no cover - callers gate on is_data_attribute
        raise TypeError(f"Not an attribute chain rooted at a symbol: {ast.dump(node)}")
    if root.id in symbol_map.all_accumulators:
        raise ValueError(
            f"Cannot take an attribute of accumulator '{root.id}'. Accumulators are "
            f"name-tracked list declarations, not graph data; append to it and take "
            f"attributes of the resulting collection outside the loop instead."
        )

    handle: edge_models.SourceHandle | None = None
    for link in links:
        label = label_helpers.unique_suffix(f"getattr_{link.attr}", nodes)
        nodes[label] = std.getattr_.recipe
        if handle is None:
            symbol_map.consume(root.id, label, _OBJ_PORT)
        else:
            symbol_map.consume_source(handle, label, _OBJ_PORT)
        constant_parser.inject_constant(nodes, symbol_map, link.attr, label, _NAME_PORT)
        handle = edge_models.SourceHandle(node=label, port=_ATTR_PORT)
    assert handle is not None  # links is non-empty: node is an ast.Attribute
    return handle


def hoist_call_arguments(
    call: ast.Call,
    symbol_map: symbol_scope.SymbolScope,
    nodes: union_types.Recipes,
) -> dict[ast.expr, edge_models.SourceHandle]:
    """Inject every data-attribute argument of *call* before the call's own node.

    Hoisting is what makes ``f(x0, comp.val)`` and ``v = comp.val; f(x0, v)`` parse
    to identical recipes: the getattr nodes are created (and their sources consumed)
    ahead of the call in both forms.
    """
    hoisted: dict[ast.expr, edge_models.SourceHandle] = {}
    arguments = list(call.args) + [kw.value for kw in call.keywords]
    for argument in arguments:
        if is_data_attribute(argument, symbol_map):
            hoisted[argument] = inject_attribute_chain(argument, symbol_map, nodes)
    return hoisted


def reject_unbound_attribute(
    node: ast.expr, symbol_map: symbol_scope.SymbolScope, context: str
) -> None:
    """Raise if *node* is an attribute chain used where a port name must be invented.

    Flowrep names a workflow output port, a flow-control input port, and a for-body
    output port after the *symbol* supplying them. An attribute chain has no symbol,
    so there is no honest name to give the port. Rather than invent one the user
    cannot predict from reading their own source, require the binding.
    """
    if not is_data_attribute(node, symbol_map):
        return
    chain = ast.unparse(node)
    raise ValueError(
        f"Attribute access must be bound to a symbol before it can be {context}: "
        f"got {chain!r}. The port name is taken from the symbol, and {chain!r} has "
        f"no symbol to take it from. Bind it first -- e.g. `my_value = {chain}` -- "
        f"and use `my_value` instead."
    )


def reject_method_call(call: ast.Call, symbol_map: symbol_scope.SymbolScope) -> None:
    """Raise if *call* invokes an attribute of a known symbol (``dc.method(x)``)."""
    if is_data_attribute(call.func, symbol_map):
        chain = ast.unparse(call.func)
        raise ValueError(
            f"Workflow python definitions cannot call methods on workflow data: "
            f"'{chain}(...)'. Bind the attribute to a symbol and call it via a "
            f"node, or wrap the method call in an @atomic function."
        )
