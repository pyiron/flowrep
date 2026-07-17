"""Parse data-access chains on workflow data into injected ``std`` nodes.

A chain rooted at a known *symbol* -- ``dc.a``, ``d[k]``, ``dc.sub["item"].val`` --
becomes one node per link: ``std.get_attr`` for an attribute link, ``std.getitem`` for
an item link. The symbol map deliberately shadows the object scope, so a workflow input
named ``os`` makes ``os.path`` an attribute access on that input, exactly as it would be
at runtime.

``chain_parser`` owns the walk; a :class:`LinkHandler` owns only what differs between
the two access syntaxes Python has. That set is closed, so the dispatch is total.

Data access is allowed anywhere a value is, with one exception: the workflow's
``return``. The rule is

    a port name may be *generated* when the port is internal to the recipe; it must
    come from a *symbol* only when the port is public IO.

Ports get their names from four places:

===============================  ==========================================
Port                             Name comes from
===============================  ==========================================
Workflow output port             the returned symbol
Flow-control input port          the enclosing symbol feeding it, or -- for a
                                 constant or an access chain, which have no
                                 symbol -- a generated name (see
                                 :func:`generate_port_name`)
For-body output port             the appended symbol, or a generated name
Atomic/workflow node input port  the *callee's* own parameter
===============================  ==========================================

Only the first is public IO. A workflow's output ports are its interface, and a
consumer wiring into them should be able to read their names straight off the source,
so ``return dc.a`` is refused (see :func:`reject_unbound_access`). Everywhere else the
port is internal wiring, and a generated name costs the reader nothing: the compiler
emits ``val_0 = x.val`` and re-parsing regenerates exactly that port, so the sugared
form and the bound form are not merely equivalent recipes -- they are the same recipe.

(``@atomic`` stays laxer about its own returns. That asymmetry is deliberate: atomic
parsing must swallow arbitrary Python functions, including ones the caller did not
write, whereas a ``@workflow`` author controls their own source.)

Two things a generated name cannot rescue, both refused elsewhere: a method call on
workflow data (``dc.method(x)``), which needs a callable reference rather than a port
name, and a chain in a ``while`` condition that depends on a symbol the loop body
reassigns (see :func:`while_parser._reject_looped_chain_symbols`), where hoisting the
access out of the loop would silently diverge from Python's per-iteration re-read.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from typing import Protocol

from flowrep import edge_models
from flowrep.parsers import attribute_parser, item_parser, label_helpers, symbol_scope
from flowrep.prospective import atomic_recipe, union_types


class LinkHandler(Protocol):
    """How one kind of chain link (``.attr`` or ``[key]``) becomes a node."""

    @property
    def recipe(self) -> atomic_recipe.AtomicRecipe:
        """The std recipe injected per link. Two inputs (obj, key), one output."""
        ...

    def label_base(self, link: ast.expr) -> str:
        """The node label prefix, before :func:`label_helpers.unique_suffix`."""
        ...

    def port_base(self, link: ast.expr) -> str:
        """The generated port name base, before :func:`label_helpers.unique_suffix`."""
        ...

    def feed_key(
        self,
        link: ast.expr,
        symbol_map: symbol_scope.SymbolScope,
        nodes: union_types.Recipes,
        label: str,
        port: str,
    ) -> None:
        """Wire this link's second input: the attribute name, or the item key."""
        ...


_HANDLERS: dict[type[ast.expr], LinkHandler] = {
    ast.Attribute: attribute_parser.HANDLER,
    ast.Subscript: item_parser.HANDLER,
}


def _handler_for(link: ast.expr) -> LinkHandler:
    try:
        return _HANDLERS[type(link)]
    except KeyError:  # pragma: no cover - callers gate on is_data_access
        raise TypeError(f"Not a data-access link: {ast.dump(link)}") from None


def chain_root(node: ast.expr) -> ast.Name | None:
    """The innermost ``ast.Name`` of a pure access chain, else ``None``."""
    while isinstance(node, ast.Attribute | ast.Subscript):
        node = node.value
    return node if isinstance(node, ast.Name) else None


def _chain_links(node: ast.expr) -> list[ast.expr]:
    """The chain's links, ordered root-outward."""
    links: list[ast.expr] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute | ast.Subscript):
        links.append(current)
        current = current.value
    links.reverse()
    return links


def is_data_access(node: ast.expr, symbol_map: symbol_scope.SymbolScope) -> bool:
    """True if *node* is attribute or item access rooted at a symbol known to the graph.

    The symbol map deliberately shadows the object scope: a workflow input named
    ``os`` makes ``os.path`` a data attribute access on that input, not a module
    lookup, exactly as it would be at runtime.
    """
    if not isinstance(node, ast.Attribute | ast.Subscript):
        return False
    root = chain_root(node)
    return root is not None and root.id in symbol_map


def inject_chain(
    node: ast.expr,
    symbol_map: symbol_scope.SymbolScope,
    nodes: union_types.Recipes,
) -> edge_models.SourceHandle:
    """Add one node (plus any constant peer) per link of *node*.

    Walks the chain root-outward so that node insertion order -- and therefore
    labels and the enclosing scope's input ordering -- are a deterministic
    function of the source. Returns the handle of the outermost link's output.

    *node* must be a data access chain (see :func:`is_data_access`); callers are
    expected to gate on that before calling.
    """
    root = chain_root(node)
    if root is None:  # pragma: no cover - callers gate on is_data_access
        raise TypeError(f"Not an access chain rooted at a symbol: {ast.dump(node)}")
    if root.id in symbol_map.all_accumulators:
        raise ValueError(
            f"Cannot take an attribute or item of accumulator '{root.id}'. "
            f"Accumulators are name-tracked list declarations, not graph data; append "
            f"to it and take attributes of the resulting collection outside the loop "
            f"instead."
        )

    handle: edge_models.SourceHandle | None = None
    for link in _chain_links(node):
        handler = _handler_for(link)
        recipe = handler.recipe
        obj_port, key_port = recipe.inputs
        (out_port,) = recipe.outputs
        label = label_helpers.unique_suffix(handler.label_base(link), nodes)
        nodes[label] = recipe
        if handle is None:
            symbol_map.consume(root.id, label, obj_port)
        else:
            symbol_map.consume_source(handle, label, obj_port)
        handler.feed_key(link, symbol_map, nodes, label, key_port)
        handle = edge_models.SourceHandle(node=label, port=out_port)
    assert handle is not None  # links is non-empty: node is a link node
    return handle


def hoist_call_arguments(
    call: ast.Call,
    symbol_map: symbol_scope.SymbolScope,
    nodes: union_types.Recipes,
) -> dict[ast.expr, edge_models.SourceHandle]:
    """Inject every data-access argument of *call* before the call's own node.

    Hoisting is what makes ``f(x0, comp.val)`` and ``v = comp.val; f(x0, v)`` parse
    to identical recipes: the injected nodes are created (and their sources consumed)
    ahead of the call in both forms.
    """
    hoisted: dict[ast.expr, edge_models.SourceHandle] = {}
    arguments = list(call.args) + [kw.value for kw in call.keywords]
    for argument in arguments:
        if is_data_access(argument, symbol_map):
            hoisted[argument] = inject_chain(argument, symbol_map, nodes)
    return hoisted


def generate_port_name(node: ast.expr, taken: Iterable[str]) -> str:
    """A deterministic port name for an access chain that has no symbol.

    Takes the outermost link's base -- for an attribute, the attribute name; for an
    item, the ``std.getitem`` output port -- and unique-suffixes it against *taken*.
    ``x.a.val`` gives ``val_0``; ``x["a"]`` gives ``item_0``.

    *taken* must include every symbol in the enclosing scope, not merely the ports
    allocated so far: a flow-control node's inputs are its condition's inputs *plus
    its body's*, and a body input port is always an enclosing symbol.
    """
    return label_helpers.unique_suffix(_handler_for(node).port_base(node), taken)


def reject_unbound_access(
    node: ast.expr, symbol_map: symbol_scope.SymbolScope, context: str
) -> None:
    """Raise if *node* is an access chain used where the port name is public IO.

    A workflow's output ports are its interface. Flowrep names them after the returned
    symbol, and an access chain has no symbol -- so the port would carry a generated
    name that a consumer cannot read off the source. Everywhere else in a workflow a
    generated name is fine (the port is internal wiring); here it is not, so we ask for
    the binding.
    """
    if not is_data_access(node, symbol_map):
        return
    chain = ast.unparse(node)
    raise ValueError(
        f"Attribute or item access must be bound to a symbol before it can be "
        f"{context}: got {chain!r}. The port name is taken from the symbol, and "
        f"{chain!r} has no symbol to take it from. Bind it first -- e.g. "
        f"`my_value = {chain}` -- and use `my_value` instead."
    )


def dependency_symbols(node: ast.expr) -> set[str]:
    """Every symbol an access chain reads: its root, plus any symbol used as an item key.

    An attribute name is a constant, but an item key can be a symbol, so ``d[k]``
    depends on both ``d`` and ``k``. The ``while`` guard needs the distinction: a chain
    that depends on a symbol the loop body reassigns cannot be hoisted faithfully.
    """
    root = chain_root(node)
    symbols = set() if root is None else {root.id}
    for link in _chain_links(node):
        if isinstance(link, ast.Subscript) and isinstance(link.slice, ast.Name):
            symbols.add(link.slice.id)
    return symbols


def reject_method_call(call: ast.Call, symbol_map: symbol_scope.SymbolScope) -> None:
    """Raise if *call* invokes an attribute or item of a known symbol."""
    if is_data_access(call.func, symbol_map):
        chain = ast.unparse(call.func)
        raise ValueError(
            f"Workflow python definitions cannot call methods on workflow data: "
            f"'{chain}(...)'. Bind the attribute or item to a symbol and call it via "
            f"a node, or wrap the method call in an @atomic function."
        )
