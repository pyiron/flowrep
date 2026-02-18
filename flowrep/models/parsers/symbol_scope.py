import dataclasses
from collections.abc import Mapping

from flowrep.models import edge_models
from flowrep.models.nodes import helper_models


@dataclasses.dataclass(frozen=True)
class SymbolConsumption:
    symbol: str
    consumer_node: str
    consumer_port: str
    source: edge_models.InputSource | edge_models.SourceHandle


@dataclasses.dataclass(frozen=True)
class SymbolProduction:
    output_port: str
    source: edge_models.SourceHandle | edge_models.InputSource


class SymbolScope(Mapping[str, edge_models.InputSource | edge_models.SourceHandle]):
    """
    Tracks which symbols are in scope and where their data comes from.

    Immutable-ish: forking for child scopes (e.g. for-node bodies) returns a new
    instance with remapped symbols.

    Accumulators follow a three-stage lifecycle:
    - declared_accumulators: locally declared via ``acc = []``. Owned by this scope
        and passed to child scopes as available_accumulators on fork.
    - available_accumulators: inherited from the parent scope's declared_accumulators.
        These are the only accumulators a scope is allowed to ``.append()`` to.  This
        guarantees that an accumulator is only consumable one nesting level below its
        declaration, preventing grandparent accumulator access.
    - consumed_accumulators: maps ``accumulator_name → appended_symbol``. Populated by
        :meth:`use_accumulator` and read by the parent to finalise control-flow node
        outputs.
    """

    def __init__(
        self,
        sources: dict[str, edge_models.InputSource | edge_models.SourceHandle],
        available_accumulators: set[str] | None = None,
        reserved_accumulators: set[str] | None = None,
    ):
        self._sources = dict(sources)
        self._consumptions: list[SymbolConsumption] = []
        self._productions: list[SymbolProduction] = []
        self.reassigned_symbols: list[str] = []
        self.declared_accumulators: set[str] = set()
        self.available_accumulators: set[str] = (
            set() if available_accumulators is None else available_accumulators
        )
        self.reserved_accumulators: set[str] = (
            set() if reserved_accumulators is None else reserved_accumulators
        )
        self.consumed_accumulators: dict[str, str] = {}

    @property
    def inputs(self) -> list[str]:
        """Ordered unique symbols consumed from InputSources."""
        seen: set[str] = set()
        result: list[str] = []
        for c in self._consumptions:
            if isinstance(c.source, edge_models.InputSource) and c.symbol not in seen:
                seen.add(c.symbol)
                result.append(c.symbol)
            # TODO: Just set.add it?
        return result

    @property
    def input_edges(self) -> edge_models.InputEdges:
        return {
            edge_models.TargetHandle(
                node=c.consumer_node, port=c.consumer_port
            ): c.source
            for c in self._consumptions
            if isinstance(c.source, edge_models.InputSource)
        }

    @property
    def edges(self) -> edge_models.Edges:
        return {
            edge_models.TargetHandle(
                node=c.consumer_node, port=c.consumer_port
            ): c.source
            for c in self._consumptions
            if isinstance(c.source, edge_models.SourceHandle)
        }

    @property
    def output_edges(self) -> edge_models.OutputEdges:
        return {
            edge_models.OutputTarget(port=p.output_port): p.source
            for p in self._productions
        }

    @property
    def outputs(self) -> list[str]:
        """Ordered unique output port names."""
        seen: set[str] = set()
        result: list[str] = []
        for p in self._productions:
            if p.output_port not in seen:
                seen.add(p.output_port)
                result.append(p.output_port)
        return result

    # --- Mapping interface ---
    def __getitem__(
        self, key: str
    ) -> edge_models.InputSource | edge_models.SourceHandle:
        try:
            return self._sources[key]
        except KeyError:
            raise KeyError(
                f"Symbol '{key}' is not in scope. " f"Available: {list(self._sources)}"
            ) from None

    def __iter__(self):
        return iter(self._sources)

    def __len__(self):
        return len(self._sources)

    @property
    def all_accumulators(self) -> set[str]:
        return (
            self.declared_accumulators
            | self.available_accumulators
            | self.reserved_accumulators
        )

    # --- Mutations ---
    def register(
        self,
        new_symbols: list[str],
        child: helper_models.LabeledNode,
    ) -> None:
        """Map new symbols 1:1 to child node outputs. Enforces uniqueness."""
        all_accumulators = self.all_accumulators
        if overshadowed := set(new_symbols).intersection(all_accumulators):
            raise ValueError(
                f"Symbol(s) {overshadowed} already registered as accumulators."
            )
        if len(new_symbols) != len(child.node.outputs):
            raise ValueError(
                f"Cannot map {child.node.outputs} to symbols {new_symbols}"
            )
        reassigned = [s for s in new_symbols if s in self._sources]
        for symbol in reassigned:
            if symbol not in self.reassigned_symbols:
                self.reassigned_symbols.append(symbol)
        self._sources.update(
            {
                sym: edge_models.SourceHandle(node=child.label, port=port)
                for sym, port in zip(new_symbols, child.node.outputs, strict=True)
            }
        )

    def register_accumulator(self, new: str) -> None:
        if new in self._sources:
            raise ValueError(f"Accumulator symbol '{new}' already in symbol scope.")
        if new in self.declared_accumulators:
            raise ValueError(f"Accumulator symbol '{new}' already declared.")
        if new in self.available_accumulators:
            raise ValueError(
                f"Accumulator symbol '{new}' already available from parent scope."
            )
        self.declared_accumulators.add(new)

    def consume(self, symbol: str, consumer_node: str, consumer_port: str) -> None:
        """Record that `consumer_node.consumer_port` reads from `symbol`."""
        self._consumptions.append(
            SymbolConsumption(
                symbol=symbol,
                consumer_node=consumer_node,
                consumer_port=consumer_port,
                source=self[symbol],
            )
        )

    def produce(self, output_port: str, symbol: str) -> None:
        """Record that `output_port` is sourced from `symbol`."""
        if any(p.output_port == output_port for p in self._productions):
            raise ValueError(f"Output port '{output_port}' already produced.")
        self._productions.append(
            SymbolProduction(output_port=output_port, source=self[symbol])
        )

    def use_accumulator(self, accumulator_symbol: str, appended_symbol: str) -> None:
        if accumulator_symbol not in self.available_accumulators:
            raise ValueError(
                f"Could not append to the symbol {accumulator_symbol}; it is not "
                f"found among available accumulator symbols: "
                f"{self.available_accumulators}. Remember that accumulators need to be "
                f"declared in the immediate parent scope relative to their use."
            )
        self.available_accumulators.remove(accumulator_symbol)
        self.consumed_accumulators[accumulator_symbol] = appended_symbol

    # --- Forking for child scopes ---
    def fork_scope(
        self,
        symbol_remap: dict[str, str] | None = None,
        available_accumulators: set[str] | None = None,
    ) -> "SymbolScope":
        """
        Create a child scope for a nested control-flow body.

        Every symbol in the current ``_sources`` is carried over as a fresh
        :class:`InputSource` in the child.  *symbol_remap* allows renaming
        symbols in transit (e.g. a for-loop replacing the iterable symbol
        with the iteration variable).

        Accumulator propagation is controlled explicitly via
        *available_accumulators*.  For-loop bodies pass the parent's
        ``declared_accumulators`` so the body can ``.append()``; while-loop
        and if/else bodies pass ``None`` (the default) to start with an
        empty set, since those control-flow models do not support
        cross-iteration accumulation.

        The parent's ``available_accumulators`` are always added to the
        child's ``reserved_accumulators`` so that erroneous grandparent
        access is caught with a clear error rather than silently ignored.
        """
        remap = {} if symbol_remap is None else symbol_remap
        return SymbolScope(
            {
                (k := remap.get(key, key)): edge_models.InputSource(port=k)
                for key in self._sources
            },
            available_accumulators=available_accumulators,
            reserved_accumulators=self.reserved_accumulators
            | self.available_accumulators,
        )
