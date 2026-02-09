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


class SymbolScope(Mapping[str, edge_models.InputSource | edge_models.SourceHandle]):
    """
    Tracks which symbols are in scope and where their data comes from.

    Immutable-ish: forking for child scopes (e.g. for-loop bodies) returns a new
    instance with remapped symbols.
    """

    def __init__(
        self, sources: dict[str, edge_models.InputSource | edge_models.SourceHandle]
    ):
        self._sources = dict(sources)
        self._consumptions: list[SymbolConsumption] = []

    @property
    def consumed_input_names(self) -> list[str]:
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

    # --- Mutations ---
    def register(
        self,
        new_symbols: list[str],
        child: helper_models.LabeledNode,
    ) -> None:
        """Map new symbols 1:1 to child node outputs. Enforces uniqueness."""
        if overshadow := set(self._sources) & set(new_symbols):
            raise ValueError(f"Symbol(s) already in scope: {overshadow}")
        if len(new_symbols) != len(child.node.outputs):
            raise ValueError(
                f"Cannot map {child.node.outputs} to symbols {new_symbols}"
            )
        self._sources.update(
            {
                sym: edge_models.SourceHandle(node=child.label, port=port)
                for sym, port in zip(new_symbols, child.node.outputs, strict=True)
            }
        )

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

    # --- Forking for child scopes ---
    def fork_scope(self, symbol_remap: dict[str, str]) -> "SymbolScope":
        """
        Create a child scope wherein some symbols are remapped.
        This is necessary when passing scope from one graph layer to another if the
        parent inputs have the same origin but different labels.
        """
        return SymbolScope(
            {
                (k := symbol_remap.get(key, key)): edge_models.InputSource(port=k)
                for key in self._sources
            }
        )
