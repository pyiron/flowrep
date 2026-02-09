from collections.abc import Mapping

from flowrep.models import edge_models
from flowrep.models.nodes import helper_models


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
