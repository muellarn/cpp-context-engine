from dataclasses import replace

from test_symbol_snapshot_compression import _batch, _put

from cpp_context_engine.models import OccurrenceKind, SourceSpan, SymbolOccurrence
from cpp_context_engine.storage.sqlite import SQLiteStore


def test_repeated_occurrence_context_uses_bounded_physical_storage(tmp_path):
    count = 8192
    unit = "tu_" + "a" * 32
    symbol = "symbol_" + "b" * 32
    batch = _batch(tmp_path, unit)
    symbols = (replace(batch.symbols[0], id=symbol), batch.symbols[1])
    occurrences = tuple(
        SymbolOccurrence(
            id=f"occ_{index:032x}",
            symbol_id=symbol,
            enclosing_symbol_id="legacy unenforced enclosing ID",
            kind=OccurrenceKind.REFERENCE,
            span=SourceSpan(tmp_path / "shared-header.hpp", index + 1, index + 1, 1, 4),
            translation_unit_id=unit,
            build_configuration_id="config_" + "c" * 32,
        )
        for index in range(count)
    )
    with SQLiteStore(tmp_path / "index.db", project_root=tmp_path) as store:
        _put(store, tmp_path, replace(batch, symbols=symbols, occurrences=occurrences))
        assert store.occurrences(symbol) == occurrences
        occupied = store._connection.execute(
            "SELECT sum(pgsize) FROM dbstat WHERE name='occurrences' "
            "OR name LIKE 'occurrences_%' OR name LIKE 'occurrence_%' "
            "OR name LIKE 'sqlite_autoindex_occurrence%'"
        ).fetchone()[0]
        assert occupied <= count * 260, (occupied, count, occupied / count)
