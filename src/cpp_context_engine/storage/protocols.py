"""Persistence interfaces independent of a specific database."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from cpp_context_engine.models import CodeSymbol


class SymbolStore(Protocol):
    def put_symbols(self, symbols: Iterable[CodeSymbol]) -> None:
        """Insert or replace normalized symbols."""
        ...

    def get_symbol(self, symbol_id: str) -> CodeSymbol | None:
        """Return one normalized symbol by stable identifier."""
        ...
