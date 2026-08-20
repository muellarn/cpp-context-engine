"""Safe filesystem-backed source reading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cpp_context_engine.models import CodeSymbol


class SourceReadError(RuntimeError):
    """Raised when a requested source excerpt cannot be read safely."""


@dataclass(frozen=True, slots=True)
class FilesystemSourceReader:
    """Read symbol spans while preventing paths from escaping a project root."""

    project_root: Path
    encoding: str = "utf-8"
    max_file_bytes: int = 8 * 1024 * 1024

    def read_symbol(self, symbol: CodeSymbol) -> str:
        root = self.project_root.resolve()
        requested = symbol.span.path
        resolved = (requested if requested.is_absolute() else root / requested).resolve()
        if not resolved.is_relative_to(root):
            raise SourceReadError(f"source path escapes project root: {requested}")

        try:
            size = resolved.stat().st_size
        except OSError as exc:
            raise SourceReadError(f"cannot inspect source file: {requested}") from exc
        if size > self.max_file_bytes:
            raise SourceReadError(f"source file exceeds {self.max_file_bytes} bytes: {requested}")

        try:
            lines = resolved.read_text(encoding=self.encoding, errors="replace").splitlines()
        except OSError as exc:
            raise SourceReadError(f"cannot read source file: {requested}") from exc

        start = symbol.span.start_line - 1
        end = symbol.span.end_line
        if start >= len(lines):
            raise SourceReadError(f"source span starts beyond end of file: {requested}")
        return "\n".join(lines[start:end])
