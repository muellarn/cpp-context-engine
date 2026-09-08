"""Safe filesystem-backed source reading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cpp_context_engine.models import CodeSymbol
from cpp_context_engine.source_paths import SourceBoundary


class SourceReadError(RuntimeError):
    """Raised when a requested source excerpt cannot be read safely."""


@dataclass(frozen=True, slots=True)
class FilesystemSourceReader:
    """Read symbol spans while preventing paths from escaping a project root."""

    project_root: Path
    encoding: str = "utf-8"
    max_file_bytes: int = 8 * 1024 * 1024
    generated_source_roots: tuple[Path, ...] = ()
    allowed_paths: frozenset[Path] | None = None

    def __post_init__(self) -> None:
        boundary = SourceBoundary(self.project_root, self.generated_source_roots)
        object.__setattr__(self, "project_root", boundary.project_root)
        object.__setattr__(self, "generated_source_roots", boundary.generated_source_roots)
        if self.generated_source_roots and self.allowed_paths is None:
            raise ValueError("generated source reads require exact persisted source provenance")
        if self.allowed_paths is not None:
            object.__setattr__(
                self,
                "allowed_paths",
                frozenset(path.resolve(strict=False) for path in self.allowed_paths),
            )

    def read_symbol(self, symbol: CodeSymbol) -> str:
        resolved = self._resolve_allowed(symbol.span.path)
        requested = symbol.span.path

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

    def display_path(self, path: Path) -> str:
        """Render one known source path without exposing an absolute host location."""

        resolved = self._resolve_allowed(path)
        return SourceBoundary(self.project_root, self.generated_source_roots).display(resolved)

    def _resolve_allowed(self, requested: Path) -> Path:
        boundary = SourceBoundary(self.project_root, self.generated_source_roots)
        root = boundary.project_root
        try:
            resolved = boundary.canonical_file(
                requested if requested.is_absolute() else root / requested
            )
        except ValueError:
            raise SourceReadError(f"source path escapes project root: {requested}") from None
        if self.allowed_paths is not None and resolved not in self.allowed_paths:
            raise SourceReadError("source file is not present in the selected build provenance")
        return resolved
