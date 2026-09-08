"""Canonical project and explicitly authorized generated-source boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def canonical_generated_source_roots(
    compilation_database: Path, roots: tuple[Path, ...]
) -> tuple[Path, ...]:
    """Resolve build-relative roots and reject nonexistent or dangerously broad roots."""

    database = compilation_database.expanduser().resolve(strict=False)
    canonical: set[Path] = set()
    for raw in roots:
        candidate = raw.expanduser()
        if not candidate.is_absolute():
            candidate = database.parent / candidate
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as error:
            raise ValueError("generated source root must exist") from error
        if not resolved.is_dir():
            raise ValueError("generated source root must be a directory")
        if resolved == Path(resolved.anchor):
            raise ValueError("generated source root must not be a filesystem root")
        canonical.add(resolved)
    return tuple(sorted(canonical, key=lambda path: path.as_posix()))


@dataclass(frozen=True, slots=True)
class SourceBoundary:
    """One source tree plus canonical generated trees authorized by one build."""

    project_root: Path
    generated_source_roots: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        try:
            project_root = self.project_root.resolve(strict=True)
        except OSError as error:
            raise ValueError("project source root must exist") from error
        if not project_root.is_dir() or project_root == Path(project_root.anchor):
            raise ValueError("project source root must be a bounded directory")
        generated_roots = tuple(path.resolve(strict=True) for path in self.generated_source_roots)
        if any(not path.is_dir() or path == Path(path.anchor) for path in generated_roots):
            raise ValueError("generated source root must be a bounded directory")
        object.__setattr__(self, "project_root", project_root)
        object.__setattr__(
            self,
            "generated_source_roots",
            generated_roots,
        )

    @property
    def roots(self) -> tuple[Path, ...]:
        return (self.project_root, *self.generated_source_roots)

    def contains(self, path: Path) -> bool:
        candidate = path.resolve(strict=False)
        return any(candidate.is_relative_to(root) for root in self.roots)

    def canonical_file(self, path: Path) -> Path:
        try:
            candidate = path.resolve(strict=True)
        except OSError as error:
            raise ValueError("source file must exist") from error
        if not candidate.is_file() or not self.contains(candidate):
            raise ValueError("source file is outside the authorized source roots")
        return candidate

    def display(self, path: Path) -> str:
        candidate = path.resolve(strict=False)
        if candidate.is_relative_to(self.project_root):
            return candidate.relative_to(self.project_root).as_posix()
        for index, root in enumerate(self.generated_source_roots):
            if candidate.is_relative_to(root):
                relative = candidate.relative_to(root).as_posix()
                return f"@generated/{index}/{relative}"
        raise ValueError("source file is outside the authorized source roots")
