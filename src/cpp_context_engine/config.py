"""Application configuration with environment-variable overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Paths and limits shared by adapters and orchestration code."""

    project_root: Path
    index_directory: Path
    max_context_tokens: int = 16_000
    retrieval_limit: int = 20

    @classmethod
    def from_environment(cls, *, cwd: Path | None = None) -> AppConfig:
        """Build configuration from ``CPP_CONTEXT_*`` environment variables."""

        base = (cwd or Path.cwd()).resolve()
        project_root = Path(os.getenv("CPP_CONTEXT_PROJECT_ROOT", base)).expanduser().resolve()
        default_index = project_root / ".cpp-context"
        index_directory = (
            Path(os.getenv("CPP_CONTEXT_INDEX_DIRECTORY", default_index)).expanduser().resolve()
        )
        return cls(
            project_root=project_root,
            index_directory=index_directory,
            max_context_tokens=_positive_int("CPP_CONTEXT_MAX_TOKENS", 16_000),
            retrieval_limit=_positive_int("CPP_CONTEXT_RETRIEVAL_LIMIT", 20),
        )


def _positive_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    value = default if raw_value is None else int(raw_value)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value
