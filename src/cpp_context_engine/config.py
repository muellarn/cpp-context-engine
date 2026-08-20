"""Application configuration with explicit ``CPP_CONTEXT_*`` environment overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Resolved paths, provider settings, and bounded runtime limits."""

    project_root: Path
    index_directory: Path
    database_path: Path | None = None
    compilation_database: Path | None = None
    libclang_library_file: Path | None = None
    max_context_tokens: int = 16_000
    retrieval_limit: int = 20
    embedding_provider: str = "local"
    embedding_base_url: str | None = None
    embedding_model: str | None = None
    embedding_api_key: str | None = field(default=None, repr=False)
    embedding_dimensions: int = 384
    provider_timeout_seconds: float = 30.0
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_api_key: str | None = field(default=None, repr=False)
    serve_host: str = "127.0.0.1"
    serve_port: int = 8000

    def __post_init__(self) -> None:
        root = self.project_root.expanduser().resolve(strict=False)
        index = self.index_directory.expanduser().resolve(strict=False)
        object.__setattr__(self, "project_root", root)
        object.__setattr__(self, "index_directory", index)
        object.__setattr__(
            self,
            "database_path",
            (self.database_path or index / "index.db").expanduser().resolve(strict=False),
        )
        object.__setattr__(
            self,
            "compilation_database",
            (self.compilation_database or root / "build" / "compile_commands.json")
            .expanduser()
            .resolve(strict=False),
        )
        if self.libclang_library_file is not None:
            object.__setattr__(
                self,
                "libclang_library_file",
                self.libclang_library_file.expanduser().resolve(strict=False),
            )
        if self.embedding_provider not in {"local", "openai"}:
            raise ValueError("embedding provider must be 'local' or 'openai'")
        if (
            min(
                self.max_context_tokens,
                self.retrieval_limit,
                self.embedding_dimensions,
                self.serve_port,
            )
            <= 0
        ):
            raise ValueError("token, retrieval, embedding, and port limits must be positive")
        if self.serve_port > 65_535:
            raise ValueError("serve port must not exceed 65535")
        if self.provider_timeout_seconds <= 0:
            raise ValueError("provider timeout must be positive")

    @classmethod
    def from_environment(cls, *, cwd: Path | None = None) -> AppConfig:
        """Read configuration without printing or otherwise exposing secret values."""

        base = (cwd or Path.cwd()).resolve()
        project_root = _path_env("CPP_CONTEXT_PROJECT_ROOT", base)
        index_directory = _path_env("CPP_CONTEXT_INDEX_DIRECTORY", project_root / ".cpp-context")
        return cls(
            project_root=project_root,
            index_directory=index_directory,
            database_path=_optional_path_env("CPP_CONTEXT_DATABASE"),
            compilation_database=_optional_path_env("CPP_CONTEXT_COMPILE_COMMANDS"),
            libclang_library_file=_optional_path_env("LIBCLANG_LIBRARY_FILE"),
            max_context_tokens=_positive_int("CPP_CONTEXT_MAX_TOKENS", 16_000),
            retrieval_limit=_positive_int("CPP_CONTEXT_RETRIEVAL_LIMIT", 20),
            embedding_provider=os.getenv("CPP_CONTEXT_EMBEDDING_PROVIDER", "local").casefold(),
            embedding_base_url=os.getenv("CPP_CONTEXT_EMBEDDING_BASE_URL"),
            embedding_model=os.getenv("CPP_CONTEXT_EMBEDDING_MODEL"),
            embedding_api_key=os.getenv("CPP_CONTEXT_EMBEDDING_API_KEY"),
            embedding_dimensions=_positive_int("CPP_CONTEXT_EMBEDDING_DIMENSIONS", 384),
            provider_timeout_seconds=_positive_float("CPP_CONTEXT_PROVIDER_TIMEOUT", 30.0),
            llm_base_url=os.getenv("CPP_CONTEXT_LLM_BASE_URL"),
            llm_model=os.getenv("CPP_CONTEXT_LLM_MODEL"),
            llm_api_key=os.getenv("CPP_CONTEXT_LLM_API_KEY"),
            serve_host=os.getenv("CPP_CONTEXT_SERVE_HOST", "127.0.0.1"),
            serve_port=_positive_int("CPP_CONTEXT_SERVE_PORT", 8000),
        )


def _path_env(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default))).expanduser().resolve(strict=False)


def _optional_path_env(name: str) -> Path | None:
    value = os.getenv(name)
    return Path(value) if value else None


def _positive_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    try:
        value = default if raw_value is None else int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _positive_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    try:
        value = default if raw_value is None else float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value
