from __future__ import annotations

import pytest

from cpp_context_engine.config import AppConfig


def test_environment_configuration_uses_project_local_index(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("CPP_CONTEXT_PROJECT_ROOT", raising=False)
    monkeypatch.delenv("CPP_CONTEXT_INDEX_DIRECTORY", raising=False)

    config = AppConfig.from_environment(cwd=tmp_path)

    assert config.project_root == tmp_path
    assert config.index_directory == tmp_path / ".cpp-context"
    assert config.database_path == tmp_path / ".cpp-context" / "index.db"
    assert config.compilation_database == tmp_path / "build" / "compile_commands.json"
    assert config.embedding_provider == "local"


def test_environment_configuration_rejects_non_positive_limit(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CPP_CONTEXT_RETRIEVAL_LIMIT", "0")

    with pytest.raises(ValueError, match="greater than zero"):
        AppConfig.from_environment(cwd=tmp_path)


def test_environment_loads_provider_configuration_without_exposing_secrets(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("CPP_CONTEXT_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("CPP_CONTEXT_EMBEDDING_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("CPP_CONTEXT_EMBEDDING_MODEL", "embed-code")
    monkeypatch.setenv("CPP_CONTEXT_EMBEDDING_API_KEY", "embedding-secret")
    monkeypatch.setenv("CPP_CONTEXT_LLM_API_KEY", "llm-secret")

    config = AppConfig.from_environment(cwd=tmp_path)

    assert config.embedding_api_key == "embedding-secret"
    assert "embedding-secret" not in repr(config)
    assert "llm-secret" not in repr(config)
