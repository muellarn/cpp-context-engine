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


def test_environment_loads_native_analyzer_and_limits(tmp_path, monkeypatch) -> None:
    binary = tmp_path / "analyzer"
    monkeypatch.setenv("CPP_CONTEXT_CLANG_ANALYZER", str(binary))
    monkeypatch.setenv("CPP_CONTEXT_ANALYZER_TIMEOUT", "12.5")
    monkeypatch.setenv("CPP_CONTEXT_ANALYZER_MAX_OUTPUT_BYTES", "4096")
    monkeypatch.setenv("CPP_CONTEXT_ANALYZER_MAX_DECODED_BYTES", "8192")
    monkeypatch.setenv("CPP_CONTEXT_ANALYZER_MAX_RECORD_BYTES", "2048")
    monkeypatch.setenv("CPP_CONTEXT_ANALYZER_MAX_WORKERS", "3")
    monkeypatch.setenv("CPP_CONTEXT_ANALYZER_MAX_SPOOL_REGISTRIES", "7")
    monkeypatch.setenv("CPP_CONTEXT_ANALYZER_MAX_SPOOL_BYTES", "65536")
    monkeypatch.setenv("CPP_CONTEXT_ANALYZER_MAX_SPOOL_FILES", "128")
    monkeypatch.setenv("CPP_CONTEXT_ANALYZER_MAX_DOMAIN_BATCHES", "2")

    config = AppConfig.from_environment(cwd=tmp_path)

    assert config.clang_analyzer_path == binary
    assert config.analyzer_timeout_seconds == 12.5
    assert config.analyzer_max_output_bytes == 4096
    assert config.analyzer_max_decoded_bytes == 8192
    assert config.analyzer_max_record_bytes == 2048
    assert config.analyzer_max_workers == 3
    assert config.analyzer_max_spool_registries == 7
    assert config.analyzer_max_spool_bytes == 65536
    assert config.analyzer_max_spool_files == 128
    assert config.analyzer_max_domain_batches == 2


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
