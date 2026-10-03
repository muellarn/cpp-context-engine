from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest
from test_native_analyzer import _fake_hello, _script

from cpp_context_engine.ingestion.native import (
    AnalyzerLimitError,
    AnalyzerProtocolError,
    NativeAnalyzerClient,
    NativeClangIngestor,
    _FactBatchBuilder,
    _FactRegistry,
    _ResourceBudget,
    _SymbolTextChunks,
)
from cpp_context_engine.models import BuildConfiguration, IndexProfile

CAPABILITY = "symbol_text_chunks_v1"
CHUNK_KIND = "symbol_text_chunk_v1"


def _split_symbol(symbol: dict) -> list[dict]:
    descriptor = dict(symbol)
    descriptions = {}
    chunks = []
    for field in ("signature", "documentation", "source_text"):
        text = descriptor.pop(field)
        descriptions[field] = {
            "bytes": len(text.encode()),
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
        for index, offset in enumerate(range(0, len(text), 64)):
            chunks.append(
                {
                    "type": "fact",
                    "fact": CHUNK_KIND,
                    "key": symbol["key"],
                    "field": field,
                    "index": index,
                    "text": text[offset : offset + 64],
                }
            )
    descriptor["text_chunks"] = descriptions
    return [descriptor, *chunks]


def _fixture(tmp_path: Path) -> tuple[BuildConfiguration, list[dict]]:
    source = tmp_path / "resource.cpp"
    text = "int resource = 0; /*" + 'é😀\\"\t' * 512 + "*/"
    source.write_text(text)
    config = BuildConfiguration(
        id="config",
        source_path=source,
        directory=tmp_path,
        arguments=("clang++", str(source)),
        command_hash="command",
        build_variant="default",
    )
    symbol = {
        "type": "fact",
        "fact": "symbol",
        "key": "usr:resource",
        "qualified_name": "resource",
        "kind": "variable",
        "span": {
            "path": str(source),
            "start_line": 1,
            "start_column": 1,
            "end_line": 1,
            "end_column": len(text) + 1,
        },
        "signature": text,
        "documentation": "complete documentation",
        "source_text": text,
        "metadata": {"is_definition": True},
    }
    return config, [symbol]


def _companion(
    tmp_path: Path,
    facts: list[dict],
    *,
    compressed: bool,
    physical: list[dict] | None = None,
    supports_chunks: bool = True,
) -> Path:
    hello = _fake_hello(gzip_transport=compressed)
    if supports_chunks:
        hello["capabilities"].append(CAPABILITY)
    fragments = (
        [part for fact in facts for part in _split_symbol(fact)] if physical is None else physical
    )
    return _script(
        tmp_path,
        f"""import json, sys, zlib
requests = [json.loads(line) for line in sys.stdin]
hello = {hello!r}
records = [hello]
if len(requests) > 1:
    request = requests[1]
    negotiated = {CAPABILITY!r} in requests[0]["required_capabilities"]
    facts = {fragments!r} if negotiated else {facts!r}
    records += [{{"type": "begin", "request_id": request["request_id"]}}, *facts,
                {{"type": "complete", "request_id": request["request_id"], "success": True}}]
raw = b"".join(json.dumps(record, separators=(",", ":")).encode() + b"\\n" for record in records)
if requests[0].get("response_transport") == "gzip_jsonl_v1":
    encoder = zlib.compressobj(wbits=31)
    raw = encoder.compress(raw) + encoder.flush()
sys.stdout.buffer.write(raw)
""",
    )


def _complete_equal(left: object, right: object) -> None:
    assert type(left) is type(right)
    if is_dataclass(left):
        for field in fields(left):
            _complete_equal(getattr(left, field.name), getattr(right, field.name))
    elif isinstance(left, Mapping):
        assert list(left) == list(right)
        for key in left:
            _complete_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _complete_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("compressed", [False, True])
def test_large_symbol_text_reaches_real_registry_and_builder_losslessly(
    tmp_path: Path,
    compressed: bool,
) -> None:
    configuration, facts = _fixture(tmp_path)
    assert len(json.dumps(facts[0]).encode()) > 2048
    client = NativeAnalyzerClient(
        _companion(tmp_path, facts, compressed=compressed),
        max_record_bytes=2048,
        profile=IndexProfile.NAVIGATION,
    )
    actual = next(
        NativeClangIngestor(client, profile=IndexProfile.NAVIGATION).iter_configuration_batches(
            tmp_path, (configuration,)
        )
    )
    expected = _FactBatchBuilder(
        tmp_path,
        configuration,
        IndexProfile.NAVIGATION,
        analyzer_identity=client.analyzer_identity,
    ).build(facts)
    _complete_equal(actual, expected)
    assert client.analyze(tmp_path, configuration) == tuple(facts)


@pytest.mark.parametrize(
    "invalid",
    [
        "missing",
        "duplicate",
        "order",
        "owner",
        "field",
        "digest",
        "length",
        "unused",
        "utf8",
        "descriptor",
        "unnegotiated",
    ],
)
def test_symbol_text_fragments_fail_closed(tmp_path: Path, invalid: str) -> None:
    configuration, facts = _fixture(tmp_path)
    physical = _split_symbol(facts[0])
    if invalid == "missing":
        physical.pop()
    elif invalid == "duplicate":
        physical.insert(2, deepcopy(physical[1]))
    elif invalid == "order":
        physical[1], physical[2] = physical[2], physical[1]
    elif invalid in {"owner", "field"}:
        physical[1]["key" if invalid == "owner" else "field"] = "wrong"
    elif invalid == "digest":
        physical[1]["text"] = "z" + physical[1]["text"][1:]
    elif invalid == "length":
        physical[0]["text_chunks"]["signature"]["bytes"] = 1
    elif invalid == "unused":
        physical.append(deepcopy(physical[-1]))
    elif invalid == "utf8":
        physical[1]["text"] = "\ud800"
    elif invalid == "descriptor":
        physical[0]["text_chunks"]["other"] = {"bytes": 1, "sha256": "0" * 64}
    if invalid != "unnegotiated":
        with _FactRegistry(max_record_bytes=2048) as registry:
            for record in physical:
                registry.add(record)
            with pytest.raises(AnalyzerProtocolError, match="symbol text"):
                _FactBatchBuilder(tmp_path, configuration, IndexProfile.NAVIGATION).build(registry)
    client = NativeAnalyzerClient(
        _companion(
            tmp_path,
            facts if invalid != "unnegotiated" else physical,
            compressed=True,
            physical=physical,
            supports_chunks=invalid != "unnegotiated",
        ),
        max_record_bytes=2048,
    )
    with pytest.raises(AnalyzerProtocolError, match="symbol text"):
        client.analyze(tmp_path, configuration)


@pytest.mark.parametrize("limit", ["decoded", "wire", "record", "spool"])
def test_symbol_fragmentation_keeps_independent_budgets(tmp_path: Path, limit: str) -> None:
    configuration, facts = _fixture(tmp_path)
    physical = _split_symbol(facts[0])
    if limit == "record":
        physical[1]["text"] = "x" * 4096
    kwargs = {"max_record_bytes": 2048}
    if limit in {"decoded", "wire"}:
        kwargs["max_decoded_bytes" if limit == "decoded" else "max_output_bytes"] = 4096
    client = NativeAnalyzerClient(
        _companion(tmp_path, facts, compressed=False, physical=physical),
        **kwargs,
    )
    byte_budget = _ResourceBudget(4096 if limit == "spool" else 1_000_000, "spool byte limit")
    fd_budget = _ResourceBudget(4, "spool file limit")
    with (
        _FactRegistry(
            max_record_bytes=2048, byte_budget=byte_budget, fd_budget=fd_budget
        ) as registry,
        pytest.raises(AnalyzerLimitError, match="limit"),
    ):
        client._analyze_registry(tmp_path, configuration, registry, cancelled=threading.Event())
    assert byte_budget.used == fd_budget.used == 0


def test_new_consumer_accepts_legacy_symbol_records(tmp_path: Path) -> None:
    configuration, facts = _fixture(tmp_path)
    client = NativeAnalyzerClient(
        _companion(tmp_path, facts, compressed=False, supports_chunks=False)
    )
    assert client.analyze(tmp_path, configuration) == tuple(facts)


def test_descriptor_cannot_allocate_from_untrusted_lengths() -> None:
    chunks = _SymbolTextChunks(assemble=False)
    chunks.accept(
        {
            "fact": "symbol",
            "key": "key",
            "text_chunks": {
                "source_text": {"bytes": 10**100, "sha256": "0" * 64},
            },
        }
    )
    with pytest.raises(AnalyzerProtocolError, match="incomplete"):
        chunks.finish()


@pytest.mark.native
@pytest.mark.parametrize("compressed", [False, True])
def test_real_native_utf8_chunks_preserve_legacy_facts_and_models(
    tmp_path: Path, compressed: bool
) -> None:
    from analyzer_discovery import analyzer_binary

    class LegacyClient(NativeAnalyzerClient):
        @staticmethod
        def _hello(*, response_transport=None, symbol_text_chunks=False):
            return NativeAnalyzerClient._hello(response_transport=response_transport)

    source = tmp_path / "resource.cpp"
    prefix = 'const char image[] = "'
    source.write_text(
        "namespace resource { " + prefix + "x" * (65535 - len(prefix)) + "😀" + r"\t\"" + '"; }'
    )
    configuration = BuildConfiguration(
        id="resource",
        source_path=source,
        directory=tmp_path,
        arguments=("clang++", "-std=c++20", str(source)),
        command_hash="command",
    )
    for profile in (IndexProfile.FULL, IndexProfile.NAVIGATION):
        client = NativeAnalyzerClient(
            analyzer_binary(), profile=profile, prefer_compression=compressed
        )
        reference = LegacyClient(client.binary, profile=profile, prefer_compression=compressed)
        facts = reference.analyze(tmp_path, configuration)
        assert CAPABILITY in client.probe().capabilities
        assert client.analyze(tmp_path, configuration) == facts
        with _FactRegistry(max_record_bytes=512 * 1024) as registry:
            client._analyze_registry(tmp_path, configuration, registry, cancelled=threading.Event())
            fragments = list(registry.records(CHUNK_KIND))
            assert fragments and all(len(item["text"].encode()) <= 65536 for item in fragments)
            assert any(item["text"].startswith("😀") for item in fragments)
            actual = _FactBatchBuilder(tmp_path, configuration, profile).build(registry)
        expected = _FactBatchBuilder(tmp_path, configuration, profile).build(facts)
        _complete_equal(actual, expected)
