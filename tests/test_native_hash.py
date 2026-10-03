from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest

import cpp_context_engine.ingestion.native as native
from cpp_context_engine.models import BuildConfiguration, IndexProfile


def _streaming_hash(*values: str) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
    return digest.hexdigest()


def test_composite_hash_uses_one_bounded_digest_input(monkeypatch: pytest.MonkeyPatch) -> None:
    values = ("default", "tu_123", "sym_a", "sym_b", "calls", "source.cpp", "7", "2", "7", "8")
    expected = _streaming_hash(*values)
    sha256 = hashlib.sha256
    initial_inputs = []
    updates = []

    class CountedHash:
        def __init__(self, data=b""):
            initial_inputs.append(data)
            self.digest = sha256(data)

        def update(self, data):
            updates.append(data)
            self.digest.update(data)

        def hexdigest(self):
            return self.digest.hexdigest()

    monkeypatch.setattr(native.hashlib, "sha256", CountedHash)
    assert native._hash_text(*values) == expected
    assert len(updates) == 0
    assert initial_inputs == [("\0".join(values) + "\0").encode("utf-8")]


@pytest.mark.parametrize(
    "values",
    [
        (),
        ("",),
        ("", ""),
        ("a", ""),
        ("a\0b", ""),
        ("a", "b\0"),
        ("α🙂", "\udc80\udcff"),
        ("x" * 4094, ""),
        ("x" * 4095, ""),
        ("🙂" * 4094, ""),
        ("source" * 10000, "tail"),
        ("",) * 4097,
    ],
)
def test_native_hash_preserves_exact_streaming_framing(values: tuple[str, ...]) -> None:
    assert native._hash_text(*values) == _streaming_hash(*values)


@pytest.mark.parametrize("values", [("prefix", "a\ud800b"), ("\udfff", "suffix")])
def test_native_hash_preserves_field_local_encoding_error(values: tuple[str, ...]) -> None:
    with pytest.raises(UnicodeEncodeError) as old:
        _streaming_hash(*values)
    with pytest.raises(UnicodeEncodeError) as new:
        native._hash_text(*values)
    assert new.value.args == old.value.args


def test_native_hash_does_not_join_large_or_single_values(monkeypatch: pytest.MonkeyPatch) -> None:
    sha256 = hashlib.sha256
    initial_inputs = []

    def observe(data=b""):
        initial_inputs.append(data)
        return sha256(data)

    monkeypatch.setattr(native.hashlib, "sha256", observe)
    for values in (("x" * 4095, ""), ("single",), (), ("",) * 4097):
        native._hash_text(*values)
    assert initial_inputs == [b""] * 4


@pytest.mark.parametrize("profile", [IndexProfile.FULL, IndexProfile.NAVIGATION])
def test_hash_change_preserves_ordered_complete_builder_models(
    tmp_path: Path, profile: IndexProfile, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.cpp"
    source.write_text("int α() { return 1; }\n", encoding="utf-8")
    configuration = BuildConfiguration(
        "build", source, tmp_path, ("clang++", str(source)), "command"
    )
    span = dict(path=str(source), start_line=1, end_line=1, start_column=1, end_column=22)
    facts = [
        dict(fact="file", key="file-key", path=str(source)),
        dict(
            fact="symbol",
            key="usr:α",
            qualified_name="α",
            kind="function",
            span=span,
            source_text="int α() { return 1; }",
            signature="int α()",
        ),
        dict(fact="occurrence", symbol_key="usr:α", span=span, kind="definition"),
        dict(
            fact="edge", source_key="file-key", target_key="usr:α", relation="contains", span=span
        ),
        dict(
            fact="callsite_v1",
            key="site",
            owner_key="usr:α",
            static_target_key="usr:α",
            dispatch_kind="direct",
            spelling_span=span,
            expansion_span=span,
            target_set_complete=True,
            callee_text="α()",
        ),
        dict(
            fact="call_target_v1",
            callsite_key="site",
            target_key="usr:α",
            certainty="certain",
            confidence=1.0,
            confidence_reason="direct",
            derivation="direct",
            evidence_span=span,
        ),
    ]
    candidate = native._FactBatchBuilder(tmp_path, configuration, profile).build(facts)
    monkeypatch.setattr(native, "_hash_text", _streaming_hash)
    reference = native._FactBatchBuilder(tmp_path, configuration, profile).build(facts)

    # Dataclass equality excludes some provenance fields; compare every field,
    # preserving tuple order and immutable metadata without asdict/deepcopy.
    def complete(value):
        if is_dataclass(value):
            return tuple(
                (field.name, complete(getattr(value, field.name))) for field in fields(value)
            )
        if isinstance(value, Mapping):
            return dict(value)
        if isinstance(value, tuple):
            return tuple(complete(item) for item in value)
        return value

    assert complete(candidate) == complete(reference)
    assert len(candidate.symbols) == 2
    assert len(candidate.occurrences) == len(candidate.edges) == len(candidate.callsites) == 1
    assert len(candidate.call_targets) == 1
