from __future__ import annotations

import gc
import threading
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest

from cpp_context_engine import kicad_canary
from cpp_context_engine.ingestion.native import NativeClangIngestor, _FactBatchBuilder
from cpp_context_engine.models import BuildConfiguration


class Payload:
    pass


def observed(iterator, total=2):
    delegate = SimpleNamespace(
        analysis_backend="test",
        advanced_facts_complete=False,
        iter_configuration_batches=lambda *_args: iterator,
    )
    return kicad_canary._ObservedIngestor(delegate, total).iter_configuration_batches(Path(), ())


def test_observer_releases_payload_before_advancing_delegate(monkeypatch):
    events = []
    monkeypatch.setattr(
        kicad_canary, "_worker_event", lambda event, **data: events.append((event, data))
    )
    references = []

    def source():
        for _ in range(3):
            if references:
                gc.collect()
                assert references[-1]() is None, "observer retained the consumed payload"
            payload = Payload()
            references.append(weakref.ref(payload))
            yield payload
            del payload

    stream = observed(source(), total=3)
    try:
        for _ in range(3):
            payload = next(stream)
            assert references[-1]() is payload
            del payload
        assert next(stream, None) is None
    finally:
        stream.close()
    assert events == [
        item
        for index in range(3)
        for item in (
            ("tu_staging", {"configuration_index": index}),
            ("tu_staged", {"completed": index + 1, "total": 3}),
        )
    ]


@pytest.mark.parametrize("failure", ["close", "consumer", "tu_staging", "tu_staged", "producer"])
def test_observer_closes_externally_retained_delegate(monkeypatch, failure):
    closed = []
    events = []

    def source():
        try:
            yield Payload()
            if failure == "producer":
                raise RuntimeError("producer")
            yield Payload()
        finally:
            closed.append(True)

    def event(name, **_data):
        events.append(name)
        if name == failure:
            raise RuntimeError(failure)

    monkeypatch.setattr(kicad_canary, "_worker_event", event)
    retained_delegate = source()
    stream = observed(retained_delegate)
    try:
        if failure == "tu_staging":
            with pytest.raises(RuntimeError, match=failure):
                next(stream)
        else:
            next(stream)
            if failure == "close":
                stream.close()
            elif failure == "consumer":
                with pytest.raises(RuntimeError, match=failure):
                    stream.throw(RuntimeError(failure))
            else:
                with pytest.raises(RuntimeError, match=failure):
                    next(stream)
        assert closed == [True]
        if failure in {"close", "consumer", "tu_staging"}:
            assert events == ["tu_staging"]
    finally:
        stream.close()
        retained_delegate.close()


@pytest.mark.parametrize("with_observer", [False, True])
def test_native_window_releases_consumed_payload_before_next_conversion(
    tmp_path, monkeypatch, with_observer
):
    all_analyzed = threading.Event()
    lock = threading.Lock()
    analyzed = 0
    references = []

    class Client:
        def probe(self):
            return object()

        def analyze(self, *_args):
            nonlocal analyzed
            with lock:
                analyzed += 1
                if analyzed == 3:
                    all_analyzed.set()
            return []

    def build(_builder, _facts):
        assert all_analyzed.wait(timeout=3)
        if references:
            gc.collect()
            assert references[-1]() is None, "native pipeline retained the consumed payload"
        payload = Payload()
        references.append(weakref.ref(payload))
        return payload

    monkeypatch.setattr(_FactBatchBuilder, "build", build)
    configurations = tuple(
        BuildConfiguration(
            id=f"build-{index}",
            source_path=tmp_path / f"{index}.cpp",
            directory=tmp_path,
            arguments=("clang++",),
            command_hash=f"hash-{index}",
        )
        for index in range(3)
    )
    ingestor = NativeClangIngestor(
        Client(), max_workers=3, max_spool_registries=3, max_domain_batches=1
    )
    if with_observer:
        ingestor = kicad_canary._ObservedIngestor(ingestor, len(configurations))
    stream = ingestor.iter_configuration_batches(tmp_path, configurations)
    try:
        for _ in range(3):
            payload = next(stream)
            assert references[-1]() is payload
            del payload
        assert next(stream, None) is None
    finally:
        stream.close()


@pytest.mark.parametrize("failure", ["tu_staging", "tu_staged"])
def test_observer_retained_event_traceback_does_not_retain_payload(monkeypatch, failure):
    references = []
    closed = []

    def source():
        try:
            payload = Payload()
            references.append(weakref.ref(payload))
            yield payload
        finally:
            closed.append(True)

    def event(name, **_data):
        if name == failure:
            raise RuntimeError(failure)

    monkeypatch.setattr(kicad_canary, "_worker_event", event)
    retained_delegate = source()
    stream = observed(retained_delegate)
    if failure == "tu_staged":
        next(stream)
    with pytest.raises(RuntimeError, match=failure) as retained_exception:
        next(stream)
    gc.collect()
    assert retained_exception.value.__traceback__ is not None
    assert closed == [True]
    assert references[0]() is None


@pytest.mark.parametrize("early_close", [False, True])
def test_observer_propagates_delegate_close_error(monkeypatch, early_close):
    events = []
    monkeypatch.setattr(kicad_canary, "_worker_event", lambda name, **_data: events.append(name))

    class Source:
        def __init__(self):
            self.remaining = 1
            self.close_calls = 0

        def __iter__(self):
            return self

        def __next__(self):
            if not self.remaining:
                raise StopIteration
            self.remaining -= 1
            return Payload()

        def close(self):
            self.close_calls += 1
            raise RuntimeError("delegate close failed")

    delegate = Source()
    stream = observed(delegate, total=1)
    next(stream)
    with pytest.raises(RuntimeError, match="delegate close failed"):
        if early_close:
            stream.close()
        else:
            next(stream)
    assert delegate.close_calls == 1
    assert events == (["tu_staging"] if early_close else ["tu_staging", "tu_staged"])
