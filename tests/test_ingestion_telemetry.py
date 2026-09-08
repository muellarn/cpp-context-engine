from __future__ import annotations

import threading
from pathlib import Path

import pytest

from cpp_context_engine.ingestion import (
    AnalyzerLimitError,
    AnalyzerPipelineEvent,
    AnalyzerSlotIdleError,
    AnalyzerSlotIdleGate,
    AnalyzerTelemetryError,
    NativeClangIngestor,
)
from cpp_context_engine.ingestion.native import _TelemetryDispatcher
from cpp_context_engine.models import BuildConfiguration


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _event(
    sequence: int,
    at: float,
    kind: str,
    *,
    slot_id: int | None = None,
    configuration_index: int | None = None,
    outcome: str | None = None,
    unscheduled_count: int = 1,
    held_registries: int = 1,
    max_spool_registries: int = 4,
    slot_count: int = 2,
) -> AnalyzerPipelineEvent:
    return AnalyzerPipelineEvent(
        sequence=sequence,
        monotonic_seconds=at,
        kind=kind,
        slot_id=slot_id,
        configuration_index=configuration_index,
        outcome=outcome,
        unscheduled_count=unscheduled_count,
        held_registries=held_registries,
        max_spool_registries=max_spool_registries,
        slot_count=slot_count,
    )


def test_idle_gate_detects_one_hidden_idle_slot_while_another_progresses() -> None:
    clock = _Clock()
    gate = AnalyzerSlotIdleGate(max_idle_seconds=10.0, clock=clock)
    gate.observe(_event(1, 0.0, "scheduling_state", unscheduled_count=4, held_registries=0))
    gate.observe(_event(2, 0.0, "analyzer_started", slot_id=0, configuration_index=0))
    gate.observe(_event(3, 0.0, "analyzer_started", slot_id=1, configuration_index=1))
    gate.observe(
        _event(
            4,
            1.0,
            "analyzer_finished",
            slot_id=0,
            configuration_index=0,
            outcome="succeeded",
        )
    )
    gate.observe(
        _event(
            5,
            5.0,
            "analyzer_finished",
            slot_id=1,
            configuration_index=1,
            outcome="succeeded",
        )
    )
    gate.observe(
        _event(6, 5.0, "analyzer_started", slot_id=1, configuration_index=2)
    )

    clock.now = 12.0
    with pytest.raises(AnalyzerSlotIdleError, match=r"slot 0.*11\.000"):
        gate.check()
    assert gate.report()[0]["maximum_idle_seconds"] == 11.0


def test_idle_gate_pauses_while_spool_is_full() -> None:
    clock = _Clock()
    gate = AnalyzerSlotIdleGate(max_idle_seconds=10.0, clock=clock)
    gate.observe(
        _event(
            1,
            0.0,
            "scheduling_state",
            unscheduled_count=1,
            held_registries=1,
            max_spool_registries=2,
            slot_count=1,
        )
    )
    clock.now = 5.0
    gate.check()
    gate.observe(
        _event(
            2,
            5.0,
            "scheduling_state",
            unscheduled_count=1,
            held_registries=2,
            max_spool_registries=2,
            slot_count=1,
        )
    )
    clock.now = 20.0
    gate.check()
    gate.observe(
        _event(
            3,
            20.0,
            "scheduling_state",
            unscheduled_count=1,
            held_registries=1,
            max_spool_registries=2,
            slot_count=1,
        )
    )
    clock.now = 29.0
    gate.check()
    clock.now = 31.0
    with pytest.raises(AnalyzerSlotIdleError, match="slot 0"):
        gate.check()


def test_idle_gate_pauses_after_all_work_is_scheduled() -> None:
    clock = _Clock()
    gate = AnalyzerSlotIdleGate(max_idle_seconds=10.0, clock=clock)
    gate.observe(
        _event(
            1,
            0.0,
            "scheduling_state",
            unscheduled_count=1,
            held_registries=1,
            max_spool_registries=2,
            slot_count=1,
        )
    )
    clock.now = 9.0
    gate.check()
    gate.observe(
        _event(
            2,
            9.0,
            "scheduling_state",
            unscheduled_count=0,
            held_registries=1,
            max_spool_registries=2,
            slot_count=1,
        )
    )
    clock.now = 100.0
    gate.check()
    assert gate.report()[0]["maximum_idle_seconds"] == 9.0


def test_idle_gate_report_and_child_protocol_are_deterministic() -> None:
    clock = _Clock()
    events = (
        _event(1, 0.0, "scheduling_state", unscheduled_count=3, held_registries=0),
        _event(2, 1.0, "analyzer_started", slot_id=1, configuration_index=1),
        _event(3, 2.0, "analyzer_started", slot_id=0, configuration_index=0),
        _event(
            4,
            4.0,
            "analyzer_finished",
            slot_id=1,
            configuration_index=1,
            outcome="succeeded",
        ),
    )
    first = AnalyzerSlotIdleGate(max_idle_seconds=10.0, clock=clock)
    second = AnalyzerSlotIdleGate(max_idle_seconds=10.0, clock=clock)
    for event in events:
        payload = event.to_protocol_payload()
        assert payload["event"] == "analyzer_pipeline"
        restored = AnalyzerPipelineEvent.from_protocol_payload(payload)
        assert restored == event
        first.observe(restored)
        second.observe(restored)
    clock.now = 5.0

    assert first.report() == second.report()
    assert [item["slot_id"] for item in first.report()] == [0, 1]
    assert first.report()[0]["maximum_idle_seconds"] == 2.0
    assert first.report()[1]["maximum_idle_seconds"] == 1.0
    assert first.report()[1]["idle_cause"] == "unscheduled_with_spool_capacity"


def _configurations(tmp_path: Path, count: int) -> tuple[BuildConfiguration, ...]:
    configurations = []
    for index in range(count):
        source = tmp_path / f"unit-{index}.cpp"
        source.write_text(f"int value_{index};\n", encoding="utf-8")
        configurations.append(
            BuildConfiguration(
                id=f"build-{index}",
                source_path=source,
                directory=tmp_path,
                arguments=("clang++", str(source)),
                command_hash=f"hash-{index}",
            )
        )
    return tuple(configurations)


class _EmptyClient:
    def probe(self) -> object:
        return object()

    def analyze(
        self, _root: Path, _configuration: BuildConfiguration
    ) -> list[dict[str, object]]:
        return []


def _live_telemetry_threads() -> list[threading.Thread]:
    return [
        thread
        for thread in threading.enumerate()
        if thread.name == "cpp-context-telemetry" and thread.is_alive()
    ]


def test_optional_observer_preserves_results_and_emits_balanced_sanitized_lifecycles(
    tmp_path: Path,
) -> None:
    configurations = _configurations(tmp_path, 5)
    baseline = list(
        NativeClangIngestor(  # type: ignore[arg-type]
            _EmptyClient(), max_workers=2, max_spool_registries=3
        ).iter_configuration_batches(tmp_path, configurations)
    )
    events: list[AnalyzerPipelineEvent] = []
    callback_threads: list[str] = []

    def observe(event: AnalyzerPipelineEvent) -> None:
        events.append(event)
        callback_threads.append(threading.current_thread().name)

    observed = list(
        NativeClangIngestor(  # type: ignore[arg-type]
            _EmptyClient(),
            max_workers=2,
            max_spool_registries=3,
            observer=observe,
        ).iter_configuration_batches(tmp_path, configurations)
    )

    assert observed == baseline
    assert [event.sequence for event in events] == list(range(1, len(events) + 1))
    assert set(callback_threads) == {"cpp-context-telemetry"}
    assert str(tmp_path) not in "".join(str(event.to_protocol_payload()) for event in events)
    starts = {
        event.configuration_index: event
        for event in events
        if event.kind == "analyzer_started"
    }
    finishes = {
        event.configuration_index: event for event in events if event.kind == "analyzer_finished"
    }
    assert starts.keys() == finishes.keys() == set(range(5))
    assert all(starts[index].slot_id == finishes[index].slot_id for index in starts)
    assert all(event.outcome == "succeeded" for event in finishes.values())
    assert not _live_telemetry_threads()


def test_worker_failure_closes_every_started_slot(tmp_path: Path) -> None:
    configurations = _configurations(tmp_path, 2)
    release = threading.Event()

    class FailingClient:
        def probe(self) -> object:
            return object()

        def analyze(
            self, _root: Path, configuration: BuildConfiguration
        ) -> list[dict[str, object]]:
            if configuration.id == "build-0":
                release.set()
                raise RuntimeError("analysis failed")
            assert release.wait(timeout=2)
            return []

    events: list[AnalyzerPipelineEvent] = []
    batches = NativeClangIngestor(  # type: ignore[arg-type]
        FailingClient(), max_workers=2, observer=events.append
    ).iter_configuration_batches(tmp_path, configurations)

    with pytest.raises(RuntimeError, match="analysis failed"):
        list(batches)

    starts = [event for event in events if event.kind == "analyzer_started"]
    finishes = [event for event in events if event.kind == "analyzer_finished"]
    assert {(event.slot_id, event.configuration_index) for event in starts} == {
        (event.slot_id, event.configuration_index) for event in finishes
    }
    assert any(event.outcome == "failed" for event in finishes)
    assert all(event.outcome in {"failed", "cancelled"} for event in finishes)
    assert not _live_telemetry_threads()


def test_closing_observed_pipeline_marks_active_slots_cancelled_and_joins_dispatcher(
    tmp_path: Path,
) -> None:
    configurations = _configurations(tmp_path, 3)
    second_started = threading.Event()

    class CancellableClient:
        def probe(self) -> object:
            return object()

        def analyze_stream(
            self,
            _root: Path,
            configuration: BuildConfiguration,
            _on_fact: object,
            *,
            cancelled: threading.Event,
        ) -> None:
            if configuration.id == "build-0":
                assert second_started.wait(timeout=2)
                return
            second_started.set()
            assert cancelled.wait(timeout=2)
            raise AnalyzerLimitError("cancelled for test")

    events: list[AnalyzerPipelineEvent] = []
    batches = NativeClangIngestor(  # type: ignore[arg-type]
        CancellableClient(), max_workers=2, observer=events.append
    ).iter_configuration_batches(tmp_path, configurations)
    next(batches)
    batches.close()

    starts = [event for event in events if event.kind == "analyzer_started"]
    finishes = [event for event in events if event.kind == "analyzer_finished"]
    assert {(event.slot_id, event.configuration_index) for event in starts} == {
        (event.slot_id, event.configuration_index) for event in finishes
    }
    assert any(event.outcome == "cancelled" for event in finishes)
    assert not _live_telemetry_threads()


def test_observer_exception_is_fatal_and_dispatcher_is_joined(tmp_path: Path) -> None:
    configurations = _configurations(tmp_path, 2)

    def fail_observer(_event: AnalyzerPipelineEvent) -> None:
        raise ValueError("observer failed for test")

    with pytest.raises(AnalyzerTelemetryError, match="observer failed"):
        list(
            NativeClangIngestor(  # type: ignore[arg-type]
                _EmptyClient(), max_workers=2, observer=fail_observer
            ).iter_configuration_batches(tmp_path, configurations)
        )

    assert not _live_telemetry_threads()


def test_observer_queue_overflow_is_fatal_without_blocking_scheduler_calls() -> None:
    observer_entered = threading.Event()
    release_observer = threading.Event()

    def block_observer(_event: AnalyzerPipelineEvent) -> None:
        observer_entered.set()
        assert release_observer.wait(timeout=2)

    dispatcher = _TelemetryDispatcher(
        block_observer,
        slot_count=1,
        total_configurations=100,
        max_spool_registries=2,
    )
    assert observer_entered.wait(timeout=1)
    # These are the exact non-blocking calls made while the scheduler condition is held.
    for remaining in range(99, 59, -1):
        dispatcher.update_state(unscheduled_count=remaining, held_registries=0)
    assert isinstance(dispatcher.failure, AnalyzerTelemetryError)
    assert "bounded event queue" in str(dispatcher.failure)

    release_observer.set()
    result = dispatcher.close()

    assert result is dispatcher.failure
    assert not _live_telemetry_threads()


def test_pipeline_event_rejects_path_or_content_extension_fields() -> None:
    event = _event(1, 0.0, "scheduling_state")
    payload = event.to_protocol_payload()

    with pytest.raises(ValueError, match="unexpected"):
        AnalyzerPipelineEvent.from_protocol_payload({**payload, "source_path": "/secret/a.cpp"})
    with pytest.raises(ValueError, match="unexpected"):
        AnalyzerPipelineEvent.from_protocol_payload({**payload, "source": "int secret;"})
