"""Structured, path-free telemetry for the bounded analyzer scheduler."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

AnalyzerEventKind = Literal[
    "scheduling_state",
    "analyzer_started",
    "analyzer_finished",
]
AnalyzerOutcome = Literal["succeeded", "failed", "cancelled"]

_PROTOCOL_EVENT = "analyzer_pipeline"
_PROTOCOL_FIELDS = frozenset(
    {
        "event",
        "sequence",
        "monotonic_seconds",
        "kind",
        "slot_id",
        "configuration_index",
        "outcome",
        "unscheduled_count",
        "held_registries",
        "max_spool_registries",
        "slot_count",
    }
)


@dataclass(frozen=True, slots=True)
class AnalyzerPipelineEvent:
    """One scheduler transition without source text or host filesystem paths."""

    sequence: int
    monotonic_seconds: float
    kind: AnalyzerEventKind
    slot_id: int | None
    configuration_index: int | None
    outcome: AnalyzerOutcome | None
    unscheduled_count: int
    held_registries: int
    max_spool_registries: int
    slot_count: int

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence < 1:
            raise ValueError("analyzer telemetry sequence must be a positive integer")
        if (
            isinstance(self.monotonic_seconds, bool)
            or not isinstance(self.monotonic_seconds, (int, float))
            or not math.isfinite(self.monotonic_seconds)
            or self.monotonic_seconds < 0
        ):
            raise ValueError("analyzer telemetry time must be finite and non-negative")
        object.__setattr__(self, "monotonic_seconds", float(self.monotonic_seconds))
        if self.kind not in {
            "scheduling_state",
            "analyzer_started",
            "analyzer_finished",
        }:
            raise ValueError("analyzer telemetry kind is invalid")
        for name in (
            "unscheduled_count",
            "held_registries",
            "max_spool_registries",
            "slot_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"analyzer telemetry {name} must be a non-negative integer")
        if self.max_spool_registries < 1 or self.slot_count < 1:
            raise ValueError("analyzer telemetry limits must be positive")
        if self.held_registries > self.max_spool_registries:
            raise ValueError("analyzer telemetry held registries exceed the spool limit")
        if self.kind == "scheduling_state":
            if any(
                value is not None
                for value in (self.slot_id, self.configuration_index, self.outcome)
            ):
                raise ValueError("scheduling state telemetry cannot name a slot or outcome")
            return
        if (
            type(self.slot_id) is not int
            or not 0 <= self.slot_id < self.slot_count
            or type(self.configuration_index) is not int
            or self.configuration_index < 0
        ):
            raise ValueError("analyzer lifecycle telemetry has an invalid slot or input index")
        if self.kind == "analyzer_started" and self.outcome is not None:
            raise ValueError("analyzer start telemetry cannot have an outcome")
        if self.kind == "analyzer_finished" and self.outcome not in {
            "succeeded",
            "failed",
            "cancelled",
        }:
            raise ValueError("analyzer finish telemetry must have a valid outcome")

    def to_protocol_payload(self) -> dict[str, Any]:
        """Return the exact JSON child-process event contract."""

        return {"event": _PROTOCOL_EVENT, **asdict(self)}

    @classmethod
    def from_protocol_payload(cls, payload: Mapping[str, Any]) -> AnalyzerPipelineEvent:
        """Validate and decode one exact child-process event."""

        unexpected = set(payload) - _PROTOCOL_FIELDS
        missing = _PROTOCOL_FIELDS - set(payload)
        if unexpected:
            raise ValueError(
                "analyzer telemetry payload has unexpected fields: " + ", ".join(sorted(unexpected))
            )
        if missing:
            raise ValueError(
                "analyzer telemetry payload is missing fields: " + ", ".join(sorted(missing))
            )
        if payload.get("event") != _PROTOCOL_EVENT:
            raise ValueError("analyzer telemetry payload has an invalid event name")
        return cls(
            sequence=payload["sequence"],
            monotonic_seconds=payload["monotonic_seconds"],
            kind=payload["kind"],
            slot_id=payload["slot_id"],
            configuration_index=payload["configuration_index"],
            outcome=payload["outcome"],
            unscheduled_count=payload["unscheduled_count"],
            held_registries=payload["held_registries"],
            max_spool_registries=payload["max_spool_registries"],
            slot_count=payload["slot_count"],
        )


AnalyzerPipelineObserver = Callable[[AnalyzerPipelineEvent], None]


class AnalyzerTelemetryError(RuntimeError):
    """Raised when enabled telemetry cannot be delivered safely."""


class AnalyzerSlotIdleError(RuntimeError):
    """Raised when an analyzer slot violates the bounded-idle gate."""


@dataclass(slots=True)
class _SlotState:
    active: bool = False
    idle_since: float | None = None
    maximum_idle_seconds: float = 0.0
    maximum_idle_cause: str | None = None
    configuration_index: int | None = None
    outcome: AnalyzerOutcome | None = None


class AnalyzerSlotIdleGate:
    """Track exact per-slot idle time from validated analyzer pipeline events."""

    _IDLE_CAUSE = "unscheduled_with_spool_capacity"

    def __init__(
        self,
        *,
        max_idle_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not math.isfinite(max_idle_seconds) or max_idle_seconds <= 0:
            raise ValueError("maximum analyzer idle time must be positive and finite")
        self._max_idle_seconds = max_idle_seconds
        self._clock = clock
        self._slots: list[_SlotState] | None = None
        self._last_sequence = 0
        self._last_event_time = 0.0
        self._unscheduled_count = 0
        self._held_registries = 0
        self._max_spool_registries = 1

    def observe(self, event: AnalyzerPipelineEvent) -> None:
        """Apply one ordered event and fail immediately on an expired idle interval."""

        if event.sequence != self._last_sequence + 1:
            raise ValueError("analyzer telemetry sequence is not contiguous")
        if event.monotonic_seconds < self._last_event_time:
            raise ValueError("analyzer telemetry time moved backwards")
        if self._slots is None:
            self._slots = [_SlotState() for _index in range(event.slot_count)]
        elif len(self._slots) != event.slot_count:
            raise ValueError("analyzer telemetry slot count changed")

        self._measure_idle(event.monotonic_seconds)
        self._last_sequence = event.sequence
        self._last_event_time = event.monotonic_seconds
        self._unscheduled_count = event.unscheduled_count
        self._held_registries = event.held_registries
        self._max_spool_registries = event.max_spool_registries

        if event.kind == "analyzer_started":
            assert event.slot_id is not None and event.configuration_index is not None
            slot = self._slots[event.slot_id]
            if slot.active:
                raise ValueError(f"analyzer slot {event.slot_id} started twice")
            slot.active = True
            slot.idle_since = None
            slot.configuration_index = event.configuration_index
            slot.outcome = None
        elif event.kind == "analyzer_finished":
            assert event.slot_id is not None and event.configuration_index is not None
            slot = self._slots[event.slot_id]
            if not slot.active or slot.configuration_index != event.configuration_index:
                raise ValueError(f"analyzer slot {event.slot_id} finished unexpected work")
            slot.active = False
            slot.configuration_index = event.configuration_index
            slot.outcome = event.outcome

        self._apply_current_idle_state(event.monotonic_seconds)

    def check(self) -> None:
        """Check still-open idle intervals against the monotonic clock."""

        self._measure_idle(self._clock())

    def report(self) -> list[dict[str, Any]]:
        """Return deterministic slot order with measured duration and current cause/state."""

        self._measure_idle(self._clock(), enforce=False)
        if self._slots is None:
            return []
        eligible = self._idle_is_actionable()
        return [
            {
                "slot_id": slot_id,
                "state": "active" if slot.active else "idle",
                "configuration_index": slot.configuration_index,
                "outcome": slot.outcome,
                "maximum_idle_seconds": slot.maximum_idle_seconds,
                "maximum_idle_cause": slot.maximum_idle_cause,
                "idle_cause": self._IDLE_CAUSE if eligible and not slot.active else None,
            }
            for slot_id, slot in enumerate(self._slots)
        ]

    def _idle_is_actionable(self) -> bool:
        return self._unscheduled_count > 0 and self._held_registries < self._max_spool_registries

    def _apply_current_idle_state(self, now: float) -> None:
        assert self._slots is not None
        actionable = self._idle_is_actionable()
        for slot in self._slots:
            if slot.active or not actionable:
                slot.idle_since = None
            elif slot.idle_since is None:
                slot.idle_since = now

    def _measure_idle(self, now: float, *, enforce: bool = True) -> None:
        if self._slots is None:
            return
        if now < self._last_event_time:
            raise ValueError("analyzer idle clock moved backwards")
        for slot_id, slot in enumerate(self._slots):
            if slot.idle_since is None:
                continue
            duration = now - slot.idle_since
            if duration > slot.maximum_idle_seconds:
                slot.maximum_idle_seconds = duration
                slot.maximum_idle_cause = self._IDLE_CAUSE
            if enforce and duration > self._max_idle_seconds:
                raise AnalyzerSlotIdleError(
                    f"analyzer slot {slot_id} was idle for {duration:.3f} seconds "
                    "while work and spool capacity remained"
                )
