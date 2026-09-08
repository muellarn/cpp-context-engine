"""Explicit, bounded materialization of TU-scoped deep compiler facts."""

from __future__ import annotations

import hashlib
import threading
import time
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import nullcontext
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from cpp_context_engine.config import AppConfig
from cpp_context_engine.ingestion.compilation_database import CompilationDatabase
from cpp_context_engine.ingestion.native import (
    PROTOCOL,
    PROTOCOL_VERSION,
    NativeAnalyzerClient,
    NativeClangIngestor,
    _ResourceBudget,
)
from cpp_context_engine.models import BuildConfiguration, BuildScope, IndexProfile
from cpp_context_engine.storage.sqlite import SCHEMA_VERSION, DeepTranslationUnitTarget, SQLiteStore

DEFAULT_MAX_TUS = 4
MAX_MAX_TUS = 32
DEFAULT_MAX_WALL_SECONDS = 120
MAX_MAX_WALL_SECONDS = 300
DEFAULT_MAX_DECODED_BYTES = 512 * 1024 * 1024
MAX_MAX_DECODED_BYTES = 1024 * 1024 * 1024
DEFAULT_MAX_SPOOL_BYTES = 512 * 1024 * 1024
MAX_MAX_SPOOL_BYTES = 1024 * 1024 * 1024
DEFAULT_MAX_SPOOL_FILES = 128
MAX_MAX_SPOOL_FILES = 512


class DeepContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CancellationSignal(Protocol):
    def is_set(self) -> bool: ...


class DeepCancellation:
    """Cancellation token whose publication guard makes the final commit race-free."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._publication_lock = threading.Lock()

    def set(self) -> None:
        with self._publication_lock:
            self._event.set()

    def is_set(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)

    def publication_guard(self):
        return self._publication_lock


class DeepRequestControl:
    """One monotonic wall deadline and cancellation signal for the whole request."""

    def __init__(self, started: float, wall_seconds: float, cancelled: CancellationSignal) -> None:
        self.started = started
        self.deadline = started + wall_seconds
        self.cancelled = cancelled

    def check(self, stage: str = "operation") -> None:
        if self.cancelled.is_set():
            raise RuntimeError("deep materialization was cancelled")
        if time.monotonic() >= self.deadline:
            raise TimeoutError(f"deep materialization exceeded max_wall_seconds during {stage}")

    def remaining(self, stage: str = "operation") -> float:
        self.check(stage)
        return max(0.001, self.deadline - time.monotonic())

    def publication_guard(self):
        guard = getattr(self.cancelled, "publication_guard", None)
        return guard() if guard is not None else nullcontext()


class MaterializeDeepRequest(DeepContract):
    symbol_id: str = Field(min_length=1, max_length=2_048)
    builds: list[str] | None = Field(default=None, min_length=1, max_length=16)
    max_tus: int = Field(default=DEFAULT_MAX_TUS, ge=1, le=MAX_MAX_TUS)
    max_wall_seconds: int = Field(default=DEFAULT_MAX_WALL_SECONDS, ge=1, le=MAX_MAX_WALL_SECONDS)
    max_decoded_bytes: int = Field(
        default=DEFAULT_MAX_DECODED_BYTES, ge=1, le=MAX_MAX_DECODED_BYTES
    )
    max_spool_bytes: int = Field(default=DEFAULT_MAX_SPOOL_BYTES, ge=1, le=MAX_MAX_SPOOL_BYTES)
    max_spool_files: int = Field(default=DEFAULT_MAX_SPOOL_FILES, ge=1, le=MAX_MAX_SPOOL_FILES)


class MaterializedUnit(DeepContract):
    build_variant: str
    translation_unit_id: str
    build_configuration_id: str
    identity_hash: str
    distance: int = Field(ge=0)
    cache_hit: bool
    control_flow: bool
    data_flow: bool
    summaries: bool
    bindings: bool


class DeepProvenance(DeepContract):
    analyzer_identity: str
    analyzer_version: str | None
    protocol: str
    protocol_version: int
    fact_schema_version: int
    profile: IndexProfile
    build_scope: list[str]
    closure_generation_id: str


class MaterializeDeepResult(DeepContract):
    status: str
    materialization_id: str
    root_symbol_id: str
    units: list[MaterializedUnit]
    closure_complete: bool
    known_tus: int = Field(ge=0)
    omitted_tus: int = Field(ge=0)
    limit_reason: str = ""
    cache_hit: bool
    elapsed_seconds: float = Field(ge=0.0)
    provenance: DeepProvenance


class DeepMaterializer:
    """Resolve, analyze and atomically publish one bounded deep overlay."""

    def __init__(self, config: AppConfig, store: SQLiteStore) -> None:
        self.config = config
        self.store = store
        self._lock = threading.Lock()
        self._admission = threading.Lock()
        self._inflight: dict[tuple[object, ...], Future[MaterializeDeepResult]] = {}

    def materialize(
        self, request: MaterializeDeepRequest, cancelled: CancellationSignal | None = None
    ) -> MaterializeDeepResult:
        control = DeepRequestControl(
            time.monotonic(), request.max_wall_seconds, cancelled or DeepCancellation()
        )
        # Only omission inherits the operator scope; an explicit empty scope
        # must never broaden into a multi-build request.
        requested_builds = (
            self.config.build_scope.variants if request.builds is None else tuple(request.builds)
        )
        key = (
            request.symbol_id,
            requested_builds,
            request.max_tus,
            request.max_wall_seconds,
            request.max_decoded_bytes,
            request.max_spool_bytes,
            request.max_spool_files,
        )
        leader = False
        with self._lock:
            future = self._inflight.get(key)
            if future is None:
                future = Future()
                self._inflight[key] = future
                leader = True
        if not leader:
            while True:
                try:
                    return future.result(timeout=min(0.05, control.remaining("coalesced wait")))
                except FutureTimeoutError:
                    control.check("coalesced wait")
        try:
            while not self._admission.acquire(
                timeout=min(0.05, control.remaining("admission queue"))
            ):
                control.check("admission queue")
            try:
                result = self._materialize(request, control)
            finally:
                self._admission.release()
        except BaseException as error:
            future.set_exception(error)
            raise
        else:
            future.set_result(result)
            return result
        finally:
            with self._lock:
                self._inflight.pop(key, None)

    def _materialize(
        self, request: MaterializeDeepRequest, control: DeepRequestControl
    ) -> MaterializeDeepResult:
        started = control.started
        control.check("setup")
        if self.config.clang_analyzer_path is None:
            raise ValueError("deep materialization requires the native Clang analyzer")
        requested_builds = (
            self.config.build_scope.variants if request.builds is None else tuple(request.builds)
        )
        requested_scope = BuildScope(requested_builds)
        denied = set(requested_scope.variants) - set(self.config.build_scope.variants)
        if denied:
            raise ValueError("build scope is not operator-enabled: " + ", ".join(sorted(denied)))
        symbol = self.store.get_symbol(
            request.symbol_id, self.config.project_root, build_scope=requested_scope
        )
        if symbol is None:
            raise ValueError("requested symbol ID is not present in the selected build scope")
        roots = self.store.deep_definition_targets(
            symbol.id,
            self.config.project_root,
            build_scope=requested_scope,
            request_control=control,
        )
        if not roots:
            raise ValueError("requested symbol has no indexed definition TU in scope")
        selected, known, omitted = self._closure(roots, symbol.id, request.max_tus, control)
        closure_complete = omitted == 0
        limit_reason = "" if closure_complete else "max_tus"
        analyzer_identity = _file_digest(self.config.clang_analyzer_path, control)
        identities = {
            item.translation_unit_id: self._identity(item, analyzer_identity, control)
            for item in selected
        }
        closure_generation_id = _digest(
            [
                "deep-closure-generation-v1",
                *requested_scope.variants,
                str(closure_complete),
                *(f"{unit_id}\0{identities[unit_id]}" for unit_id in sorted(identities)),
                analyzer_identity,
                PROTOCOL,
                str(PROTOCOL_VERSION),
                str(SCHEMA_VERSION),
                IndexProfile.FULL.value,
            ],
            control,
        )
        materialization_id = _digest(
            [
                "deep-materialization-v1",
                symbol.id,
                closure_generation_id,
                str(known),
                str(omitted),
                limit_reason,
            ],
            control,
        )
        # A persisted cache identity describes the indexed snapshot. Verify the
        # snapshot still exists on disk before accepting a restart cache hit.
        self._revalidate(selected, control)
        configuration_by_id = self._load_configurations(requested_scope, control)
        configurations = []
        for target in selected:
            configuration = configuration_by_id.get(target.build_configuration_id)
            if configuration is None or configuration.command_hash != target.command_hash:
                raise RuntimeError(
                    "navigation index is stale; refresh it before materializing deep facts"
                )
            configurations.append(configuration)
        indexed_states = self.store.translation_unit_states(
            self.config.project_root,
            build_scope=requested_scope,
            translation_unit_ids=tuple(item.translation_unit_id for item in selected),
            request_control=control,
        )
        if all(
            (state := indexed_states.get(item.translation_unit_id)) is not None
            and state.index_profile is IndexProfile.FULL
            and state.cfg_facts_complete
            and state.data_flow_facts_complete
            and state.summary_facts_complete
            for item in selected
        ):
            cached = self.store.deep_cached_closure(
                closure_generation_id, identities, self.config.project_root
            )
            if not self.store.deep_materialization_matches(
                materialization_id, identities, self.config.project_root
            ):
                if cached is None:
                    cached = self.store.publish_full_profile_materialization(
                        self.config.project_root,
                        root_symbol_id=symbol.id,
                        materialization_id=materialization_id,
                        closure_generation_id=closure_generation_id,
                        identities=identities,
                        distances={item.translation_unit_id: item.distance for item in selected},
                        closure_complete=closure_complete,
                        known_tus=known,
                        omitted_tus=omitted,
                        limit_reason=limit_reason,
                        analyzer_identity=analyzer_identity,
                        analyzer_version="unknown",
                        protocol=PROTOCOL,
                        protocol_version=PROTOCOL_VERSION,
                        profile=IndexProfile.FULL,
                        build_scope=requested_scope,
                        request_control=control,
                    )
                else:
                    # Replacing exact cache rows would cascade-delete tokens already
                    # attached to this closure; a new root needs only another alias.
                    cached = self.store.publish_deep_materialization_alias(
                        self.config.project_root,
                        root_symbol_id=symbol.id,
                        materialization_id=materialization_id,
                        closure_generation_id=closure_generation_id,
                        identities=identities,
                        distances={item.translation_unit_id: item.distance for item in selected},
                        closure_complete=closure_complete,
                        known_tus=known,
                        omitted_tus=omitted,
                        limit_reason=limit_reason,
                        build_scope=requested_scope,
                        request_control=control,
                    )
            if cached is None:
                raise RuntimeError("full-profile cache changed before result publication")
            cached_provenance = cached[0]
            provenance = DeepProvenance(
                analyzer_identity=cached_provenance.analyzer_identity,
                analyzer_version=cached_provenance.analyzer_version,
                protocol=cached_provenance.protocol,
                protocol_version=cached_provenance.protocol_version,
                fact_schema_version=cached_provenance.fact_schema_version,
                profile=cached_provenance.profile,
                build_scope=list(requested_scope.variants),
                closure_generation_id=closure_generation_id,
            )
            return self._result(
                "cache_hit",
                materialization_id,
                symbol.id,
                selected,
                closure_complete,
                known,
                omitted,
                limit_reason,
                True,
                started,
                identities,
                provenance,
            )
        cached = self.store.deep_cached_closure(
            closure_generation_id, identities, self.config.project_root
        )
        if cached is not None:
            if not self.store.deep_materialization_matches(
                materialization_id, identities, self.config.project_root
            ):
                cached = self.store.publish_deep_materialization_alias(
                    self.config.project_root,
                    root_symbol_id=symbol.id,
                    materialization_id=materialization_id,
                    closure_generation_id=closure_generation_id,
                    identities=identities,
                    distances={item.translation_unit_id: item.distance for item in selected},
                    closure_complete=closure_complete,
                    known_tus=known,
                    omitted_tus=omitted,
                    limit_reason=limit_reason,
                    build_scope=requested_scope,
                    request_control=control,
                )
            cached_provenance = cached[0]
            provenance = DeepProvenance(
                analyzer_identity=cached_provenance.analyzer_identity,
                analyzer_version=cached_provenance.analyzer_version,
                protocol=cached_provenance.protocol,
                protocol_version=cached_provenance.protocol_version,
                fact_schema_version=cached_provenance.fact_schema_version,
                profile=cached_provenance.profile,
                build_scope=list(requested_scope.variants),
                closure_generation_id=closure_generation_id,
            )
            return self._result(
                "cache_hit",
                materialization_id,
                symbol.id,
                selected,
                closure_complete,
                known,
                omitted,
                limit_reason,
                True,
                started,
                identities,
                provenance,
            )
        worker_count = min(2, self.config.analyzer_max_workers, len(configurations))
        per_process_timeout = control.remaining("analyzer setup")
        decoded_limit = min(request.max_decoded_bytes, self.config.analyzer_max_decoded_bytes)
        spool_limit = min(
            request.max_spool_bytes,
            self.config.analyzer_max_spool_bytes or request.max_spool_bytes,
        )
        spool_files = min(
            request.max_spool_files,
            self.config.analyzer_max_spool_files or request.max_spool_files,
        )
        client = NativeAnalyzerClient(
            self.config.clang_analyzer_path,
            timeout_seconds=per_process_timeout,
            max_input_bytes=self.config.analyzer_max_input_bytes,
            max_output_bytes=self.config.analyzer_max_output_bytes,
            max_decoded_bytes=decoded_limit,
            max_record_bytes=self.config.analyzer_max_record_bytes,
            max_stderr_bytes=self.config.analyzer_max_stderr_bytes,
            profile=IndexProfile.FULL,
            deadline_monotonic=control.deadline,
            external_cancelled=control.cancelled,
            decoded_budget=_ResourceBudget(
                decoded_limit, "deep materialization exceeded the aggregate decoded output limit"
            ),
        )
        info = client.probe()
        control.check("analyzer probe")
        ingestor = NativeClangIngestor(
            client,
            max_workers=worker_count,
            max_spool_registries=max(2, min(2, len(configurations))),
            max_spool_bytes=spool_limit,
            max_spool_fds=spool_files,
            max_domain_batches=1,
            profile=IndexProfile.FULL,
        )
        batches = tuple(
            ingestor.iter_configuration_batches(self.config.project_root, configurations)
        )
        control.check("batch merge")
        derived_targets = self.store.deep_navigation_derived_targets(
            tuple(item.translation_unit_id for item in selected),
            self.config.project_root,
            request_control=control,
        )
        control.check("navigation target merge")
        merged = NativeClangIngestor._merge_batches(
            list(batches),
            profile=IndexProfile.FULL,
            additional_call_targets=derived_targets,
            check_callback=control.check,
        )
        batches = (merged,)
        self._revalidate(selected, control)
        if _file_digest(self.config.clang_analyzer_path, control) != analyzer_identity:
            raise RuntimeError("analyzer changed during deep materialization")
        post_configurations = self._load_configurations(requested_scope, control)
        if any(
            (configuration := post_configurations.get(item.build_configuration_id)) is None
            or configuration.command_hash != item.command_hash
            for item in selected
        ):
            raise RuntimeError("compile command changed during deep materialization")
        control.check("publication validation")
        self.store.validate_deep_navigation_parity(
            self.config.project_root, batches, request_control=control
        )
        self.store.apply_deep_overlay(
            self.config.project_root,
            batches,
            root_symbol_id=symbol.id,
            materialization_id=materialization_id,
            identities=identities,
            command_hashes={item.translation_unit_id: item.command_hash for item in selected},
            distances={item.translation_unit_id: item.distance for item in selected},
            deadline_monotonic=control.deadline,
            cancelled=control.cancelled,
            request_control=control,
            analyzer_identity=analyzer_identity,
            analyzer_version=info.analyzer_version,
            protocol=info.protocol,
            protocol_version=info.protocol_version,
            profile=IndexProfile.FULL,
            closure_complete=closure_complete,
            closure_generation_id=closure_generation_id,
            known_tus=known,
            omitted_tus=omitted,
            limit_reason=limit_reason,
            build_scope=requested_scope,
        )
        provenance = DeepProvenance(
            analyzer_identity=analyzer_identity,
            analyzer_version=info.analyzer_version,
            protocol=info.protocol,
            protocol_version=info.protocol_version,
            fact_schema_version=SCHEMA_VERSION,
            profile=IndexProfile.FULL,
            build_scope=list(requested_scope.variants),
            closure_generation_id=closure_generation_id,
        )
        return self._result(
            "complete" if closure_complete else "partial",
            materialization_id,
            symbol.id,
            selected,
            closure_complete,
            known,
            omitted,
            limit_reason,
            False,
            started,
            identities,
            provenance,
        )

    def _load_configurations(
        self, requested_scope: BuildScope, control: DeepRequestControl
    ) -> dict[str, BuildConfiguration]:
        configuration_by_id = {}
        variant_by_name = {variant.name: variant for variant in self.config.build_variants}
        for variant_name in requested_scope.variants:
            control.check("compilation database load")
            variant = variant_by_name.get(variant_name)
            if variant is None:
                raise ValueError(f"build scope is not configured: {variant_name}")
            database = CompilationDatabase.load(
                variant.compilation_database,
                build_variant=variant.name,
                check_cancelled=lambda: control.check("compilation database load"),
            )
            control.check("compilation database load")
            configuration_by_id.update(
                (configuration.id, configuration) for configuration in database.configurations
            )
        return configuration_by_id

    def _closure(
        self,
        roots: tuple[DeepTranslationUnitTarget, ...],
        root_symbol_id: str,
        max_tus: int,
        control: DeepRequestControl,
    ) -> tuple[tuple[DeepTranslationUnitTarget, ...], int, int]:
        ordered_roots = tuple(
            sorted(
                roots,
                key=lambda item: (
                    item.distance,
                    item.build_variant,
                    item.build_configuration_id,
                    item.translation_unit_id,
                ),
            )
        )
        known: dict[tuple[str, str], DeepTranslationUnitTarget] = {
            (item.build_variant, item.translation_unit_id): item for item in ordered_roots
        }
        if len(known) > max_tus:
            ordered = tuple(known.values())
            return ordered[:max_tus], len(ordered), len(ordered) - max_tus
        pending = [(item, root_symbol_id) for item in ordered_roots]
        expanded: set[tuple[str, str, str]] = set()
        while pending and len(known) <= max_tus:
            control.check("closure")
            pending.sort(
                key=lambda item: (
                    item[0].distance,
                    item[0].build_variant,
                    item[0].translation_unit_id,
                    item[1],
                )
            )
            current, owner_symbol_id = pending.pop(0)
            key = (current.build_variant, current.translation_unit_id)
            known.setdefault(key, current)
            if len(known) > max_tus:
                break
            expansion_key = (*key, owner_symbol_id)
            if expansion_key in expanded:
                continue
            expanded.add(expansion_key)
            pending.extend(
                (callee, symbol_id)
                for symbol_id, callee in self.store.deep_symbol_callees(
                    current,
                    (owner_symbol_id,),
                    self.config.project_root,
                    request_control=control,
                )
            )
        ordered = tuple(
            sorted(
                known.values(),
                key=lambda item: (item.distance, item.build_variant, item.translation_unit_id),
            )
        )
        omitted = max(0, len(ordered) - max_tus)
        if pending and omitted == 0:
            omitted = 1
        return ordered[:max_tus], len(ordered), omitted

    def _identity(
        self,
        target: DeepTranslationUnitTarget,
        analyzer_identity: str,
        control: DeepRequestControl,
    ) -> str:
        return _digest(
            [
                str(self.config.project_root.resolve(strict=False)),
                target.build_variant,
                target.build_configuration_id,
                target.command_hash,
                target.translation_unit_id,
                target.content_hash,
                *(f"{path}\0{content_hash}" for path, content_hash in target.dependencies),
                analyzer_identity,
                str(PROTOCOL_VERSION),
                str(SCHEMA_VERSION),
                IndexProfile.FULL.value,
            ],
            control,
        )

    def _revalidate(
        self, targets: tuple[DeepTranslationUnitTarget, ...], control: DeepRequestControl
    ) -> None:
        for target in targets:
            control.check("input hashing")
            if _file_digest(target.source_path, control) != target.content_hash:
                raise RuntimeError("source changed during deep materialization")
            for path, expected in target.dependencies:
                if _file_digest(path, control) != expected:
                    raise RuntimeError("dependency changed during deep materialization")

    @staticmethod
    def _result(
        status: str,
        materialization_id: str,
        symbol_id: str,
        targets: tuple[DeepTranslationUnitTarget, ...],
        closure_complete: bool,
        known: int,
        omitted: int,
        limit_reason: str,
        cache_hit: bool,
        started: float,
        identities: dict[str, str],
        provenance: DeepProvenance,
    ) -> MaterializeDeepResult:
        return MaterializeDeepResult(
            status=status,
            materialization_id=materialization_id,
            root_symbol_id=symbol_id,
            units=[
                MaterializedUnit(
                    build_variant=item.build_variant,
                    translation_unit_id=item.translation_unit_id,
                    build_configuration_id=item.build_configuration_id,
                    identity_hash=identities[item.translation_unit_id],
                    distance=item.distance,
                    cache_hit=cache_hit,
                    control_flow=True,
                    data_flow=True,
                    summaries=closure_complete,
                    bindings=closure_complete,
                )
                for item in targets
            ],
            closure_complete=closure_complete,
            known_tus=known,
            omitted_tus=omitted,
            limit_reason=limit_reason,
            cache_hit=cache_hit,
            elapsed_seconds=time.monotonic() - started,
            provenance=provenance,
        )


def _digest(parts: list[str], control: DeepRequestControl | None = None) -> str:
    digest = hashlib.sha256()
    for part in parts:
        if control is not None:
            control.check("identity hashing")
        digest.update(part.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
    return digest.hexdigest()


def _file_digest(path: Path, control: DeepRequestControl | None = None) -> str:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                if control is not None:
                    control.check("file hashing")
                digest.update(chunk)
        if control is not None:
            control.check("file hashing")
        return digest.hexdigest()
    except TimeoutError:
        raise
    except OSError as error:
        raise RuntimeError("deep materialization input is unavailable") from error
