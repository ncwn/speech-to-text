"""Backend execution with one comparable whole-corpus wall boundary."""

from __future__ import annotations

import math
import os
from numbers import Real
from pathlib import Path
from typing import Any

import soundfile as sf

from stt.audio import PreparedAudio
from stt.backends.base import ASRBackend
from stt.provenance import ModelBinding, ModelProvenance, ProvenanceError
from stt.registry import get_backend
from stt.results import Segment, TranscriptionResult
from stt.telemetry import ResourceUsage, describe_host, measure


class BackendUnavailableError(RuntimeError):
    """The requested backend runtime is not installed or usable."""


_CONTRACT_ERROR_PREFIX = "backend contract violation"
_TIMING_TOLERANCE_S = 1e-6


def create_backend(
    backend_name: str,
    model: str | None,
    options: dict[str, Any],
    binding: ModelBinding | None = None,
):
    cls = get_backend(backend_name)
    available, reason = cls.is_available()
    if not available:
        raise BackendUnavailableError(f"{backend_name} is unavailable: {reason}")
    kwargs = {key: value for key, value in options.items() if value is not None}
    instance = cls(model, **kwargs) if model else cls(**kwargs)
    instance.bind_model(binding)
    return instance


def mark_run_trust(
    results: list[TranscriptionResult],
    *,
    extra_issue: str | None = None,
    require_provenance: bool = False,
) -> bool:
    """Stamp a corpus trusted only when every expected record completed."""
    run_issues: list[str] = []
    if extra_issue:
        run_issues.append(extra_issue)
    failed = sum(result.error is not None for result in results)
    if failed:
        run_issues.append(f"{failed}/{len(results)} backend result(s) failed")
    if require_provenance:
        provenance_failures = 0
        for result in results:
            try:
                provenance = ModelProvenance.from_dict(result.model_provenance or {})
                if not provenance.complete:
                    provenance_failures += 1
            except (KeyError, TypeError, ValueError, ProvenanceError):
                provenance_failures += 1
        if provenance_failures:
            run_issues.append(
                f"{provenance_failures}/{len(results)} result(s) lack complete model provenance"
            )

    complete = bool(results) and not run_issues
    for result in results:
        issues = list(dict.fromkeys([*result.trust_issues, *run_issues]))
        if result.error and result.error not in issues:
            issues.append(result.error)
        result.trusted = complete and not issues
        result.trust_issues = issues
    return complete


def _error_results(
    paths: list[Path],
    *,
    backend: str,
    model: str,
    language: str | None,
    detail: str,
    metadata: dict[str, Any],
    model_provenance: dict[str, Any] | None = None,
) -> list[TranscriptionResult]:
    return [
        TranscriptionResult(
            audio_path=str(path),
            text="",
            backend=backend,
            model=model,
            language=language,
            error=detail,
            metadata=dict(metadata),
            model_provenance=model_provenance,
        )
        for path in paths
    ]


def _finite_number(value: object) -> bool:
    """Return whether ``value`` is a real, finite number (not a boolean)."""
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _provenance_identity_issues(
    model_provenance: dict[str, Any] | None,
    *,
    backend_name: str,
    model: object,
    instance: Any,
) -> list[str]:
    """Check collected provenance identity using the backend alias contract."""
    if model_provenance is None:
        return []
    try:
        provenance = ModelProvenance.from_dict(model_provenance)
        expected_backends = ASRBackend._identity_aliases(
            instance,
            backend_name,
            ("provenance_backend_aliases", "backend_aliases"),
        )
        expected_models = ASRBackend._identity_aliases(
            instance,
            model,
            ("provenance_model_aliases", "model_aliases"),
        )
    except (KeyError, TypeError, ValueError, ProvenanceError) as exc:
        return [f"model provenance identity could not be validated: {type(exc).__name__}: {exc}"]
    except Exception as exc:  # noqa: BLE001 - a custom alias property must fail closed
        return [f"model provenance identity could not be validated: {type(exc).__name__}: {exc}"]

    issues: list[str] = []
    if provenance.backend not in expected_backends:
        issues.append(
            "model provenance backend mismatch: "
            f"{provenance.backend!r} not in {sorted(expected_backends)!r}"
        )
    if provenance.requested_model not in expected_models:
        issues.append(
            "model provenance model mismatch: "
            f"{provenance.requested_model!r} not in {sorted(expected_models)!r}"
        )
    return issues


def _coerce_audio_path(value: object) -> tuple[Path | None, str | None]:
    """Coerce a backend path value, returning a per-item issue on bad input."""
    if not isinstance(value, (str, os.PathLike)):
        return None, f"audio_path must be a non-empty path-like string, got {type(value).__name__}"
    try:
        raw = os.fspath(value)
    except Exception as exc:  # noqa: BLE001 - custom path-like values must fail closed
        return None, f"audio_path could not be coerced to a path: {type(exc).__name__}: {exc}"
    if not isinstance(raw, str):
        return None, f"audio_path must resolve to a string, got {type(raw).__name__}"
    if not raw.strip():
        return None, "audio_path must be a non-empty path-like string"
    try:
        return Path(raw), None
    except Exception as exc:  # noqa: BLE001 - malformed path values stay per-item diagnostics
        return None, f"audio_path is malformed: {type(exc).__name__}: {exc}"


def _backend_contract_issues(
    result: TranscriptionResult,
    *,
    backend_name: str,
    model: object,
    duration_s: float,
) -> list[str]:
    """Return runtime-visible violations of the :class:`ASRBackend` contract.

    The test-suite has a deliberately small conformance helper.  The execution
    boundary repeats the checks that affect trust so a production run cannot
    silently publish a success-shaped result from a broken backend.
    """
    issues: list[str] = []
    if result.backend != backend_name:
        issues.append(
            f"backend identity mismatch: returned {result.backend!r}, expected {backend_name!r}"
        )
    if result.model != model:
        issues.append(f"model identity mismatch: returned {result.model!r}, expected {model!r}")
    if not isinstance(result.metadata, dict):
        issues.append(f"metadata must be a dict, got {type(result.metadata).__name__}")

    text = result.text
    if not isinstance(text, str):
        issues.append(f"text must be a string, got {type(text).__name__}")
    elif result.error is None and not text.strip():
        issues.append("empty transcript with no error set")

    error = result.error
    if error is not None and not isinstance(error, str):
        issues.append(f"error must be a string or None, got {type(error).__name__}")
    elif isinstance(error, str):
        if not error.strip():
            issues.append("error must be a non-empty string or None")
        elif isinstance(text, str) and text.strip():
            issues.append("result reports an error but also returned text")

    segments = result.segments
    if segments is None:
        return issues
    if not isinstance(segments, list):
        issues.append(f"segments must be a list or None, got {type(segments).__name__}")
        return issues

    previous_timing: tuple[float, float] | None = None
    for index, segment in enumerate(segments):
        label = f"segment[{index}]"
        if not isinstance(segment, Segment):
            issues.append(f"{label} must be a Segment, got {type(segment).__name__}")
            continue
        if not isinstance(segment.text, str):
            issues.append(f"{label}.text must be a string")

        valid_start = _finite_number(segment.start)
        valid_end = _finite_number(segment.end)
        if not valid_start:
            issues.append(f"{label}.start must be a finite number")
        if not valid_end:
            issues.append(f"{label}.end must be a finite number")
        if not (valid_start and valid_end):
            continue

        start = float(segment.start)
        end = float(segment.end)
        if start < 0:
            issues.append(f"{label}.start is before audio: {start}")
        if end < start:
            issues.append(f"{label}.end is before its start: {end} < {start}")
        if start > duration_s + _TIMING_TOLERANCE_S:
            issues.append(f"{label}.start is after audio duration: {start} > {duration_s}")
        if end > duration_s + _TIMING_TOLERANCE_S:
            issues.append(f"{label}.end is after audio duration: {end} > {duration_s}")
        timing = (start, end)
        if previous_timing is not None and timing < previous_timing:
            issues.append(
                f"segments are out of order at index {index}: "
                f"{timing!r} follows {previous_timing!r}"
            )
        previous_timing = timing

        confidence = segment.confidence
        if confidence is not None:
            if not _finite_number(confidence):
                issues.append(f"{label}.confidence must be a finite number or None")
            elif not 0.0 <= float(confidence) <= 1.0:
                issues.append(f"{label}.confidence is outside [0, 1]: {confidence}")
        if segment.speaker is not None and not isinstance(segment.speaker, str):
            issues.append(f"{label}.speaker must be a string or None")
        if not isinstance(segment.source, str) or not segment.source:
            issues.append(f"{label}.source must be a non-empty string")
    return issues


def _record_contract_issues(result: TranscriptionResult, issues: list[str]) -> None:
    """Attach contract diagnostics while keeping the result serializable."""
    if not issues:
        return
    unique = list(dict.fromkeys(issues))
    existing = result.trust_issues
    if existing is None:
        existing_issues: list[str] = []
    elif isinstance(existing, list):
        existing_issues = [str(issue) for issue in existing]
    else:
        existing_issues = [str(existing)]
    result.trust_issues = list(dict.fromkeys([*existing_issues, *unique]))

    if not isinstance(result.metadata, dict):
        result.metadata = {}
    prior = result.metadata.get("contract_violations", [])
    if isinstance(prior, list):
        prior_issues = [str(issue) for issue in prior]
    elif prior:
        prior_issues = [str(prior)]
    else:
        prior_issues = []
    result.metadata["contract_violations"] = list(dict.fromkeys([*prior_issues, *unique]))

    # Empty success-shaped output is the contract failure that would otherwise
    # be scored as a terrible transcription.  Turn it into a normal failed
    # record; identity/timing violations retain their text for diagnostics and
    # are made ineligible through trust_issues.
    empty_issue = next(
        (
            issue
            for issue in unique
            if issue == "empty transcript with no error set"
            or issue.startswith("text must be a string")
        ),
        None,
    )
    if empty_issue is not None and result.error is None:
        result.error = f"{_CONTRACT_ERROR_PREFIX}: {empty_issue}"
    elif result.error is not None and (
        not isinstance(result.error, str) or not result.error.strip()
    ):
        result.error = f"{_CONTRACT_ERROR_PREFIX}: error field has invalid type or is empty"
    if any(issue.startswith("text must be a string") for issue in unique):
        result.text = ""
    path_issue = next(
        (issue for issue in unique if issue.startswith("audio_path ")),
        None,
    )
    if path_issue is not None and result.error is None:
        if isinstance(result.text, str) and result.text:
            result.metadata["contract_original_text"] = result.text
            result.text = ""
        result.error = f"{_CONTRACT_ERROR_PREFIX}: {path_issue}"
    segment_items = result.segments if isinstance(result.segments, list) else ()
    malformed_segment_shape = any(
        " must be a Segment" in issue
        or any(
            issue.startswith(f"segment[{index}].{field} must be")
            for index in range(len(segment_items))
            for field in ("text", "start", "end", "confidence", "speaker", "source")
        )
        for issue in unique
    )
    if result.segments is not None and any(
        issue.startswith("segments must be a list") for issue in unique
    ):
        malformed_segment_shape = True
    if result.segments is not None and malformed_segment_shape:
        # A malformed segment payload must not make JSONL writing fail later.
        result.segments = None


def transcribe_corpus(
    instance: Any,
    backend_name: str,
    files: list[PreparedAudio],
    language: str | None,
    batch_size: int,
    *,
    profile: bool,
    load_usage: ResourceUsage | None = None,
    host: dict[str, Any] | None = None,
    phase: str = "corpus",
    require_provenance: bool = False,
) -> tuple[list[TranscriptionResult], ResourceUsage]:
    """Transcribe a complete corpus once and retain failed work in its wall."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not files:
        raise ValueError("at least one prepared audio input is required")

    host = host or describe_host()
    paths = [item.prepared_path for item in files]
    expected_model = instance.model
    provenance_collector = getattr(instance, "model_provenance", None)
    provenance_required = (
        require_provenance
        or getattr(instance, "model_binding", None) is not None
        or callable(provenance_collector)
    )
    set_execution_settings = getattr(instance, "set_execution_settings", None)
    if callable(set_execution_settings):
        set_execution_settings(language=language, batch_size=batch_size)
    device = getattr(instance, "resolved_device", None)
    model_provenance: dict[str, Any] | None = None
    provenance_issue: str | None = None
    try:
        model_provenance = instance.model_provenance().finalized().to_dict()
    except Exception as exc:  # noqa: BLE001 - preserve diagnostic work, fail trust closed
        provenance_issue = (
            f"{phase} model provenance unavailable before corpus call: {type(exc).__name__}: {exc}"
        )
    on_gpu = device in {"mps", "cuda"}
    with measure(
        sample_gpu=on_gpu,
        profile=profile,
        device=device,
        phase=phase,
    ) as measured:
        try:
            batch = instance.transcribe(paths, language=language, batch_size=batch_size)
        except Exception as exc:  # noqa: BLE001 - preserve a diagnostic for every input
            detail = f"{type(exc).__name__}: {exc}"
            batch = _error_results(
                paths,
                backend=backend_name,
                model=expected_model,
                language=language,
                detail=detail,
                metadata={"corpus_exception": True},
                model_provenance=model_provenance,
            )

    usage = measured[0]
    try:
        # A device fallback or lazy loader resolution may happen during the
        # corpus call. The post-call manifest is authoritative for the result.
        model_provenance = instance.model_provenance().finalized().to_dict()
    except Exception as exc:  # noqa: BLE001 - preserve diagnostic work, fail trust closed
        post_issue = f"{phase} post-call model provenance unavailable: {type(exc).__name__}: {exc}"
        provenance_issue = "; ".join(issue for issue in (provenance_issue, post_issue) if issue)
    provenance_identity_issues = _provenance_identity_issues(
        model_provenance,
        backend_name=backend_name,
        model=expected_model,
        instance=instance,
    )
    try:
        batch = list(batch)
    except Exception as exc:  # noqa: BLE001 - retain one diagnostic per input
        detail = (
            f"{_CONTRACT_ERROR_PREFIX}: {backend_name} returned an invalid result collection "
            f"({type(exc).__name__}: {exc})"
        )
        batch = _error_results(
            paths,
            backend=backend_name,
            model=expected_model,
            language=language,
            detail=detail,
            metadata={"corpus_contract_error": True},
            model_provenance=model_provenance,
        )
        for result in batch:
            result.trust_issues.append(detail)
    if len(batch) != len(files):
        detail = f"{backend_name} returned {len(batch)} result(s) for {len(files)} input(s)"
        batch = _error_results(
            paths,
            backend=backend_name,
            model=expected_model,
            language=language,
            detail=detail,
            metadata={"corpus_contract_error": True},
            model_provenance=model_provenance,
        )

    fallback = on_gpu and getattr(instance, "resolved_device", None) not in {"mps", "cuda"}
    if fallback:
        usage.gpu_util = None
        usage.gpu_util_mean = None
        usage.gpu_util_peak = None
        usage.gpu_mem = None

    contract_failure_count = 0
    for index, (result, prepared) in enumerate(zip(batch, files, strict=True)):
        contract_issues: list[str] = []
        if not isinstance(result, TranscriptionResult):
            detail = (
                f"{_CONTRACT_ERROR_PREFIX}: result at index {index} is "
                f"{type(result).__name__}, expected TranscriptionResult"
            )
            result = TranscriptionResult(
                audio_path=str(prepared.prepared_path),
                text="",
                backend=backend_name,
                model=expected_model,
                language=language,
                error=detail,
                metadata={"corpus_contract_error": True},
                trust_issues=[detail],
            )
            batch[index] = result
            contract_issues.append(detail)
        returned, path_issue = _coerce_audio_path(result.audio_path)
        if path_issue is not None:
            contract_issues.append(path_issue)
            if not isinstance(result.metadata, dict):
                contract_issues.append(
                    f"metadata must be a dict, got {type(result.metadata).__name__}"
                )
                result.metadata = {}
            result.metadata["returned_audio_path"] = repr(result.audio_path)
            result.audio_path = str(prepared.prepared_path)
            returned = prepared.prepared_path
        else:
            assert returned is not None
            try:
                returned = returned.resolve(strict=False)
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                path_issue = f"audio_path could not be resolved: {type(exc).__name__}: {exc}"
                contract_issues.append(path_issue)
                if not isinstance(result.metadata, dict):
                    contract_issues.append(
                        f"metadata must be a dict, got {type(result.metadata).__name__}"
                    )
                    result.metadata = {}
                result.metadata["returned_audio_path"] = repr(result.audio_path)
                result.audio_path = str(prepared.prepared_path)
                returned = prepared.prepared_path
        expected = prepared.prepared_path.resolve(strict=False)
        if returned != expected:
            raise RuntimeError(
                f"{backend_name} returned results out of order: expected "
                f"{prepared.prepared_path}, got {result.audio_path}"
            )
        result.audio_path = str(prepared.prepared_path)
        result.audio_id = prepared.audio_id
        result.source_path = str(prepared.source_path)
        result.source_sha256 = prepared.source_sha256
        result.reference_id = prepared.reference_id
        result.model_provenance = model_provenance
        result.audio_duration_s = sf.info(prepared.prepared_path).duration
        contract_issues.extend(
            _backend_contract_issues(
                result,
                backend_name=backend_name,
                model=expected_model,
                duration_s=result.audio_duration_s,
            )
        )
        if not isinstance(result.metadata, dict):
            # Keep enrichment and JSONL writing safe while retaining the
            # violation in the per-result diagnostics below.
            contract_issues.append(f"metadata must be a dict, got {type(result.metadata).__name__}")
            result.metadata = {}
        if contract_issues:
            contract_failure_count += 1
        result.metadata.setdefault("host", host)
        result.metadata["corpus_index"] = index
        result.metadata["corpus_size"] = len(files)
        result.metadata["requested_batch_size"] = batch_size
        result.metadata["corpus_wall_s"] = usage.wall_s
        if fallback:
            result.metadata["gpu_fallback"] = True
        if index == 0:
            result.resources = usage
            result.metadata["resource_scope"] = "corpus"
            if load_usage is not None:
                result.metadata["load"] = load_usage.to_dict(include_series=False)
        if load_usage is not None:
            result.metadata["load_s"] = load_usage.wall_s
        _record_contract_issues(result, contract_issues)

    contract_issue = (
        f"{contract_failure_count}/{len(batch)} result(s) violate the backend contract"
        if contract_failure_count
        else None
    )
    extra_issues = [
        issue
        for issue in (
            provenance_issue if provenance_required else None,
            *provenance_identity_issues,
            contract_issue,
        )
        if issue
    ]
    mark_run_trust(
        batch,
        extra_issue="; ".join(extra_issues) if extra_issues else None,
        require_provenance=provenance_required,
    )
    return batch, usage
