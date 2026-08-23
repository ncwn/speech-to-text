"""Render measured documentation from identity-verified result artifacts.

Evidence blocks are deliberately fail closed: a numeric table is rendered only
when every declared run is trusted, complete, and covers exactly the reference
corpus named by the manifest.  Legacy artifacts remain inspectable elsewhere,
but can produce only an explicit unverified status here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stt.burmese import NormalizeOptions
from stt.derive import DeriveContext, DerivedTable, Requirements, baseline_table, validate_settings
from stt.derive import derive as run_deriver
from stt.evaluate import load_references, score_results
from stt.measurement import MeasurementError
from stt.provenance import ModelProvenance, ProvenanceError
from stt.results import TranscriptionResult, read_jsonl

MARKER_PREFIX = "stt-evidence"
_MARKER_RE = re.compile(r"<!-- stt-evidence:([^:\s]+):(start|end) -->")
UNVERIFIED_MESSAGE = (
    "> **Unverified legacy evidence.** Numeric publication is blocked because the declared "
    "artifacts are missing trusted audio identity, complete corpus coverage, or matching "
    "provenance. Regenerate the runs before publishing measured results."
)


class EvidenceError(ValueError):
    """A manifest or artifact cannot support publishable measurements."""


@dataclass(frozen=True)
class MetricSpec:
    name: str
    label: str
    digits: int = 4


@dataclass(frozen=True)
class RunSpec:
    artifact: Path
    label: str
    backend: str
    model: str
    settings: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BlockSpec:
    id: str
    corpus: str
    reference: Path
    runs: tuple[RunSpec, ...]
    metrics: tuple[MetricSpec, ...]
    status_only: bool = False
    status_reason: str | None = None
    deriver: str | None = None
    reference_sha256: str | None = None
    expected_reference_count: int | None = None
    normalization: dict[str, Any] = field(default_factory=dict)
    aligned_segments: bool = False
    source_kind: str = "transcription-jsonl"
    source: Path | None = None


@dataclass(frozen=True)
class EvidenceManifest:
    path: Path
    document: Path
    blocks: tuple[BlockSpec, ...]


@dataclass
class EvidenceCheck:
    """Outcome returned to the CLI and documentation tests."""

    document: Path
    matches: bool
    publishable: bool
    issues: list[str] = field(default_factory=list)
    expected_text: str = ""

    @property
    def ok(self) -> bool:
        """Whether the checked-in document matches the renderer.

        Publication eligibility is deliberately separate: a synchronized
        unverified-status block is a valid documentation state while legacy
        evidence awaits regeneration.
        """
        return self.matches


def _required(mapping: dict[str, Any], name: str, context: str) -> Any:
    if name not in mapping:
        raise EvidenceError(f"{context} is missing {name!r}")
    return mapping[name]


def _path(root: Path, value: Any, context: str) -> Path:
    if not isinstance(value, str) or not value:
        raise EvidenceError(f"{context} must be a non-empty path string")
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_manifest(path: Path) -> EvidenceManifest:
    """Load and strictly validate a version-1 JSON evidence manifest."""
    path = path.resolve()
    root = path.parent.parent if path.parent.name == "evidence" else path.parent
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read evidence manifest {path}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise EvidenceError("evidence manifest must be an object with version 1")
    unknown = set(raw) - {"version", "document", "blocks"}
    if unknown:
        raise EvidenceError(f"unknown manifest field(s): {', '.join(sorted(unknown))}")

    document = _path(root, _required(raw, "document", "manifest"), "document")
    raw_blocks = _required(raw, "blocks", "manifest")
    if not isinstance(raw_blocks, list) or not raw_blocks:
        raise EvidenceError("manifest blocks must be a non-empty list")

    blocks: list[BlockSpec] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw_blocks):
        context = f"blocks[{index}]"
        if not isinstance(item, dict):
            raise EvidenceError(f"{context} must be an object")
        unknown = set(item) - {
            "id",
            "corpus",
            "reference",
            "runs",
            "metrics",
            "status_only",
            "status_reason",
            "deriver",
            "reference_sha256",
            "expected_reference_count",
            "normalization",
            "aligned_segments",
            "source_kind",
            "source",
        }
        if unknown:
            raise EvidenceError(f"{context} has unknown field(s): {', '.join(sorted(unknown))}")
        block_id = _required(item, "id", context)
        if not isinstance(block_id, str) or not block_id or any(c.isspace() for c in block_id):
            raise EvidenceError(f"{context}.id must be a non-empty whitespace-free string")
        if block_id in seen_ids:
            raise EvidenceError(f"duplicate evidence block id {block_id!r}")
        seen_ids.add(block_id)

        source_kind = str(item.get("source_kind", "transcription-jsonl"))
        if source_kind not in {"transcription-jsonl", "baseline-v2"}:
            raise EvidenceError(f"{context}.source_kind is unsupported: {source_kind!r}")
        raw_runs = item.get("runs", [])
        if not isinstance(raw_runs, list):
            raise EvidenceError(f"{context}.runs must be a list")
        if source_kind == "transcription-jsonl" and not raw_runs:
            raise EvidenceError(f"{context}.runs must be a non-empty list")
        source = (
            _path(root, item["source"], f"{context}.source")
            if item.get("source") is not None
            else None
        )
        if source_kind != "transcription-jsonl" and source is None:
            raise EvidenceError(f"{context}.source is required for {source_kind}")
        runs: list[RunSpec] = []
        for run_index, run in enumerate(raw_runs):
            run_context = f"{context}.runs[{run_index}]"
            if not isinstance(run, dict):
                raise EvidenceError(f"{run_context} must be an object")
            unknown = set(run) - {"artifact", "label", "backend", "model", "settings"}
            if unknown:
                raise EvidenceError(
                    f"{run_context} has unknown field(s): {', '.join(sorted(unknown))}"
                )
            runs.append(
                RunSpec(
                    artifact=_path(
                        root, _required(run, "artifact", run_context), f"{run_context}.artifact"
                    ),
                    label=str(_required(run, "label", run_context)),
                    backend=str(_required(run, "backend", run_context)),
                    model=str(_required(run, "model", run_context)),
                    settings=dict(run.get("settings", {})),
                )
            )

        raw_metrics = item.get("metrics", [])
        if not isinstance(raw_metrics, list):
            raise EvidenceError(f"{context}.metrics must be a list")
        metrics: list[MetricSpec] = []
        for metric_index, metric in enumerate(raw_metrics):
            metric_context = f"{context}.metrics[{metric_index}]"
            if not isinstance(metric, dict):
                raise EvidenceError(f"{metric_context} must be an object")
            unknown = set(metric) - {"name", "label", "digits"}
            if unknown:
                raise EvidenceError(
                    f"{metric_context} has unknown field(s): {', '.join(sorted(unknown))}"
                )
            name = str(_required(metric, "name", metric_context))
            if name not in {"cer", "wer", "rtf", "n_total", "n_scored"}:
                raise EvidenceError(f"unsupported evidence metric {name!r}")
            digits = metric.get("digits", 4)
            if not isinstance(digits, int) or not 0 <= digits <= 9:
                raise EvidenceError(f"{metric_context}.digits must be an integer from 0 to 9")
            metrics.append(
                MetricSpec(name=name, label=str(metric.get("label", name.upper())), digits=digits)
            )

        status_only = item.get("status_only", False)
        if not isinstance(status_only, bool):
            raise EvidenceError(f"{context}.status_only must be a boolean")
        aligned_segments = item.get("aligned_segments", False)
        if not isinstance(aligned_segments, bool):
            raise EvidenceError(f"{context}.aligned_segments must be a boolean")
        if not metrics and not status_only and item.get("deriver") is None:
            raise EvidenceError(f"{context} needs metrics or status_only=true")
        blocks.append(
            BlockSpec(
                id=block_id,
                corpus=str(_required(item, "corpus", context)),
                reference=_path(
                    root, _required(item, "reference", context), f"{context}.reference"
                ),
                runs=tuple(runs),
                metrics=tuple(metrics),
                status_only=status_only,
                status_reason=(
                    str(item["status_reason"]) if item.get("status_reason") is not None else None
                ),
                deriver=(str(item["deriver"]) if item.get("deriver") is not None else None),
                reference_sha256=(
                    str(item["reference_sha256"])
                    if item.get("reference_sha256") is not None
                    else None
                ),
                expected_reference_count=(
                    int(item["expected_reference_count"])
                    if item.get("expected_reference_count") is not None
                    else None
                ),
                normalization=dict(item.get("normalization", {})),
                aligned_segments=aligned_segments,
                source_kind=source_kind,
                source=source,
            )
        )
    return EvidenceManifest(path=path, document=document, blocks=tuple(blocks))


def _identity(result: TranscriptionResult) -> str | None:
    value = getattr(result, "audio_id", None)
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"pcm16:([1-9]\d*):([1-9]\d*):([0-9a-f]{64})", value)
    return value if match else None


def _provenance_issues(result: TranscriptionResult, run: RunSpec) -> list[str]:
    """Validate the immutable model manifest carried by a result record."""
    raw = result.model_provenance
    if not isinstance(raw, dict):
        return ["missing model provenance"]
    issues: list[str] = []
    try:
        provenance = ModelProvenance.from_dict(raw)
    except (KeyError, TypeError, ValueError, ProvenanceError) as exc:
        return [f"invalid model provenance: {exc}"]
    if provenance.backend != run.backend:
        issues.append(
            f"model provenance backend mismatch ({provenance.backend!r} != {run.backend!r})"
        )
    if provenance.requested_model != run.model:
        issues.append(
            f"model provenance model mismatch ({provenance.requested_model!r} != {run.model!r})"
        )
    if not provenance.complete:
        issues.append("model provenance is incomplete or ineligible")
    if provenance.adapter_git_dirty:
        issues.append("model provenance records a dirty worktree")
    if provenance.fallback_history:
        issues.append("model provenance records a runtime fallback")
    finalized = provenance.finalized()
    if provenance.content_sha256 != finalized.content_sha256:
        issues.append("model provenance content hash is inconsistent")
    if provenance.execution_sha256 != finalized.execution_sha256:
        issues.append("model provenance execution hash is inconsistent")
    return issues


def _validate_run(run: RunSpec, expected_reference_ids: set[str]) -> list[TranscriptionResult]:
    if not run.artifact.is_file():
        raise EvidenceError(f"missing artifact: {run.artifact}")
    results = read_jsonl(run.artifact)
    if not results:
        raise EvidenceError(f"empty artifact: {run.artifact}")
    if any(result.backend != run.backend for result in results):
        found = sorted({result.backend for result in results})
        raise EvidenceError(
            f"{run.label}: backend mismatch, expected {run.backend!r}, found {found}"
        )
    if any(result.model != run.model for result in results):
        found = sorted({result.model for result in results})
        raise EvidenceError(f"{run.label}: model mismatch, expected {run.model!r}, found {found}")

    invalid: list[str] = []
    reference_ids: list[str] = []
    audio_ids: list[str] = []
    execution_ids: set[str] = set()
    for index, result in enumerate(results):
        reasons: list[str] = []
        if not getattr(result, "trusted", False):
            reasons.append("untrusted")
        reasons.extend(getattr(result, "trust_issues", []) or [])
        if result.error:
            reasons.append(f"backend error: {result.error}")
        audio_id = _identity(result)
        if audio_id is None:
            reasons.append("missing canonical audio_id")
        reference_id = getattr(result, "reference_id", None)
        if not isinstance(reference_id, str) or not reference_id:
            reasons.append("missing reference_id")
        source_path = getattr(result, "source_path", None)
        if not isinstance(source_path, str) or not source_path:
            reasons.append("missing source_path")
        source_sha256 = getattr(result, "source_sha256", None)
        if not isinstance(source_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
            reasons.append("missing or invalid source_sha256")
        reasons.extend(_provenance_issues(result, run))
        if isinstance(result.model_provenance, dict):
            execution_id = result.model_provenance.get("execution_sha256")
            if isinstance(execution_id, str) and execution_id:
                execution_ids.add(execution_id)
        if reasons:
            invalid.append(f"record {index}: {', '.join(reasons)}")
        if audio_id is not None:
            audio_ids.append(audio_id)
        if isinstance(reference_id, str) and reference_id:
            reference_ids.append(reference_id)
    if invalid:
        detail = "; ".join(invalid[:3])
        suffix = f"; and {len(invalid) - 3} more" if len(invalid) > 3 else ""
        raise EvidenceError(f"{run.label}: {detail}{suffix}")
    if len(set(audio_ids)) != len(audio_ids):
        raise EvidenceError(f"{run.label}: duplicate audio_id records")
    if len(set(reference_ids)) != len(reference_ids):
        raise EvidenceError(f"{run.label}: duplicate reference_id records")
    if len(execution_ids) != 1:
        raise EvidenceError(
            f"{run.label}: expected one model execution identity, found {len(execution_ids)}"
        )
    if run.settings:
        try:
            validate_settings((results,), run.settings)
        except (MeasurementError, TypeError, ValueError) as exc:
            raise EvidenceError(f"{run.label}: declared settings differ: {exc}") from exc
    found_ids = set(reference_ids)
    if found_ids != expected_reference_ids:
        missing = sorted(expected_reference_ids - found_ids)
        extra = sorted(found_ids - expected_reference_ids)
        raise EvidenceError(
            f"{run.label}: corpus mismatch (missing={missing[:3]}, extra={extra[:3]})"
        )
    return results


def _validate_cross_run_identity(
    runs: tuple[RunSpec, ...], results_by_run: list[list[TranscriptionResult]]
) -> None:
    """Require every model to have seen the same waveform for each reference."""
    expected = {result.reference_id: result.audio_id for result in results_by_run[0]}
    for run, results in zip(runs[1:], results_by_run[1:], strict=True):
        found = {result.reference_id: result.audio_id for result in results}
        mismatched = sorted(
            reference_id
            for reference_id, audio_id in expected.items()
            if found.get(reference_id) != audio_id
        )
        if mismatched:
            raise EvidenceError(
                f"{run.label}: audio identity mismatch for reference(s) {mismatched[:3]}"
            )


def _metric_value(
    metric: MetricSpec,
    results: list[TranscriptionResult],
    references: dict[str, str],
    normalization: dict[str, Any] | None = None,
) -> str:
    try:
        options = NormalizeOptions(**(normalization or {}))
    except TypeError as exc:
        raise EvidenceError(f"invalid normalization settings: {exc}") from exc
    score = score_results(results, references, options=options, check_encoding=True)
    if score.n_failed or len(score.scored) != len(results):
        raise EvidenceError(
            f"scoring incomplete: {len(score.scored)}/{len(results)} records scored"
        )
    values: dict[str, float | int | None] = {
        "cer": score.cer,
        "wer": score.wer,
        "rtf": score.rtf,
        "n_total": score.total,
        "n_scored": score.n_scored,
    }
    value = values[metric.name]
    if value is None or (isinstance(value, float) and value != value):
        raise EvidenceError(f"metric {metric.name!r} is unavailable")
    if metric.name.startswith("n_"):
        return str(int(value))
    return f"{float(value):.{metric.digits}f}"


#: A measured-looking figure: a decimal carrying enough places to be a score, or
#: a ratio or percentage. Deliberately loose — the point is to notice numbers,
#: not to judge which ones matter.
_MEASURED_NUMBER_RE = re.compile(r"(?<![\w.])\d+\.\d{3,}(?![\w])|(?<![\w.])\d+(?:\.\d+)?\s*(?:×|%)")


def unowned_measured_numbers(document: str) -> list[tuple[int, str]]:
    """Return measured-looking figures outside every evidence block.

    `_validate_document_ownership` rejects an unowned Markdown *table*, but a
    number in a sentence was never checked — and that is where most of this
    document's figures live. Such a number cannot be regenerated, cannot be
    invalidated when its artifact changes, and `stt evidence --update` cannot
    correct it. It is the same defect as a stale table, spread thinner.

    This reports rather than raises, because the existing backlog has to be
    migrated into derived blocks before it can become a hard gate. The tests
    ratchet on it so the backlog cannot grow in the meantime.
    """
    found: list[tuple[int, str]] = []
    inside = False
    fenced = False
    for line_number, line in enumerate(document.splitlines(), start=1):
        marker = _MARKER_RE.fullmatch(line.strip())
        if marker:
            inside = marker.group(2) == "start"
            continue
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
            continue
        if inside or fenced:
            continue
        found.extend((line_number, match.strip()) for match in _MEASURED_NUMBER_RE.findall(line))
    return found


def _validate_document_ownership(document: str, blocks: tuple[BlockSpec, ...]) -> None:
    """Reject evidence markers and Markdown tables outside the manifest."""
    declared = {block.id for block in blocks}
    counts: dict[tuple[str, str], int] = {}
    active: str | None = None
    fenced = False
    for line_number, line in enumerate(document.splitlines(), start=1):
        match = _MARKER_RE.fullmatch(line.strip())
        if match:
            block_id, kind = match.groups()
            if block_id not in declared:
                raise EvidenceError(f"document contains undeclared evidence block {block_id!r}")
            key = (block_id, kind)
            counts[key] = counts.get(key, 0) + 1
            if kind == "start":
                if active is not None:
                    raise EvidenceError(
                        f"evidence block {block_id!r} starts inside block {active!r}"
                    )
                active = block_id
            elif active != block_id:
                raise EvidenceError(
                    f"evidence block {block_id!r} ends outside its matching start marker"
                )
            else:
                active = None
            continue
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
            continue
        if active is None and not fenced and line.lstrip().startswith("|"):
            raise EvidenceError(f"unowned Markdown table row at document line {line_number}")

    if active is not None:
        raise EvidenceError(f"evidence block {active!r} is missing its end marker")
    for block_id in declared:
        for kind in ("start", "end"):
            if counts.get((block_id, kind), 0) != 1:
                raise EvidenceError(
                    f"document must contain exactly one {kind} marker for evidence block "
                    f"{block_id!r}"
                )


def _render_derived_table(table: DerivedTable) -> str:
    table.validate()
    header = [column.label for column in table.columns]
    rows = [
        [
            str(row[column.name])
            if column.name == table.row_key
            else (
                str(int(row[column.name]))
                if column.digits == 0
                else f"{float(row[column.name]):.{column.digits}f}"
            )
            for column in table.columns
        ]
        for row in table.rows
    ]
    return "\n".join(
        [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---", *["---:" for _ in table.columns[1:]]]) + " |",
            *("| " + " | ".join(row) + " |" for row in rows),
        ]
    )


def _render_block(block: BlockSpec) -> tuple[str, list[str], bool]:
    if block.status_only:
        reason = block.status_reason or UNVERIFIED_MESSAGE
        return reason, [f"{block.id}: status_only legacy block"], False

    issues: list[str] = []
    try:
        if block.source_kind == "baseline-v2":
            assert block.source is not None
            return _render_derived_table(baseline_table(block.source)), issues, True
        references = load_references(block.reference)
        if not references:
            raise EvidenceError(f"empty reference corpus: {block.reference}")
        requirements = Requirements(
            reference_sha256=block.reference_sha256,
            expected_reference_count=block.expected_reference_count,
            settings={},
        )
        if block.reference_sha256 is not None or block.expected_reference_count is not None:
            from stt.derive import validate_reference_identity

            validate_reference_identity(
                references,
                path=block.reference,
                requirements=requirements,
            )
        results_by_run = [_validate_run(run, set(references)) for run in block.runs]
        _validate_cross_run_identity(block.runs, results_by_run)
        if block.deriver:
            context = DeriveContext(
                tuple(
                    (run.label, tuple(results))
                    for run, results in zip(block.runs, results_by_run, strict=True)
                ),
                references,
                Requirements(settings={}, aligned_segments=block.aligned_segments),
            )
            table = run_deriver(context, block.deriver, reference_path=block.reference)
            return _render_derived_table(table), issues, True
        header = ["Model", "Backend", *(metric.label for metric in block.metrics)]
        rows = [
            [
                run.label,
                f"`{run.backend}`",
                *(
                    _metric_value(metric, results, references, block.normalization)
                    for metric in block.metrics
                ),
            ]
            for run, results in zip(block.runs, results_by_run, strict=True)
        ]
        table = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---", "---", *["---:" for _ in block.metrics]]) + " |",
            *("| " + " | ".join(row) + " |" for row in rows),
        ]
        return "\n".join(table), issues, True
    except (EvidenceError, MeasurementError, OSError) as exc:
        issues.append(f"{block.id}: {exc}")
        return UNVERIFIED_MESSAGE, issues, False


def _replace_block(document: str, block_id: str, content: str) -> str:
    start = f"<!-- {MARKER_PREFIX}:{block_id}:start -->"
    end = f"<!-- {MARKER_PREFIX}:{block_id}:end -->"
    if document.count(start) != 1 or document.count(end) != 1:
        raise EvidenceError(
            f"document must contain exactly one start/end marker for evidence block {block_id!r}"
        )
    before, remainder = document.split(start, 1)
    _old, after = remainder.split(end, 1)
    return f"{before}{start}\n{content}\n{end}{after}"


def render_evidence(manifest: EvidenceManifest) -> tuple[str, list[str], bool]:
    """Return the complete expected Markdown document without writing it."""
    try:
        document = manifest.document.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvidenceError(f"cannot read evidence document {manifest.document}: {exc}") from exc
    _validate_document_ownership(document, manifest.blocks)
    issues: list[str] = []
    publishable = True
    for block in manifest.blocks:
        rendered, block_issues, block_publishable = _render_block(block)
        document = _replace_block(document, block.id, rendered)
        issues.extend(block_issues)
        publishable = publishable and block_publishable
    return document, issues, publishable


def check_evidence(manifest_path: Path) -> EvidenceCheck:
    """Compare generated evidence blocks with their checked-in Markdown."""
    manifest = load_manifest(manifest_path)
    expected, issues, publishable = render_evidence(manifest)
    current = manifest.document.read_text(encoding="utf-8")
    return EvidenceCheck(
        document=manifest.document,
        matches=current == expected,
        publishable=publishable,
        issues=issues,
        expected_text=expected,
    )


def update_evidence(manifest_path: Path) -> EvidenceCheck:
    """Rewrite only manifest-owned Markdown blocks and return their status."""
    manifest = load_manifest(manifest_path)
    expected, issues, publishable = render_evidence(manifest)
    manifest.document.write_text(expected, encoding="utf-8")
    return EvidenceCheck(
        document=manifest.document,
        matches=True,
        publishable=publishable,
        issues=issues,
        expected_text=expected,
    )
