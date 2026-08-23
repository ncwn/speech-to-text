"""Typed evidence derivers fail closed on identity and setting drift."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from stt.derive import DeriveContext, Requirements, derive, reference_sha256
from stt.measurement import MeasurementError
from stt.provenance import ArtifactDigest, ModelProvenance
from stt.results import Segment, TranscriptionResult


def _result(reference: str, digest: str = "a" * 64) -> TranscriptionResult:
    provenance = ModelProvenance(
        backend="fake",
        requested_model="model",
        source_kind="test",
        source_locator="test/model",
        upstream_revision="b" * 40,
        revision_status="pinned",
        artifacts=(ArtifactDigest("weights", "model.bin", 1, "c" * 64),),
        runtime_packages={"runtime": "1"},
        resolved_settings={"device": "cpu", "dtype": "float32"},
        adapter_git_commit="d" * 40,
        uv_lock_sha256="e" * 64,
    ).finalized()
    return TranscriptionResult(
        audio_path=f"prepared/{reference}.wav",
        source_path=f"source/{reference}.wav",
        source_sha256="f" * 64,
        audio_id=f"pcm16:16000:1:{digest}",
        reference_id=reference,
        text="stable",
        backend="fake",
        model="model",
        trusted=True,
        model_provenance=provenance.to_dict(),
    )


def _context(tmp_path: Path) -> tuple[DeriveContext, Path]:
    reference = tmp_path / "refs.tsv"
    reference.write_text("audio_id\ttranscript\nclip\tstable\n", encoding="utf-8")
    requirements = Requirements(
        reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        expected_reference_count=1,
        settings={"device": "cpu", "dtype": "float32"},
    )
    context = DeriveContext((("fake", (_result("clip"),)),), {"clip": "stable"}, requirements)
    return context, reference


def test_transcript_deriver_returns_typed_rows_and_recomputes_identity(tmp_path):
    context, reference = _context(tmp_path)

    table = derive(context, "transcripts:v1", reference_path=reference)

    assert table.to_dict()["rows"] == [{"run": "fake", "n_records": 1}]
    assert reference_sha256(reference) == context.requirements.reference_sha256


def test_deriver_rejects_waveform_map_drift(tmp_path):
    context, reference = _context(tmp_path)
    changed = replace(_result("clip"), audio_id="pcm16:16000:1:" + "b" * 64)
    invalid = replace(context, runs=(("fake", (context.runs[0][1][0],)), ("other", (changed,))))

    with pytest.raises(MeasurementError, match="same reference-to-waveform map"):
        derive(invalid, "transcripts:v1", reference_path=reference)


def test_deriver_rejects_declared_setting_drift(tmp_path):
    context, reference = _context(tmp_path)
    requirements = replace(context.requirements, settings={"device": "mps"})
    invalid = replace(context, requirements=requirements)

    with pytest.raises(MeasurementError, match="declared setting"):
        derive(invalid, "transcripts:v1", reference_path=reference)


def test_confidence_deriver_requires_aligned_segments_and_allows_silence_gaps(tmp_path):
    context, reference = _context(tmp_path)
    result = replace(
        _result("clip"),
        audio_duration_s=2.0,
        segments=[
            Segment("a", 0.0, 0.5, confidence=0.25, source="aligned"),
            Segment("b", 1.0, 1.5, confidence=0.75, source="aligned"),
        ],
    )
    context = replace(
        context,
        runs=(("fake", (result,)),),
        requirements=replace(context.requirements, aligned_segments=True),
    )

    table = derive(context, "confidence:v1", reference_path=reference)

    assert table.rows[0]["p50"] == pytest.approx(0.5)


def test_vote_and_route_derivers_reuse_production_joins(tmp_path):
    context, reference = _context(tmp_path)
    base = replace(
        _result("clip"),
        audio_duration_s=1.0,
        segments=[Segment("a", 0.0, 1.0, confidence=0.5, source="aligned")],
    )
    strong = replace(base, model="strong")
    paired = replace(context, runs=(("base", (base,)), ("strong", (strong,))))
    requirements = replace(paired.requirements, aligned_segments=True)
    paired = replace(paired, requirements=requirements)

    vote = derive(paired, "vote:v1", reference_path=reference)
    route = derive(paired, "route:v1", reference_path=reference)

    assert vote.rows[0]["n_voters"] == 2
    assert route.rows[0]["duration_s"] == 1.0
