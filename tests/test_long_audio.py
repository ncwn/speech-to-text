"""Long-audio runner coverage is explicit about gaps, duplicates, and order."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stt.long_audio import (
    LongAudioRunnerSpec,
    observe_segments,
    read_sentinel_spans,
    run_runner,
    run_runner_report,
    sentinel_identity,
    verify_observation,
    write_report,
)
from stt.results import Segment, TranscriptionResult
from stt.sentinel import annotation


def _result(segments):
    return TranscriptionResult(
        audio_path="sentinel.wav",
        text=" ".join(segment.text for segment in segments),
        backend="fake",
        model="model",
        audio_duration_s=45.0,
        segments=list(segments),
    )


def test_sentinel_spans_and_runner_identity_are_read_from_checked_in_data():
    root = Path(__file__).parents[1]
    annotation_path = root / "data" / "sentinels" / "long-audio-boundary-v1.json"
    spans = read_sentinel_spans(annotation_path)
    audio_sha, annotation_sha = sentinel_identity(annotation_path, annotation_path)
    spec = LongAudioRunnerSpec(
        "adapter", "fake", "model", "adapter.transcribe", audio_sha, annotation_sha
    )

    spec.validate()
    assert len(spans) == 6
    assert annotation()["artifact_kind"] == "long-audio-sentinel-v1"


def test_observation_counts_silence_gaps_and_boundary_spans():
    expected = ((18.75, 19.75), (19.75, 20.75), (20.75, 21.75))
    result = _result(
        [
            Segment("before", 18.75, 19.75, source="chunk"),
            Segment("across", 19.75, 20.75, source="chunk"),
            Segment("after", 20.75, 21.75, source="chunk"),
        ]
    )

    observation = observe_segments("adapter", result, expected, elapsed_s=0.5)

    assert observation.matched_spans == 3
    assert observation.missed_spans == 0
    assert observation.duplicate_spans == 0
    assert observation.ordered
    assert observation.boundary_hits == (20.0,)


def test_runner_captures_failures_as_observations():
    spec = LongAudioRunnerSpec("native", "fake", "model", "native.transcribe", "a" * 64, "b" * 64)

    observation = run_runner(spec, lambda: (_ for _ in ()).throw(RuntimeError("boom")), ())

    assert observation.error == "RuntimeError: boom"
    assert observation.n_segments == 0


def test_archived_sentinel_observations_bind_the_fixture_identities():
    root = Path(__file__).parents[1]
    audio = root / "data" / "sentinels" / "long-audio-boundary-v1.wav"
    annotation_path = root / "data" / "sentinels" / "long-audio-boundary-v1.json"
    reports = sorted((root / "evidence" / "experiments" / "long-audio-sentinel").glob("*.json"))

    assert reports
    assert all(
        any(
            "legacy" in issue
            for issue in verify_observation(
                report, audio_path=audio, annotation_path=annotation_path
            )
        )
        for report in reports
    )


def _write_v2_report(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = Path(__file__).parents[1]
    audio = root / "data" / "sentinels" / "long-audio-boundary-v1.wav"
    annotation_path = root / "data" / "sentinels" / "long-audio-boundary-v1.json"
    audio_sha, annotation_sha = sentinel_identity(annotation_path, audio)
    spec = LongAudioRunnerSpec(
        "adapter",
        "fake",
        "model",
        "adapter.transcribe",
        audio_sha,
        annotation_sha,
    )
    report = run_runner_report(
        spec,
        lambda: _result(
            [
                Segment("before", 18.75, 19.75, source="chunk"),
                Segment("across", 19.75, 20.75, source="chunk"),
                Segment("after", 20.75, 21.75, source="chunk"),
            ]
        ),
        read_sentinel_spans(annotation_path),
    )
    report_path = tmp_path / "report.json"
    write_report(report, report_path)
    return report_path, audio, annotation_path


def test_v2_report_recomputes_from_raw_result_and_binds_fixture(tmp_path):
    report_path, audio, annotation_path = _write_v2_report(tmp_path)

    assert verify_observation(report_path, audio_path=audio, annotation_path=annotation_path) == []


def test_v2_report_makes_checkout_paths_portable(tmp_path):
    root = Path(__file__).parents[1]
    audio = root / "data" / "sentinels" / "long-audio-boundary-v1.wav"
    annotation_path = root / "data" / "sentinels" / "long-audio-boundary-v1.json"
    audio_sha, annotation_sha = sentinel_identity(annotation_path, audio)
    spec = LongAudioRunnerSpec(
        "adapter", "fake", "model", "adapter.transcribe", audio_sha, annotation_sha
    )
    result = _result([Segment("all", 0.0, 45.0, source="chunk")])
    result.audio_path = str(audio)
    result.source_path = str(audio)
    result.model_provenance = {"artifacts": [{"path": str(root / ".cache" / "weights.bin")}]}
    report = run_runner_report(
        spec,
        lambda: result,
        read_sentinel_spans(annotation_path),
    )
    report_path = tmp_path / "portable.json"

    write_report(report, report_path)

    value = json.loads(report_path.read_text(encoding="utf-8"))
    assert value["result"]["audio_path"] == "data/sentinels/long-audio-boundary-v1.wav"
    assert value["result"]["source_path"] == "data/sentinels/long-audio-boundary-v1.wav"
    assert value["result"]["model_provenance"]["artifacts"][0]["path"] == (".cache/weights.bin")


@pytest.mark.parametrize(
    ("location", "mutate"),
    [
        ("coverage", lambda value: value["observation"].update(matched_spans=99)),
        ("boundary", lambda value: value["observation"].update(boundary_hits=[])),
        (
            "ordering",
            lambda value: value["result"]["segments"].reverse(),
        ),
        ("transcript", lambda value: value["result"].update(text="edited")),
        ("error", lambda value: value["result"].update(error="edited")),
        (
            "segment",
            lambda value: value["result"]["segments"][0].update(start=18.0),
        ),
    ],
)
def test_v2_report_rejects_tampered_raw_or_derived_data(tmp_path, location, mutate):
    report_path, audio, annotation_path = _write_v2_report(tmp_path)
    value = json.loads(report_path.read_text(encoding="utf-8"))
    mutate(value)
    report_path.write_text(json.dumps(value), encoding="utf-8")

    issues = verify_observation(report_path, audio_path=audio, annotation_path=annotation_path)

    assert issues, location


def test_v2_report_rejects_tampered_boundaries_and_elapsed(tmp_path):
    report_path, audio, annotation_path = _write_v2_report(tmp_path)
    value = json.loads(report_path.read_text(encoding="utf-8"))
    value["spec"]["boundaries_s"] = [19.0, 40.0]
    value["elapsed_s"] += 1.0
    report_path.write_text(json.dumps(value), encoding="utf-8")

    issues = verify_observation(report_path, audio_path=audio, annotation_path=annotation_path)

    assert issues
