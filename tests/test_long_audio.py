"""Long-audio runner coverage is explicit about gaps, duplicates, and order."""

from __future__ import annotations

from pathlib import Path

from stt.long_audio import (
    LongAudioRunnerSpec,
    observe_segments,
    read_sentinel_spans,
    run_runner,
    sentinel_identity,
    verify_observation,
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
        verify_observation(report, audio_path=audio, annotation_path=annotation_path) == []
        for report in reports
    )
