"""CLI-level guards for evidence identity, coverage, and failure handling."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager

import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

import stt.cli as cli_mod
from stt.audio import PreparedAudio, prepare_audio
from stt.cli import DEFAULT_CACHE, _run_backend, app
from stt.results import Segment, TranscriptionResult, read_jsonl, write_jsonl
from stt.telemetry import ResourceUsage

runner = CliRunner()


def _audio_id(label: str) -> str:
    return f"pcm16:16000:1:{hashlib.sha256(label.encode()).hexdigest()}"


def _result(
    label: str,
    *,
    model: str,
    text: str = "စာ",
    error: str | None = None,
    path: str | None = None,
) -> TranscriptionResult:
    return TranscriptionResult(
        audio_path=path or f"{label}.wav",
        source_path=f"source/{label}.wav",
        source_sha256=hashlib.sha256(f"source:{label}".encode()).hexdigest(),
        reference_id=label,
        audio_id=_audio_id(label),
        text=text,
        backend="test",
        model=model,
        elapsed_s=1.0,
        audio_duration_s=2.0,
        error=error,
        trusted=error is None,
    )


def test_eval_suppresses_legacy_metrics_unless_partial_is_explicit(tmp_path):
    run = tmp_path / "legacy.jsonl"
    refs = tmp_path / "refs.tsv"
    write_jsonl(
        [
            TranscriptionResult(
                audio_path="clip.wav",
                text="စာ",
                backend="test",
                model="legacy",
                elapsed_s=1.0,
                audio_duration_s=2.0,
            )
        ],
        run,
    )
    refs.write_text("clip\tစာ\n", encoding="utf-8")

    strict = runner.invoke(app, ["eval", str(run), "--reference", str(refs)])
    partial = runner.invoke(
        app,
        ["eval", str(run), "--reference", str(refs), "--allow-partial"],
    )

    assert strict.exit_code == 1
    assert "Trusted metrics suppressed" in strict.output
    assert partial.exit_code == 0, partial.output
    assert "partial diagnostic" in partial.output
    assert "0.0000" in partial.output


def test_vote_omits_failed_nonpivot_only_in_explicit_partial_mode(tmp_path):
    pivot_path, failed_path, out = (
        tmp_path / "pivot.jsonl",
        tmp_path / "failed.jsonl",
        tmp_path / "vote.jsonl",
    )
    write_jsonl([_result("clip", model="pivot", text="အကျိုးရှိ")], pivot_path)
    write_jsonl([_result("clip", model="small", text="", error="decoder failed")], failed_path)

    strict = runner.invoke(app, ["vote", str(pivot_path), str(failed_path), "-o", str(out)])
    assert strict.exit_code != 0
    assert not out.exists()

    partial = runner.invoke(
        app,
        ["vote", str(pivot_path), str(failed_path), "-o", str(out), "--allow-partial"],
    )
    assert partial.exit_code == 0, partial.output
    (combined,) = read_jsonl(out)
    assert combined.text == "အကျိုးရှိ"
    assert combined.metadata["used_voters"] == ["pivot"]
    assert combined.metadata["failed_voters"] == ["small"]
    assert not combined.trusted


def test_vote_joins_same_audio_id_across_different_runtime_paths(tmp_path):
    first, second, out = tmp_path / "a.jsonl", tmp_path / "b.jsonl", tmp_path / "out.jsonl"
    write_jsonl([_result("clip", model="pivot", path="cache/clip.wav")], first)
    write_jsonl([_result("clip", model="other", path="original/clip.flac")], second)

    result = runner.invoke(app, ["vote", str(first), str(second), "-o", str(out)])

    assert result.exit_code == 0, result.output
    (combined,) = read_jsonl(out)
    assert combined.audio_id == _audio_id("clip")
    assert not combined.trusted
    assert combined.model_provenance is None
    assert "derived vote lacks complete model provenance" in combined.trust_issues
    assert combined.metadata["used_voters"] == ["other", "pivot"]


def test_transcribe_writes_failed_diagnostic_but_exits_nonzero(monkeypatch, tmp_path):
    source = tmp_path / "clip.wav"
    sf.write(source, np.zeros(1600), 16_000, subtype="PCM_16")
    out = tmp_path / "run.jsonl"
    prepared = PreparedAudio(source, source, "clip", "0" * 64, _audio_id("clip"))
    monkeypatch.setattr(cli_mod, "_prepare", lambda *args, **kwargs: [prepared])
    monkeypatch.setattr(
        cli_mod,
        "_run_backend",
        lambda *args, **kwargs: [_result("clip", model="broken", text="", error="boom")],
    )

    result = runner.invoke(
        app,
        ["transcribe", str(source), "--output", str(out), "--no-show"],
    )

    assert result.exit_code == 1
    assert out.exists()
    assert not read_jsonl(out)[0].trusted


def test_run_backend_keeps_identity_and_resources_for_all_failed_batch(monkeypatch, tmp_path):
    source = tmp_path / "clip.wav"
    sf.write(source, np.zeros(1600), 16_000, subtype="PCM_16")
    prepared = PreparedAudio(source, source, "clip", "a" * 64, _audio_id("clip"))

    class Backend:
        install_hint = ""

        @classmethod
        def is_available(cls):
            return True, "test"

        def __init__(self, model=None, **kwargs):
            self.model = model or "fake"
            self.resolved_device = "cpu"

        def load(self):
            return None

        def unload(self):
            return None

        def transcribe(self, paths, language=None, batch_size=1):
            return [
                TranscriptionResult(
                    audio_path=str(path),
                    text="",
                    backend="fake",
                    model=self.model,
                    error="boom",
                )
                for path in paths
            ]

    @contextmanager
    def measured(**kwargs):
        holder: list[ResourceUsage] = []
        try:
            yield holder
        finally:
            holder.append(ResourceUsage(wall_s=0.5, cpu_s=0.25, peak_rss_mb=10.0))

    monkeypatch.setattr(cli_mod, "get_backend", lambda name: Backend)
    monkeypatch.setattr(cli_mod, "measure", measured)
    monkeypatch.setattr(cli_mod, "describe_host", lambda: {"chip": "test"})

    (result,) = _run_backend("fake", None, [prepared], None, 1, {})

    assert result.audio_id == prepared.audio_id
    assert result.reference_id == "clip"
    assert result.resources is not None
    assert result.metadata["resource_scope"] == "corpus"
    assert not result.trusted


def test_run_backend_retains_resources_when_backend_batch_raises(monkeypatch, tmp_path):
    source = tmp_path / "clip.wav"
    sf.write(source, np.zeros(1600), 16_000, subtype="PCM_16")
    prepared = PreparedAudio(source, source, "clip", "a" * 64, _audio_id("clip"))

    class Backend:
        install_hint = ""

        @classmethod
        def is_available(cls):
            return True, "test"

        def __init__(self, model=None, **kwargs):
            self.model = model or "fake"
            self.resolved_device = "cpu"

        def load(self):
            return None

        def unload(self):
            return None

        def transcribe(self, paths, language=None, batch_size=1):
            raise RuntimeError("batch exploded")

    @contextmanager
    def measured(**kwargs):
        holder: list[ResourceUsage] = []
        try:
            yield holder
        finally:
            holder.append(ResourceUsage(wall_s=0.5, cpu_s=0.25, peak_rss_mb=10.0))

    monkeypatch.setattr(cli_mod, "get_backend", lambda name: Backend)
    monkeypatch.setattr(cli_mod, "measure", measured)
    monkeypatch.setattr(cli_mod, "describe_host", lambda: {"chip": "test"})

    (result,) = _run_backend("fake", None, [prepared], None, 1, {})

    assert result.error == "RuntimeError: batch exploded"
    assert result.resources is not None
    assert result.metadata["corpus_exception"] is True
    assert result.metadata["resource_scope"] == "corpus"
    assert not result.trusted


def test_run_backend_validates_bound_model_before_load_and_after_execution(monkeypatch):
    events: list[str] = []
    binding = object()
    result = _result("clip", model="fake")

    class Backend:
        install_hint = ""

        @classmethod
        def is_available(cls):
            return True, "test"

        def __init__(self, model=None, **kwargs):
            self.model = model or "fake"

        def bind_model(self, value):
            assert value is binding
            events.append("bind")

        def load(self):
            events.append("load")

        def unload(self):
            events.append("unload")

    @contextmanager
    def measured(**kwargs):
        holder: list[ResourceUsage] = []
        try:
            yield holder
        finally:
            holder.append(ResourceUsage(wall_s=0.5, cpu_s=0.25, peak_rss_mb=10.0))

    def validate(value):
        assert value is binding
        events.append("validate")
        return ()

    def transcribe_corpus(*args, **kwargs):
        events.append("transcribe")
        return [result], ResourceUsage(wall_s=0.5, cpu_s=0.25, peak_rss_mb=10.0)

    monkeypatch.setattr(cli_mod, "get_backend", lambda name: Backend)
    monkeypatch.setattr(cli_mod, "preflight_model_binding", lambda *args, **kwargs: binding)
    monkeypatch.setattr(cli_mod, "validate_binding", validate)
    monkeypatch.setattr(cli_mod, "transcribe_corpus", transcribe_corpus)
    monkeypatch.setattr(cli_mod, "measure", measured)
    monkeypatch.setattr(cli_mod, "describe_host", lambda: {"chip": "test"})

    results = _run_backend("fake", None, [], None, 1, {})

    assert results == [result]
    assert events == [
        "bind",
        "validate",
        "load",
        "validate",
        "transcribe",
        "unload",
        "validate",
    ]


def test_run_backend_rejects_invalid_bound_model_before_load(monkeypatch):
    events: list[str] = []
    binding = object()

    class Backend:
        install_hint = ""

        @classmethod
        def is_available(cls):
            return True, "test"

        def __init__(self, model=None, **kwargs):
            self.model = model or "fake"

        def bind_model(self, value):
            assert value is binding
            events.append("bind")

        def load(self):
            events.append("load")

    monkeypatch.setattr(cli_mod, "get_backend", lambda name: Backend)
    monkeypatch.setattr(cli_mod, "preflight_model_binding", lambda *args, **kwargs: binding)
    monkeypatch.setattr(
        cli_mod,
        "validate_binding",
        lambda value: ("artifact digest changed: weights",),
    )

    with pytest.raises(RuntimeError, match="model binding validation failed before load"):
        _run_backend("fake", None, [], None, 1, {})

    assert events == ["bind"]


def test_align_results_preserves_transcription_lineage(monkeypatch, tmp_path):
    audio = tmp_path / "clip.wav"
    sf.write(audio, np.zeros(1600, dtype=np.float32), 16_000, subtype="PCM_16")
    prepared = prepare_audio(audio, DEFAULT_CACHE)
    source = _result("clip", model="original-model", path=str(prepared.prepared_path))
    source.audio_id = prepared.audio_id
    source.source_path = str(prepared.source_path)
    source.source_sha256 = prepared.source_sha256
    source.elapsed_s = 3.25
    run = tmp_path / "source.jsonl"
    write_jsonl([source], run)

    monkeypatch.setattr(
        "stt.align.align",
        lambda *args, **kwargs: [Segment("စာ", 0.0, 0.1, confidence=0.9, source="aligned")],
    )
    from stt.align import LoadedAligner

    monkeypatch.setattr(
        "stt.align.load_aligner",
        lambda device: LoadedAligner(
            processor=object(),
            model=object(),
            device=device,
            provenance_issues=("test aligner provenance unavailable",),
        ),
    )
    stem = tmp_path / "aligned"
    result = runner.invoke(
        app,
        ["align", str(audio), "--results", str(run), "--output", str(stem), "--no-srt"],
    )

    assert result.exit_code == 0, result.output
    (aligned,) = read_jsonl(stem.with_suffix(".jsonl"))
    assert aligned.backend == "test"
    assert aligned.model == "original-model"
    assert aligned.elapsed_s == 3.25
    assert aligned.audio_id == prepared.audio_id
    assert aligned.metadata["alignment"]["model"] == "mms-1b-all"
    assert aligned.metadata["alignment"]["backend"] == "hf"
    assert not aligned.metadata["alignment"]["trusted"]
    assert not aligned.trusted


def test_align_results_does_not_match_legacy_same_stem_from_another_path(tmp_path):
    audio = tmp_path / "target" / "clip.wav"
    audio.parent.mkdir()
    sf.write(audio, np.zeros(1600, dtype=np.float32), 16_000, subtype="PCM_16")
    wrong = _result("wrong", model="legacy", path=str(tmp_path / "other" / "clip.wav"))
    wrong.audio_id = None
    run = tmp_path / "legacy.jsonl"
    write_jsonl([wrong], run)

    result = runner.invoke(
        app,
        ["align", str(audio), "--results", str(run), "--no-srt"],
    )

    assert result.exit_code != 0
    assert "No result" in result.output


def test_align_results_rejects_duplicate_verified_audio_ids(monkeypatch, tmp_path):
    audio = tmp_path / "clip.wav"
    sf.write(audio, np.zeros(1600, dtype=np.float32), 16_000, subtype="PCM_16")
    prepared = prepare_audio(audio, DEFAULT_CACHE)
    first = _result("first", model="legacy-a", text="first", path=str(audio))
    second = _result("second", model="legacy-b", text="second", path=str(audio))
    first.audio_id = prepared.audio_id
    second.audio_id = prepared.audio_id
    run = tmp_path / "duplicate.jsonl"
    write_jsonl([first, second], run)
    loader_calls: list[str] = []
    monkeypatch.setattr(
        "stt.align.load_aligner",
        lambda device: loader_calls.append(device),
    )

    result = runner.invoke(
        app,
        ["align", str(audio), "--results", str(run), "--no-srt"],
    )

    assert result.exit_code != 0
    assert "Ambiguous results" in result.output
    assert loader_calls == []


def test_align_results_rejects_ambiguous_normalized_legacy_paths(tmp_path):
    audio = tmp_path / "clip.wav"
    sf.write(audio, np.zeros(1600, dtype=np.float32), 16_000, subtype="PCM_16")
    first = _result("first", model="legacy-a", path=str(audio))
    second = _result(
        "second",
        model="legacy-b",
        path=str(audio.parent / "nested" / ".." / audio.name),
    )
    first.audio_id = None
    second.audio_id = None
    run = tmp_path / "legacy.jsonl"
    write_jsonl([first, second], run)

    result = runner.invoke(
        app,
        ["align", str(audio), "--results", str(run), "--no-srt"],
    )

    assert result.exit_code != 0
    assert "Ambiguous legacy results" in result.output


def test_inline_alignment_records_provenance_and_downgrades_untrusted_result(monkeypatch):
    from stt.align import LoadedAligner

    result = _result("clip", model="decoder")
    monkeypatch.setattr(
        "stt.align.load_aligner",
        lambda device: LoadedAligner(
            processor=object(),
            model=object(),
            device=device,
            provenance_issues=("missing immutable aligner binding",),
        ),
    )
    monkeypatch.setattr(
        "stt.align.align",
        lambda *args, **kwargs: [Segment("စာ", 0.0, 0.1, confidence=0.9, source="aligned")],
    )

    cli_mod._add_alignment([result])

    assert result.segments is not None
    assert result.metadata["alignment"]["backend"] == "hf"
    assert result.metadata["alignment"]["status"] == "completed"
    assert not result.metadata["alignment"]["trusted"]
    assert not result.trusted
    assert "missing immutable aligner binding" in result.trust_issues


def test_align_text_serializes_the_same_alignment_metadata(monkeypatch, tmp_path):
    from stt.align import LoadedAligner

    audio = tmp_path / "clip.wav"
    transcript = tmp_path / "clip.txt"
    sf.write(audio, np.zeros(1600, dtype=np.float32), 16_000, subtype="PCM_16")
    transcript.write_text("စာ", encoding="utf-8")
    monkeypatch.setattr(
        "stt.align.load_aligner",
        lambda device: LoadedAligner(
            processor=object(),
            model=object(),
            device=device,
            provenance_issues=("missing immutable aligner binding",),
        ),
    )
    monkeypatch.setattr(
        "stt.align.align",
        lambda *args, **kwargs: [Segment("စာ", 0.0, 0.1, confidence=0.9, source="aligned")],
    )

    stem = tmp_path / "aligned"
    result = runner.invoke(
        app,
        [
            "align",
            str(audio),
            "--text",
            str(transcript),
            "--output",
            str(stem),
            "--no-srt",
        ],
    )

    assert result.exit_code == 0, result.output
    (aligned,) = read_jsonl(stem.with_suffix(".jsonl"))
    assert aligned.backend == "align"
    assert aligned.metadata["alignment"]["backend"] == "hf"
    assert aligned.metadata["alignment"]["status"] == "completed"
    assert not aligned.metadata["alignment"]["trusted"]
    assert not aligned.trusted
    assert aligned.model_provenance is None


def test_bench_rejects_partial_baseline_updates_before_loading_models():
    result = runner.invoke(app, ["bench", "--update", "--only", "hf"])

    assert result.exit_code != 0
    assert "--update cannot be combined with --only" in result.output


def test_bench_rejects_smoke_protocol_for_baseline_update():
    result = runner.invoke(app, ["bench", "--update"])

    assert result.exit_code != 0
    assert "baseline update requires" in result.output


def test_common_corpus_rtf_uses_outer_wall_instead_of_adapter_timings():
    first = _result("first", model="fake")
    second = _result("second", model="fake")
    first.elapsed_s = 100.0
    second.elapsed_s = 100.0
    first.audio_duration_s = 1.0
    second.audio_duration_s = 3.0
    first.resources = ResourceUsage(
        wall_s=2.0,
        cpu_s=1.0,
        peak_rss_mb=10.0,
        rss_peak_mb=8.0,
    )
    first.metadata["resource_scope"] = "corpus"

    assert cli_mod._common_wall_rtf([first, second]) == 0.5

    second.error = "boom"
    assert cli_mod._common_wall_rtf([first, second]) is None
    assert cli_mod._common_wall_rtf([first, second], allow_incomplete=True) == 0.5


@pytest.mark.parametrize(
    ("trusted", "trust_issues"),
    [
        (False, []),
        (True, ["model provenance unavailable"]),
    ],
    ids=["untrusted-flag", "trust-issues"],
)
def test_compare_without_reference_fails_closed_on_untrusted_results(
    monkeypatch, tmp_path, trusted, trust_issues
):
    prepared = PreparedAudio(
        tmp_path / "clip.wav",
        tmp_path / "clip.wav",
        "clip",
        "0" * 64,
        _audio_id("clip"),
    )

    monkeypatch.setattr(cli_mod, "_prepare", lambda *args, **kwargs: [prepared])

    class AvailableBackend:
        @classmethod
        def is_available(cls):
            return True, "test"

    def run_untrusted(*args, **kwargs):
        result = _result("clip", model="fake")
        result.trusted = trusted
        result.trust_issues = list(trust_issues)
        return [result]

    monkeypatch.setattr(cli_mod, "all_backends", lambda: {"fake": AvailableBackend})
    monkeypatch.setattr(cli_mod, "_run_backend", run_untrusted)
    output_dir = tmp_path / "outputs"

    strict = runner.invoke(
        app,
        [
            "compare",
            str(prepared.source_path),
            "--only",
            "fake",
            "--yes",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert strict.exit_code == 1, strict.output
    assert "untrusted" in strict.output
    # Keyed on backend and model, so two models behind one backend
    # cannot overwrite each other.
    (strict_result,) = read_jsonl(output_dir / "fake-fake.jsonl")
    assert not strict_result.trusted

    partial = runner.invoke(
        app,
        [
            "compare",
            str(prepared.source_path),
            "--only",
            "fake",
            "--yes",
            "--allow-partial",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert partial.exit_code == 0, partial.output
    assert "partial diagnostic artifacts" in partial.output
    (partial_result,) = read_jsonl(output_dir / "fake-fake.jsonl")
    assert not partial_result.trusted


def test_align_refuses_to_overwrite_a_transcription_run(tmp_path):
    """The exact defect that turned a 7B run into an alignment record.

    `stt align -o` writes backend="align", model="mms-1b-all". Pointed at a
    name a transcription run owns, it replaced that run's identity in place:
    the transcript survived but the backend, model and timings that said where
    it came from did not, and nothing reported it.
    """
    from stt.cli import _refuse_to_clobber_another_run

    path = tmp_path / "eternity.jsonl"
    write_jsonl([_result("clip", model="omniASR_LLM_Unlimited_7B_v2")], path)
    aligned = TranscriptionResult(
        audio_path="clip.wav", text="စာ", backend="align", model="mms-1b-all"
    )

    with pytest.raises(Exception) as caught:
        _refuse_to_clobber_another_run(path, aligned)
    assert "already holds a run from" in str(caught.value)

    # Rewriting a run with its own identity is ordinary, and a new name is fine.
    _refuse_to_clobber_another_run(path, _result("clip", model="omniASR_LLM_Unlimited_7B_v2"))
    _refuse_to_clobber_another_run(tmp_path / "fresh.jsonl", aligned)
