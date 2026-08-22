"""The common execution boundary owns the complete corpus."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from stt.audio import prepare_audio
from stt.backends.base import ASRBackend
from stt.execution import transcribe_corpus
from stt.provenance import ArtifactDigest, ModelBinding, ModelProvenance
from stt.results import Segment, TranscriptionResult, write_jsonl


class FakeBackend:
    model = "fake-model"
    resolved_device = "cpu"

    def __init__(self) -> None:
        self.calls: list[tuple[list[Path], str | None, int]] = []

    def transcribe(self, paths, language=None, batch_size=1):
        self.calls.append((list(paths), language, batch_size))
        return [TranscriptionResult(str(path), path.stem, "fake", self.model) for path in paths]


def _complete_provenance(*, backend: str = "fake", model: str = "fake-model") -> ModelProvenance:
    return ModelProvenance(
        backend=backend,
        requested_model=model,
        source_kind="test",
        source_locator="test/fake-model",
        upstream_revision="a" * 40,
        revision_status="pinned",
        artifacts=(ArtifactDigest("weights", "model.bin", 1, "1" * 64, path="/tmp/model.bin"),),
        runtime_packages={"runtime": "1"},
        requested_settings={"language": "mya_Mymr", "batch_size": 1},
        resolved_settings={"device": "cpu", "dtype": "float32"},
        adapter_git_commit="b" * 40,
        uv_lock_sha256="2" * 64,
    ).finalized()


class BoundBackend(ASRBackend):
    name = "fake"

    def __init__(self, **options):
        super().__init__("fake-model", **options)
        self.resolved_device = "mps"
        self.resolved_dtype = "float32"

    def load(self):
        self._loaded = True

    def transcribe(self, audio_paths, language=None, batch_size=1):
        return [
            TranscriptionResult(str(path), "text", self.name, self.model) for path in audio_paths
        ]


class PostCallProvenanceFailure(FakeBackend):
    def __init__(self):
        super().__init__()
        self.provenance_calls = 0

    def model_provenance(self):
        self.provenance_calls += 1
        if self.provenance_calls > 1:
            raise RuntimeError("post-call capture failed")
        return _complete_provenance()


def _inputs(tmp_path, count=2):
    prepared = []
    for index in range(count):
        path = tmp_path / f"clip-{index}.wav"
        sf.write(path, np.zeros(1600), 16_000, subtype="PCM_16")
        prepared.append(prepare_audio(path, tmp_path / "cache"))
    return prepared


def test_complete_corpus_is_passed_to_backend_once(tmp_path):
    backend = FakeBackend()
    inputs = _inputs(tmp_path)

    results, usage = transcribe_corpus(
        backend, "fake", inputs, "mya_Mymr", 2, profile=False, host={"chip": "test"}
    )

    assert len(backend.calls) == 1
    assert backend.calls[0] == ([item.prepared_path for item in inputs], "mya_Mymr", 2)
    assert all(result.trusted for result in results)
    assert all(result.audio_duration_s == pytest.approx(0.1) for result in results)
    assert results[0].resources is usage
    assert results[0].metadata["resource_scope"] == "corpus"
    assert results[1].resources is None
    assert usage.rss_peak_mb is not None


def test_all_failed_corpus_keeps_common_wall_and_identity(tmp_path):
    class Broken(FakeBackend):
        def transcribe(self, paths, language=None, batch_size=1):
            raise RuntimeError("boom")

    inputs = _inputs(tmp_path, count=1)
    results, usage = transcribe_corpus(
        Broken(), "fake", inputs, None, 1, profile=False, host={"chip": "test"}
    )

    assert results[0].error == "RuntimeError: boom"
    assert results[0].audio_id == inputs[0].audio_id
    assert results[0].resources is usage
    assert results[0].metadata["corpus_exception"] is True
    assert not results[0].trusted


def test_out_of_order_results_fail_the_corpus_contract(tmp_path):
    class Reversed(FakeBackend):
        def transcribe(self, paths, language=None, batch_size=1):
            return [
                TranscriptionResult(str(path), "x", "fake", self.model) for path in reversed(paths)
            ]

    with pytest.raises(RuntimeError, match="out of order"):
        transcribe_corpus(
            Reversed(),
            "fake",
            _inputs(tmp_path),
            None,
            2,
            profile=False,
            host={"chip": "test"},
        )


def test_contract_violations_are_diagnostic_per_item(tmp_path):
    class Violating(FakeBackend):
        def transcribe(self, paths, language=None, batch_size=1):
            results = super().transcribe(paths, language, batch_size)
            results[0].backend = "other-backend"
            results[0].model = "other-model"
            results[0].text = ""
            return results

    results, _ = transcribe_corpus(
        Violating(),
        "fake",
        _inputs(tmp_path),
        None,
        2,
        profile=False,
        host={"chip": "test"},
    )

    bad, good = results
    assert bad.error is not None
    assert any("backend identity mismatch" in issue for issue in bad.trust_issues)
    assert any("model identity mismatch" in issue for issue in bad.trust_issues)
    assert bad.metadata["contract_violations"]
    assert good.error is None
    assert good.text
    assert not bad.trusted
    assert not good.trusted  # corpus trust is all-or-nothing, but the item survives


def test_contract_validates_segment_order_and_audio_bounds(tmp_path):
    class BrokenSegments(FakeBackend):
        def transcribe(self, paths, language=None, batch_size=1):
            results = super().transcribe(paths, language, batch_size)
            results[0].segments = [
                Segment("late", 0.2, 0.3),
                Segment("backwards", 0.0, -0.1),
            ]
            return results

    results, _ = transcribe_corpus(
        BrokenSegments(),
        "fake",
        _inputs(tmp_path, count=1),
        None,
        1,
        profile=False,
        host={"chip": "test"},
    )

    issues = results[0].trust_issues
    assert any("segments are out of order" in issue for issue in issues)
    assert any("after audio duration" in issue for issue in issues)
    assert any("before its start" in issue for issue in issues)
    assert results[0].text == "clip-0"
    assert not results[0].trusted


def test_malformed_segment_items_become_serializable_diagnostics(tmp_path):
    class MalformedSegments(FakeBackend):
        def transcribe(self, paths, language=None, batch_size=1):
            results = super().transcribe(paths, language, batch_size)
            results[0].segments = [object()]
            return results

    results, _ = transcribe_corpus(
        MalformedSegments(),
        "fake",
        _inputs(tmp_path, count=1),
        None,
        1,
        profile=False,
        host={"chip": "test"},
    )

    assert any("must be a Segment" in issue for issue in results[0].trust_issues)
    assert results[0].segments is None
    assert not results[0].trusted
    write_jsonl(results, tmp_path / "diagnostic.jsonl")


def test_custom_provenance_identity_mismatch_blocks_trust(tmp_path):
    class ForgedProvenance(FakeBackend):
        def model_provenance(self):
            return _complete_provenance(backend="wrong-backend", model="wrong-model")

    results, _ = transcribe_corpus(
        ForgedProvenance(),
        "fake",
        _inputs(tmp_path, count=1),
        None,
        1,
        profile=False,
        host={"chip": "test"},
    )

    assert results[0].model_provenance is not None
    assert any("model provenance backend mismatch" in issue for issue in results[0].trust_issues)
    assert any("model provenance model mismatch" in issue for issue in results[0].trust_issues)
    assert not results[0].trusted


def test_custom_provenance_accepts_explicit_identity_aliases(tmp_path):
    class AliasedProvenance(FakeBackend):
        provenance_backend_aliases = ("legacy-fake",)
        provenance_model_aliases = ("legacy-model",)

        def model_provenance(self):
            return _complete_provenance(backend="legacy-fake", model="legacy-model")

    results, _ = transcribe_corpus(
        AliasedProvenance(),
        "fake",
        _inputs(tmp_path, count=1),
        None,
        1,
        profile=False,
        host={"chip": "test"},
    )

    assert results[0].trusted
    assert not any("model provenance" in issue for issue in results[0].trust_issues)


@pytest.mark.parametrize("bad_path", [None, 123, b"clip.wav", ""])
def test_malformed_result_audio_path_is_a_per_item_contract_error(tmp_path, bad_path):
    class MalformedPath(FakeBackend):
        def transcribe(self, paths, language=None, batch_size=1):
            results = super().transcribe(paths, language, batch_size)
            results[0].audio_path = bad_path
            return results

    inputs = _inputs(tmp_path, count=1)
    results, _ = transcribe_corpus(
        MalformedPath(),
        "fake",
        inputs,
        None,
        1,
        profile=False,
        host={"chip": "test"},
    )

    result = results[0]
    assert result.audio_path == str(inputs[0].prepared_path)
    assert result.error is not None
    assert "audio_path" in result.error
    assert any("audio_path" in issue for issue in result.trust_issues)
    assert result.metadata["returned_audio_path"] == repr(bad_path)
    assert not result.trusted


def test_bound_backend_marks_resolved_setting_drift_incomplete():
    provenance = _complete_provenance()
    binding = ModelBinding(provenance, provenance.artifacts)
    binding.validate()
    backend = BoundBackend(language="mya_Mymr", batch_size=1)
    backend.bind_model(binding)
    backend.set_execution_settings(language="mya_Mymr", batch_size=2)

    current = backend.model_provenance()

    assert current.resolved_settings["device"] == "mps"
    assert current.requested_settings["batch_size"] == 2
    assert any(
        "resolved setting 'device' changed from preflight" in issue for issue in current.issues
    )
    assert any(
        "requested setting 'batch_size' changed from preflight" in issue for issue in current.issues
    )
    assert not current.complete


def test_post_call_provenance_failure_keeps_diagnostic_but_not_trust(tmp_path):
    backend = PostCallProvenanceFailure()
    results, _ = transcribe_corpus(
        backend,
        "fake",
        _inputs(tmp_path, count=1),
        "mya_Mymr",
        1,
        profile=False,
        host={"chip": "test"},
    )

    assert backend.provenance_calls == 2
    assert results[0].model_provenance is not None
    assert not results[0].trusted
    assert any(
        "post-call model provenance unavailable" in issue for issue in results[0].trust_issues
    )
