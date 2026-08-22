from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pytest

from stt.backends.omniasr_torch import OmniASRTorchBackend
from stt.provenance import ArtifactDigest, ModelBinding, ModelProvenance, digest_file


class RecordingPipeline:
    def __init__(self, transcripts: dict[str, str] | None = None) -> None:
        self.calls: list[tuple[list[str], list[str] | None, int]] = []
        self.transcripts = transcripts or {}

    def transcribe(self, paths, *, lang, batch_size):
        self.calls.append((list(paths), lang, batch_size))
        return [self.transcripts.get(Path(path).name, Path(path).stem) for path in paths]


def _backend(pipeline) -> OmniASRTorchBackend:
    backend = OmniASRTorchBackend("omniASR_LLM_Unlimited_300M_v2")
    backend.pipeline = pipeline
    backend.resolved_device = "cpu"
    backend.resolved_dtype = "float32"
    # Keep these tests independent of the optional omnilingual_asr package.
    backend._check_language = lambda language: None
    return backend


def test_transcribe_passes_all_paths_and_matching_language_list(monkeypatch, tmp_path):
    paths = [tmp_path / name for name in ("first.wav", "second.wav", "third.wav")]
    durations = {path: float(index + 1) for index, path in enumerate(paths)}
    monkeypatch.setattr("stt.audio.duration_of", durations.__getitem__)
    pipeline = RecordingPipeline(
        {"first.wav": " one ", "second.wav": " two ", "third.wav": " three "}
    )

    results = _backend(pipeline).transcribe(paths, language="mya_Mymr", batch_size=4)

    assert pipeline.calls == [([str(path) for path in paths], ["mya_Mymr"] * 3, 4)]
    assert [result.text for result in results] == ["one", "two", "three"]
    assert [result.audio_path for result in results] == [str(path) for path in paths]
    assert all(result.metadata["elapsed_s_source"] == "batch_proportional" for result in results)
    assert sum(result.elapsed_s or 0.0 for result in results) == pytest.approx(
        results[0].metadata["batch_elapsed_s"]
    )


def test_length_validation_keeps_invalid_file_out_of_batch(monkeypatch, tmp_path):
    long_path = tmp_path / "too-long.wav"
    short_path = tmp_path / "short.wav"
    durations = {long_path: 41.0, short_path: 2.0}
    monkeypatch.setattr("stt.audio.duration_of", durations.__getitem__)
    pipeline = RecordingPipeline()
    backend = OmniASRTorchBackend("omniASR_LLM_7B_v2")
    backend.pipeline = pipeline
    backend.resolved_device = "cpu"
    backend.resolved_dtype = "float32"
    backend._check_language = lambda language: None

    results = backend.transcribe([long_path, short_path], batch_size=2)

    assert pipeline.calls == [([str(short_path)], None, 2)]
    assert "accepts at most 40s" in (results[0].error or "")
    assert results[1].error is None
    assert [result.audio_path for result in results] == [str(long_path), str(short_path)]


def test_batch_exception_retries_each_file_and_preserves_errors(monkeypatch, tmp_path):
    paths = [tmp_path / name for name in ("ok.wav", "bad.wav", "also-ok.wav")]
    monkeypatch.setattr("stt.audio.duration_of", lambda path: 1.0)

    class BatchFails:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def transcribe(self, paths, *, lang, batch_size):
            paths = list(paths)
            self.calls.append(paths)
            if len(paths) > 1:
                time.sleep(0.01)
                raise RuntimeError("batch unsupported")
            if Path(paths[0]).name == "bad.wav":
                time.sleep(0.005)
                raise ValueError("bad audio")
            time.sleep(0.005)
            return [Path(paths[0]).stem]

    pipeline = BatchFails()
    results = _backend(pipeline).transcribe(paths, batch_size=3)

    assert pipeline.calls[0] == [str(path) for path in paths]
    assert pipeline.calls[1:] == [[str(path)] for path in paths]
    assert [result.text for result in results] == ["ok", "", "also-ok"]
    assert results[1].error == "ValueError: bad audio"
    assert all(result.metadata.get("batch_fallback") for result in results)
    assert all(
        result.metadata.get("elapsed_s_source") == "measured"
        for result in results
        if not result.error
    )
    assert results[0].elapsed_s is not None and results[0].elapsed_s >= 0.004
    assert results[2].elapsed_s is not None and results[2].elapsed_s >= 0.004


def test_batch_size_must_be_positive(monkeypatch, tmp_path):
    path = tmp_path / "clip.wav"
    monkeypatch.setattr("stt.audio.duration_of", lambda _: 1.0)
    backend = _backend(RecordingPipeline())
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        backend.transcribe([path], batch_size=0)


def test_bound_load_passes_torch_device_to_fairseq2(monkeypatch, tmp_path):
    """Bound fairseq2 APIs receive a device object, not its provenance label."""
    import torch
    from fairseq2.assets import AssetCard
    from fairseq2.data.tokenizers import hub as tokenizer_hub
    from fairseq2.models import hub as model_hub
    from omnilingual_asr.models.inference import pipeline as pipeline_module

    model = "omniASR_LLM_Unlimited_300M_v2"
    tokenizer_name = "omniASR_tokenizer_v2"
    checkpoint = tmp_path / "checkpoint.pt"
    tokenizer = tmp_path / "tokenizer.model"
    checkpoint.write_bytes(b"checkpoint")
    tokenizer.write_bytes(b"tokenizer")
    checkpoint_uri = checkpoint.resolve().as_uri()
    tokenizer_uri = tokenizer.resolve().as_uri()

    source_card = AssetCard(
        model,
        {
            "model_family": "wav2vec2_llama",
            "model_arch": "wav2vec2_llama",
            "checkpoint": checkpoint_uri,
            "tokenizer_ref": tokenizer_name,
        },
    )
    source_tokenizer = AssetCard(
        tokenizer_name,
        {
            "tokenizer_family": "sentencepiece",
            "tokenizer": tokenizer_uri,
        },
    )
    card_semantics = {
        "name": source_card.name,
        "model_family": source_card.field("model_family").as_(str),
        "model_arch": source_card.field("model_arch").as_(str),
        "checkpoint": str(source_card.field("checkpoint").as_uri()),
        "tokenizer_ref": tokenizer_name,
        "tokenizer_family": source_tokenizer.field("tokenizer_family").as_(str),
        "tokenizer": str(source_tokenizer.field("tokenizer").as_uri()),
    }
    card_json = json.dumps(
        card_semantics, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    card_digest = hashlib.sha256(card_json).hexdigest()
    artifacts = (
        digest_file(checkpoint, role="weights", name=checkpoint.name),
        digest_file(tokenizer, role="tokenizer", name=tokenizer.name),
        ArtifactDigest("card-metadata", f"{model}.card.json", len(card_json), card_digest),
    )
    provenance = ModelProvenance(
        backend="omniasr-torch",
        requested_model=model,
        source_kind="fairseq2-card",
        source_locator=model,
        upstream_revision=None,
        revision_status="unavailable-content-addressed",
        artifacts=artifacts,
        resolved_settings={"device": "cpu", "dtype": "float32"},
    ).finalized()
    binding = ModelBinding(provenance, artifacts)

    class Store:
        def retrieve_card(self, name):
            return {model: source_card, tokenizer_name: source_tokenizer}[name]

    load_model_devices: list[object] = []
    pipeline_devices: list[object] = []

    def fake_load_model(card, *, device, dtype, progress):
        del card, dtype, progress
        load_model_devices.append(device)
        return object()

    def fake_load_tokenizer(card, *, progress):
        del card, progress
        return object()

    class FakePipeline:
        def __init__(self, model_card, *, model, tokenizer, device, dtype):
            del model_card, model, tokenizer, dtype
            pipeline_devices.append(device)

    monkeypatch.setattr("fairseq2.assets.get_asset_store", lambda: Store())
    monkeypatch.setattr(model_hub, "load_model", fake_load_model)
    monkeypatch.setattr(tokenizer_hub, "load_tokenizer", fake_load_tokenizer)
    monkeypatch.setattr(pipeline_module, "ASRInferencePipeline", FakePipeline)

    backend = OmniASRTorchBackend(model, device="cpu", dtype="float32")
    backend.bind_model(binding)
    backend.load()

    assert backend.resolved_device == "cpu"
    assert all(isinstance(device, torch.device) for device in load_model_devices)
    assert all(isinstance(device, torch.device) for device in pipeline_devices)
    assert load_model_devices[0].type == "cpu"
    assert pipeline_devices[0].type == "cpu"
