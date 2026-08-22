import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest

from stt.backends.transformers_asr import TransformersASRBackend, _bound_hf_snapshot
from stt.provenance import ArtifactDigest, ModelBinding, ModelProvenance, digest_file


def _hf_binding(tmp_path, model: str) -> tuple[ModelBinding, Path]:
    root = tmp_path / "snapshot"
    files = {
        "config.json": b"{}",
        "model.safetensors": b"weights",
        "tokenizer.json": b"tokenizer",
        "feature_extractor_config.json": b"features",
    }
    if model == "mms-1b-all":
        files["adapter.mya.safetensors"] = b"adapter"
        files["vocabs/mya.txt"] = b"vocab"
    artifacts: list[ArtifactDigest] = []
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        artifacts.append(digest_file(path, role="model-or-processor", name=name))
    provenance = ModelProvenance(
        backend="hf",
        requested_model=model,
        source_kind="huggingface",
        source_locator="example/model",
        upstream_revision="a" * 40,
        revision_status="pinned",
        artifacts=tuple(artifacts),
    )
    binding = ModelBinding(provenance=provenance, paths=tuple(artifacts))
    binding.validate()
    return binding, root


class RecordingPipeline:
    def __init__(self) -> None:
        self.calls: list[tuple[object, dict]] = []

    def __call__(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        if isinstance(audio, list):
            return [{"text": Path(path).stem} for path in audio]
        return {"text": Path(audio).stem}


def test_hf_pipeline_batch_forwards_list_and_preserves_order(monkeypatch, tmp_path):
    paths = [tmp_path / name for name in ("first.wav", "second.wav", "third.wav")]
    monkeypatch.setattr(
        "stt.audio.duration_of",
        {path: float(index + 1) for index, path in enumerate(paths)}.__getitem__,
    )
    pipe = RecordingPipeline()
    backend = TransformersASRBackend("mms-1b-all")
    backend.pipe = pipe
    backend._loaded = True
    backend.resolved_device = "cpu"
    backend.resolved_dtype = "float32"
    backend._check_language = lambda language: None

    results = backend.transcribe(paths, language="mya_Mymr", batch_size=2)

    assert [result.text for result in results] == ["first", "second", "third"]
    assert isinstance(pipe.calls[0][0], list)
    assert pipe.calls[0][0] == [str(path) for path in paths[:2]]
    assert pipe.calls[0][1]["batch_size"] == 2
    assert results[0].metadata["elapsed_s_source"] == "batch_proportional"
    # The trailing chunk holds one file, so it is decoded serially and its
    # time is measured rather than apportioned.
    assert results[2].metadata["elapsed_s_source"] == "measured"


def test_hf_batch_failure_retries_individually(monkeypatch, tmp_path):
    paths = [tmp_path / "first.wav", tmp_path / "second.wav"]
    monkeypatch.setattr("stt.audio.duration_of", lambda path: 1.0)

    class BatchFails(RecordingPipeline):
        def __call__(self, audio, **kwargs):
            if isinstance(audio, list):
                raise RuntimeError("batch unsupported")
            return super().__call__(audio, **kwargs)

    pipe = BatchFails()
    backend = TransformersASRBackend("mms-1b-all")
    backend.pipe = pipe
    backend._loaded = True
    backend.resolved_device = "cpu"
    backend.resolved_dtype = "float32"
    backend._check_language = lambda language: None

    results = backend.transcribe(paths, batch_size=2)

    assert [result.text for result in results] == ["first", "second"]
    assert all(result.metadata["batch_fallback"] for result in results)


def test_hf_bound_pipeline_uses_exact_snapshot_for_all_components(monkeypatch, tmp_path):
    binding, root = _hf_binding(tmp_path, "whisper-my-small")
    calls: list[tuple[str, dict]] = []

    def fake_pipeline(task, **kwargs):
        calls.append((task, kwargs))
        return object()

    transformers = ModuleType("transformers")
    transformers.__version__ = "4.57.6"
    transformers.AutoProcessor = object()
    transformers.pipeline = fake_pipeline
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    backend = TransformersASRBackend("whisper-my-small", device="cpu")
    backend._resolve_device = lambda: "cpu"
    backend._resolve_dtype = lambda device: "float32"
    backend.bind_model(binding)
    backend.load()

    task, kwargs = calls[0]
    assert task == "automatic-speech-recognition"
    assert kwargs["model"] == str(root)
    assert kwargs["tokenizer"] == str(root)
    assert kwargs["feature_extractor"] == str(root)
    assert kwargs["model_kwargs"] == {"local_files_only": True}
    assert "revision" not in kwargs


def test_hf_bound_processor_and_model_load_from_exact_snapshot(monkeypatch, tmp_path):
    binding, root = _hf_binding(tmp_path, "seamless-m4t-v2")
    calls: list[tuple[str, str, dict]] = []

    class FakeModel:
        device = "cpu"
        dtype = "float32"

        def to(self, device):
            return self

        def eval(self):
            return self

    class FakeProcessor:
        pass

    processor = FakeProcessor()
    model = FakeModel()

    class AutoProcessor:
        @staticmethod
        def from_pretrained(source, **kwargs):
            calls.append(("processor", source, kwargs))
            return processor

    class SeamlessModel:
        @staticmethod
        def from_pretrained(source, **kwargs):
            calls.append(("model", source, kwargs))
            return model

    def fake_pipeline(*args, **kwargs):
        raise AssertionError("Seamless must not construct an ASR pipeline")

    transformers = ModuleType("transformers")
    transformers.__version__ = "4.57.6"
    transformers.AutoProcessor = AutoProcessor
    transformers.SeamlessM4Tv2ForSpeechToText = SeamlessModel
    transformers.pipeline = fake_pipeline
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    backend = TransformersASRBackend("seamless-m4t-v2", device="cpu")
    backend._resolve_device = lambda: "cpu"
    backend._resolve_dtype = lambda device: "float32"
    backend.bind_model(binding)
    backend.load()

    assert calls == [
        ("processor", str(root), {"local_files_only": True}),
        ("model", str(root), {"local_files_only": True, "dtype": "float32"}),
    ]


def test_hf_bound_mms_adapter_stays_on_local_snapshot(monkeypatch, tmp_path):
    binding, root = _hf_binding(tmp_path, "mms-1b-all")
    calls: list[tuple[str, object, dict]] = []

    class Tokenizer:
        def set_target_lang(self, language):
            calls.append(("target_lang", language, {}))

    class Processor:
        tokenizer = Tokenizer()
        feature_extractor = object()

    class Model:
        def load_adapter(self, language, **kwargs):
            calls.append(("adapter", language, kwargs))

        def to(self, device):
            return self

        def eval(self):
            return self

    processor = Processor()
    model = Model()

    class AutoProcessor:
        @staticmethod
        def from_pretrained(source, **kwargs):
            calls.append(("processor", source, kwargs))
            return processor

    class MmsModel:
        @staticmethod
        def from_pretrained(source, **kwargs):
            calls.append(("model", source, kwargs))
            return model

    def fake_pipeline(*args, **kwargs):
        calls.append(("pipeline", kwargs["model"], kwargs))
        return object()

    transformers = ModuleType("transformers")
    transformers.__version__ = "4.57.6"
    transformers.AutoProcessor = AutoProcessor
    transformers.Wav2Vec2ForCTC = MmsModel
    transformers.pipeline = fake_pipeline
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    backend = TransformersASRBackend("mms-1b-all", device="cpu")
    backend._resolve_device = lambda: "cpu"
    backend._resolve_dtype = lambda device: "float32"
    backend.bind_model(binding)
    backend.load()

    assert ("processor", str(root), {"local_files_only": True}) in calls
    assert ("model", str(root), {"local_files_only": True, "dtype": "float32"}) in calls
    assert ("adapter", "mya", {"local_files_only": True}) in calls


def test_hf_bound_snapshot_rejects_duplicate_path_selection(tmp_path):
    binding, _ = _hf_binding(tmp_path, "whisper-my-small")
    duplicate = replace(binding.paths[0], path=str(tmp_path / "alternate" / "config.json"))
    malformed = ModelBinding(binding.provenance, (*binding.paths, duplicate))

    with pytest.raises(RuntimeError, match="duplicate path names"):
        _bound_hf_snapshot(malformed)


class SleepingPipeline:
    """A pipeline whose per-file cost differs measurably."""

    def __init__(self, seconds: dict[str, float]) -> None:
        self.seconds = seconds

    def __call__(self, audio, **kwargs):
        import time as _time

        if isinstance(audio, list):
            raise AssertionError("whisper must not be batched")
        _time.sleep(self.seconds[Path(audio).stem])
        return {"text": Path(audio).stem}


def test_serial_decode_times_each_file_instead_of_copying_the_loop(monkeypatch, tmp_path):
    """A serial loop is measured per item, not billed the whole loop each time.

    Whisper never batches, so before this every file in a chunk was stamped
    with the wall time of the entire loop -- three files reported three times
    the work actually done, and the corpus RTF derived from those numbers was
    wrong by the chunk size.
    """
    names = ("first", "second", "third")
    paths = [tmp_path / f"{name}.wav" for name in names]
    monkeypatch.setattr("stt.audio.duration_of", lambda path: 1.0)
    costs = {"first": 0.002, "second": 0.02, "third": 0.002}

    backend = TransformersASRBackend("whisper-my-small")
    backend.pipe = SleepingPipeline(costs)
    backend._loaded = True
    backend.resolved_device = "cpu"
    backend.resolved_dtype = "float32"
    backend._check_language = lambda language: None

    results = backend.transcribe(paths, language="my", batch_size=3)

    assert [result.text for result in results] == list(names)
    assert all(result.metadata["elapsed_s_source"] == "measured" for result in results)
    elapsed = [result.elapsed_s for result in results]
    assert all(value is not None for value in elapsed)
    # Disjoint sub-intervals of one loop: they must sum to no more than the
    # loop, where copying the total would have summed to three times it.
    assert sum(elapsed) <= sum(costs.values()) * 3 + 0.5
    # The slow file must be visibly the slow one.
    assert elapsed[1] > elapsed[0]
    assert elapsed[1] > elapsed[2]
