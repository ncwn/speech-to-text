import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest

from stt.backends.transformers_asr import TransformersASRBackend, _bound_hf_snapshot
from stt.provenance import ArtifactDigest, ModelBinding, ModelProvenance, digest_file
from stt.results import Segment


def _hf_binding(tmp_path, model: str) -> tuple[ModelBinding, Path]:
    root = tmp_path / "blobs"
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
    for index, (name, content) in enumerate(files.items()):
        # Hugging Face snapshots expose logical names through symlinks into
        # flat content-addressed blobs.  The worker binding records the blob
        # path while provenance retains the snapshot-relative loader name.
        path = root / f"{index:064x}"
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


def test_seamless_mps_batches_by_duration_and_releases_cache(monkeypatch, tmp_path):
    import torch

    paths = [tmp_path / f"{name}.wav" for name in "abcde"]
    durations = dict(zip(paths, (5.0, 1.0, 4.0, 2.0, 3.0), strict=True))
    monkeypatch.setattr("stt.audio.duration_of", durations.__getitem__)
    synchronized: list[str] = []
    emptied: list[str] = []
    monkeypatch.setattr(torch.mps, "synchronize", lambda: synchronized.append("sync"))
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: emptied.append("empty"))
    calls: list[list[Path]] = []
    backend = TransformersASRBackend("seamless-m4t-v2")
    backend._loaded = True
    backend.resolved_device = "mps"
    backend.resolved_dtype = "float32"
    backend._check_language = lambda language: None

    def transcribe_batch(group, batch_size):
        del batch_size
        calls.append(list(group))
        return [[Segment(path.stem, 0.0, durations[path])] for path in group]

    def transcribe_one(path):
        calls.append([path])
        return [Segment(path.stem, 0.0, durations[path])]

    backend._transcribe_seamless_batch = transcribe_batch
    backend._transcribe_seamless = transcribe_one
    results = backend.transcribe(paths, language="mya_Mymr", batch_size=4)

    assert calls == [[paths[1], paths[3]], [paths[4], paths[2]], [paths[0]]]
    assert [result.text for result in results] == list("abcde")
    assert [result.metadata["batch_group"] for result in results] == [2, 0, 1, 0, 1]
    assert all(result.metadata["batch_size"] == 4 for result in results)
    assert all(result.metadata["effective_batch_size"] == 2 for result in results)
    assert synchronized == ["sync"] * 3
    assert emptied == ["empty"] * 3


def test_hf_bound_pipeline_uses_exact_snapshot_for_all_components(monkeypatch, tmp_path):
    binding, _ = _hf_binding(tmp_path, "whisper-my-small")
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
    snapshot = Path(kwargs["model"])
    assert snapshot.is_dir()
    assert kwargs["tokenizer"] == str(snapshot)
    assert kwargs["feature_extractor"] == str(snapshot)
    assert kwargs["model_kwargs"] == {"local_files_only": True}
    assert "revision" not in kwargs
    for artifact in binding.provenance.artifacts:
        target = snapshot / artifact.name
        assert target.is_symlink()
        assert target.resolve() == Path(
            next(item.path for item in binding.paths if item.name == artifact.name)
        )


def test_hf_bound_processor_and_model_load_from_exact_snapshot(monkeypatch, tmp_path):
    binding, _ = _hf_binding(tmp_path, "seamless-m4t-v2")
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

    snapshot = Path(calls[0][1])
    assert calls == [
        ("processor", str(snapshot), {"local_files_only": True}),
        ("model", str(snapshot), {"local_files_only": True, "dtype": "float32"}),
    ]
    assert all((snapshot / artifact.name).is_symlink() for artifact in binding.provenance.artifacts)


def test_hf_bound_mms_adapter_stays_on_local_snapshot(monkeypatch, tmp_path):
    binding, _ = _hf_binding(tmp_path, "mms-1b-all")
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

    snapshot = Path(next(call[1] for call in calls if call[0] == "processor"))
    assert ("processor", str(snapshot), {"local_files_only": True}) in calls
    assert ("model", str(snapshot), {"local_files_only": True, "dtype": "float32"}) in calls
    assert ("adapter", "mya", {"local_files_only": True}) in calls
    assert (snapshot / "vocabs/mya.txt").is_symlink()


def test_hf_bound_snapshot_rejects_duplicate_path_selection(tmp_path):
    binding, _ = _hf_binding(tmp_path, "whisper-my-small")
    duplicate = replace(binding.paths[0], path=str(tmp_path / "alternate" / "config.json"))
    malformed = ModelBinding(binding.provenance, (*binding.paths, duplicate))

    with pytest.raises(RuntimeError, match="duplicate path names"):
        _bound_hf_snapshot(malformed)


def test_hf_bound_snapshot_rejects_artifact_name_escape(tmp_path):
    binding, _ = _hf_binding(tmp_path, "whisper-my-small")
    escaped_name = "../outside/config.json"
    artifact = replace(binding.provenance.artifacts[0], name=escaped_name)
    path = replace(binding.paths[0], name=escaped_name)
    malformed = ModelBinding(
        replace(binding.provenance, artifacts=(artifact, *binding.provenance.artifacts[1:])),
        (path, *binding.paths[1:]),
    )

    with pytest.raises(RuntimeError, match="escapes its snapshot"):
        _bound_hf_snapshot(malformed)


def test_hf_bound_snapshot_reuses_materialized_blob_view(tmp_path):
    binding, _ = _hf_binding(tmp_path, "mms-1b-all")

    first = _bound_hf_snapshot(binding)
    second = _bound_hf_snapshot(binding)

    assert second == first
    assert all((first / artifact.name).is_symlink() for artifact in binding.provenance.artifacts)
    assert (first / "vocabs/mya.txt").read_bytes() == b"vocab"


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


def test_mms_parity_hooks_capture_pipeline_and_direct_tensors(tmp_path):
    import types

    import numpy as np
    import soundfile as sf
    import torch
    from transformers.pipelines.audio_utils import ffmpeg_read

    from stt.parity import compare_traces, contract_for

    path = tmp_path / "clip.wav"
    sf.write(path, np.linspace(-0.5, 0.5, 8000), 16_000, subtype="PCM_16")

    class Batch(dict):
        def to(self, device=None, dtype=None):
            for key, value in tuple(self.items()):
                if isinstance(value, torch.Tensor):
                    value = value.to(device=device)
                    if dtype is not None and value.is_floating_point():
                        value = value.to(dtype=dtype)
                    self[key] = value
            return self

    class FeatureExtractor:
        sampling_rate = 16_000

        def __call__(self, audio, **kwargs):
            del kwargs
            values = torch.from_numpy(np.array(audio, dtype=np.float32, copy=True)).unsqueeze(0)
            return Batch(
                input_values=values,
                attention_mask=torch.ones_like(values, dtype=torch.long),
            )

    class Model(torch.nn.Module):
        main_input_name = "input_values"
        dtype = torch.float32

        def forward(self, input_values, attention_mask=None):
            del attention_mask
            return types.SimpleNamespace(logits=torch.stack((-input_values, input_values), dim=-1))

    class Tokenizer:
        @staticmethod
        def decode(tokens, skip_special_tokens=False):
            del skip_special_tokens
            return "".join("က" if token else " " for token in tokens)

    class Pipe:
        feature_extractor = FeatureExtractor()
        model = Model()
        tokenizer = Tokenizer()
        device = torch.device("cpu")

        def __call__(self, audio_path, **kwargs):
            del kwargs
            pcm = ffmpeg_read(Path(audio_path).read_bytes(), self.feature_extractor.sampling_rate)
            inputs = self.feature_extractor(pcm).to(device=self.device, dtype=self.model.dtype)
            output = self.model(**inputs)
            tokens = output.logits.argmax(dim=-1)
            return {"text": self.tokenizer.decode(tokens[0].tolist())}

    backend = TransformersASRBackend("mms-1b-all", device="cpu", dtype="float32")
    backend.pipe = Pipe()
    backend._loaded = True
    backend.resolved_device = "cpu"
    backend.resolved_dtype = "float32"

    adapter = backend.parity_adapter_trace(path, language="mya_Mymr")
    reference = backend.parity_reference_trace(path, language="mya_Mymr")
    contract = contract_for("hf", "mms-1b-all")
    comparisons = compare_traces(adapter, reference, tolerances=contract.tolerances)

    assert all(item.status == "match" for item in comparisons)
    assert adapter.metadata["entrypoint"] == "transformers-pipeline"
    assert reference.metadata["entrypoint"] == "feature-extractor-model-tokenizer"


def test_seamless_parity_hooks_capture_adapter_and_direct_encoder(tmp_path):
    import types

    import numpy as np
    import soundfile as sf
    import torch

    from stt.parity import compare_traces, contract_for

    path = tmp_path / "clip.wav"
    sf.write(path, np.linspace(-0.5, 0.5, 8000), 16_000, subtype="PCM_16")

    class Batch(dict):
        def to(self, device=None, dtype=None):
            for key, value in tuple(self.items()):
                if isinstance(value, torch.Tensor):
                    value = value.to(device=device)
                    if dtype is not None and value.is_floating_point():
                        value = value.to(dtype=dtype)
                    self[key] = value
            return self

    class Processor:
        def __call__(self, **kwargs):
            audio = kwargs.get("audio", kwargs.get("audios"))
            values = torch.from_numpy(np.array(audio, dtype=np.float32, copy=True)).unsqueeze(0)
            return Batch(
                input_features=values,
                attention_mask=torch.ones_like(values, dtype=torch.long),
            )

        @staticmethod
        def decode(tokens, skip_special_tokens=True):
            del tokens, skip_special_tokens
            return " စာ "

    class Encoder(torch.nn.Module):
        def forward(self, input_features, attention_mask=None):
            del attention_mask
            return types.SimpleNamespace(last_hidden_state=input_features * 2)

    class Model(torch.nn.Module):
        device = torch.device("cpu")
        dtype = torch.float32

        def __init__(self):
            super().__init__()
            self.speech_encoder = Encoder()

        def generate(self, input_features, attention_mask=None, tgt_lang=None):
            del tgt_lang
            self.speech_encoder(
                input_features=input_features,
                attention_mask=attention_mask,
            )
            return torch.tensor([[1, 2, 3]])

    backend = TransformersASRBackend("seamless-m4t-v2", device="cpu", dtype="float32")
    backend._seamless = (Processor(), Model())
    backend._loaded = True
    backend.resolved_device = "cpu"
    backend.resolved_dtype = "float32"

    adapter = backend.parity_adapter_trace(path, language="mya_Mymr")
    reference = backend.parity_reference_trace(path, language="mya_Mymr")
    contract = contract_for("hf", "seamless-m4t-v2")
    comparisons = compare_traces(adapter, reference, tolerances=contract.tolerances)

    assert all(item.status == "match" for item in comparisons)
    assert adapter.metadata["entrypoint"] == "repository-single-window"
    assert reference.metadata["entrypoint"] == "processor-generate-direct"
