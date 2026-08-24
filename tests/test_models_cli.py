import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

from stt import cli
from stt.registry import all_backends, get_backend
from stt.results import TranscriptionResult

runner = CliRunner()


def test_models_download_requires_a_backend():
    result = runner.invoke(cli.app, ["models", "--download", "small"], terminal_width=240)

    assert result.exit_code == 2
    assert "--backend is required" in result.output


def test_models_rejects_unknown_backend_and_model(monkeypatch):
    result = runner.invoke(cli.app, ["models", "--backend", "not-a-backend"])
    assert result.exit_code == 2
    assert "Unknown backend" in result.output

    backend = get_backend("hf")
    monkeypatch.setattr(backend, "is_available", classmethod(lambda cls: (True, "test")))
    result = runner.invoke(cli.app, ["models", "--backend", "hf", "--download", "not-a-model"])
    assert result.exit_code == 2
    assert "Unknown HF model" in result.output


def test_models_downloads_only_the_selected_model(monkeypatch):
    selected: list[tuple[str, str]] = []
    backend = get_backend("hf")
    monkeypatch.setattr(backend, "is_available", classmethod(lambda cls: (True, "test")))
    monkeypatch.setattr(
        backend, "download_weights", lambda self: selected.append((self.name, self.model))
    )
    for name, other in all_backends().items():
        if name != "hf":
            monkeypatch.setattr(
                other,
                "download_weights",
                lambda self, name=name: pytest.fail(f"unexpected download: {name}"),
            )

    result = runner.invoke(
        cli.app,
        ["models", "--backend", "hf", "--download", "whisper-my-small"],
    )

    assert result.exit_code == 0, result.output
    assert selected == [("hf", "whisper-my-small")]


def test_models_missing_runtime_includes_install_hint(monkeypatch):
    backend = get_backend("dolphin")
    monkeypatch.setattr(
        backend, "is_available", classmethod(lambda cls: (False, "missing dependency: dolphin"))
    )

    result = runner.invoke(
        cli.app,
        ["models", "--backend", "dolphin", "--download", "small"],
        terminal_width=240,
    )

    assert result.exit_code == 2
    assert backend.install_hint in result.output


def test_gguf_rejects_a_device_override(monkeypatch):
    backend = get_backend("omniasr-gguf")
    monkeypatch.setattr(backend, "is_available", classmethod(lambda cls: (True, "test")))

    with pytest.raises(typer.BadParameter, match="--device is not supported"):
        cli._run_backend("omniasr-gguf", None, [], None, 1, {"device": "cpu"})


def test_transcribe_rejects_an_unknown_dtype_before_loading(monkeypatch):
    backend = get_backend("hf")
    monkeypatch.setattr(backend, "is_available", classmethod(lambda cls: (True, "test")))
    monkeypatch.setattr(cli, "_prepare", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        backend,
        "load",
        lambda self: pytest.fail("backend load ran before dtype validation"),
    )

    result = runner.invoke(
        cli.app,
        ["transcribe", "audio.wav", "--backend", "hf", "--dtype", "float8"],
    )

    assert result.exit_code == 2
    assert "Unknown dtype 'float8'" in result.output


def test_compare_explains_an_unknown_cache_status(monkeypatch):
    class UnknownCacheBackend:
        name = "unknown-cache"
        model = "large"

        @classmethod
        def is_available(cls):
            return True, "test"

        def weights_cached(self):
            return None

    monkeypatch.setattr(cli, "_prepare", lambda *args, **kwargs: [Path("audio.wav")])
    monkeypatch.setattr(cli, "all_backends", lambda: {"unknown-cache": UnknownCacheBackend})
    monkeypatch.setattr(
        cli, "_run_backend", lambda *args, **kwargs: pytest.fail("backend ran implicitly")
    )

    result = runner.invoke(cli.app, ["compare", "audio.wav"])

    assert result.exit_code == 1
    assert "cache status cannot be verified" in result.output
    assert "--only unknown-cache" in result.output.replace("\n", " ")


def test_compare_does_not_run_dolphin_with_a_bad_checkpoint(monkeypatch, tmp_path):
    from stt.backends import dolphin

    cls = get_backend("dolphin")
    backend = cls()
    directory = tmp_path / backend.model
    directory.mkdir()
    for filename, _ in backend.spec.artifacts:
        (directory / filename).write_bytes(b"not an official artifact")

    monkeypatch.setattr(dolphin, "CACHE_ROOT", tmp_path)
    expected = dict(backend.spec.artifacts)
    monkeypatch.setattr(
        dolphin,
        "_sha256",
        lambda path: "wrong" if path.name == "small.pt" else expected[path.name],
    )
    monkeypatch.setattr(cls, "is_available", classmethod(lambda cls: (True, "test")))
    monkeypatch.setattr(cli, "_prepare", lambda *args, **kwargs: [Path("audio.wav")])
    monkeypatch.setattr(cli, "all_backends", lambda: {"dolphin": cls})
    monkeypatch.setattr(
        cli, "_run_backend", lambda *args, **kwargs: pytest.fail("download path ran")
    )

    assert backend.weights_cached() is False
    result = runner.invoke(cli.app, ["compare", "audio.wav"])

    assert result.exit_code == 1
    assert "--backend dolphin --download small" in result.output.replace("\n", " ")


def test_implicit_compare_makes_only_hf_cached_only(monkeypatch):
    class HFBackend:
        name = "hf"
        model = "hf-default"

        @classmethod
        def is_available(cls):
            return True, "test"

        def weights_cached(self):
            return True

    class OtherBackend:
        name = "other"
        model = "other-default"

        @classmethod
        def is_available(cls):
            return True, "test"

        def weights_cached(self):
            return True

    calls = []

    def run(name, model, files, language, batch_size, options):
        calls.append((name, options))
        return [TranscriptionResult(audio_path="audio.wav", text="", backend=name, model="model")]

    monkeypatch.setattr(cli, "_prepare", lambda *args, **kwargs: [Path("audio.wav")])
    monkeypatch.setattr(cli, "all_backends", lambda: {"hf": HFBackend, "other": OtherBackend})
    monkeypatch.setattr(cli, "_run_backend", run)
    monkeypatch.setattr(cli, "write_jsonl", lambda *args, **kwargs: None)

    result = runner.invoke(cli.app, ["compare", "audio.wav"])
    assert result.exit_code == 0, result.output
    assert calls == [("hf", {"local_files_only": True}), ("other", {})]

    calls.clear()
    result = runner.invoke(cli.app, ["compare", "audio.wav", "--only", "hf", "--yes"])
    assert result.exit_code == 0, result.output
    assert calls == [("hf", {})]


def test_gguf_cache_probe_stream_hashes_the_model(monkeypatch, tmp_path):
    payload = b"verified GGUF fixture"
    backend = get_backend("omniasr-gguf")()
    backend.spec = replace(backend.spec, sha256=hashlib.sha256(payload).hexdigest())
    monkeypatch.setitem(sys.modules, "crispasr", SimpleNamespace(cache_dir=lambda: tmp_path))
    path = tmp_path / backend.spec.filename

    assert backend.weights_cached() is False
    path.write_bytes(b"partial")
    assert backend.weights_cached() is False
    path.write_bytes(payload)
    assert backend.weights_cached() is True
    path.unlink()
    path.symlink_to(tmp_path / "missing.gguf")
    assert backend.weights_cached() is False


def test_gguf_download_validates_before_returning(monkeypatch, tmp_path):
    backend = get_backend("omniasr-gguf")()
    path = tmp_path / backend.spec.filename
    calls = []

    def download(filename, url, *, quiet):
        calls.append((filename, url, quiet))
        path.write_bytes(b"invalid")
        return str(path)

    monkeypatch.setitem(
        sys.modules,
        "crispasr",
        SimpleNamespace(cache_dir=lambda: tmp_path, cache_ensure_file=download),
    )

    with pytest.raises(RuntimeError, match="download failed integrity validation"):
        backend.download_weights()

    assert calls == [(backend.spec.filename, backend.spec.url, False)]


@pytest.mark.parametrize("entry", ["partial", "dangling"])
def test_gguf_load_refuses_an_invalid_existing_model(monkeypatch, tmp_path, entry):
    backend = get_backend("omniasr-gguf")()
    path = tmp_path / backend.spec.filename
    if entry == "partial":
        path.write_bytes(b"partial")
    else:
        path.symlink_to(tmp_path / "missing.gguf")

    monkeypatch.setitem(
        sys.modules,
        "crispasr",
        SimpleNamespace(
            cache_dir=lambda: tmp_path,
            cache_ensure_file=lambda *args, **kwargs: pytest.fail("download overwrote cache"),
            Session=lambda *args, **kwargs: pytest.fail("native loader received invalid cache"),
        ),
    )

    with pytest.raises(RuntimeError, match="cache failed integrity validation"):
        backend.load()
    if entry == "partial":
        assert path.read_bytes() == b"partial"
    else:
        assert path.is_symlink() and not path.exists()


def test_gguf_load_downloads_missing_model_then_reuses_it(monkeypatch, tmp_path):
    payload = b"verified GGUF fixture"
    backend = get_backend("omniasr-gguf")()
    backend.spec = replace(backend.spec, sha256=hashlib.sha256(payload).hexdigest())
    path = tmp_path / backend.spec.filename
    (tmp_path / f"{backend.spec.filename}.src").write_text("stale advisory URL")
    downloads = []

    def download(filename, url, *, quiet):
        downloads.append((filename, url, quiet))
        path.write_bytes(payload)
        return str(path)

    sessions = []
    monkeypatch.setitem(
        sys.modules,
        "crispasr",
        SimpleNamespace(
            cache_dir=lambda: tmp_path,
            cache_ensure_file=download,
            Session=lambda model_path, **options: sessions.append((model_path, options)),
        ),
    )

    backend.load()
    backend.load()

    assert downloads == [(backend.spec.filename, backend.spec.url, False)]
    assert sessions == [
        (str(path), {"backend": backend.spec.crisp_backend}),
        (str(path), {"backend": backend.spec.crisp_backend}),
    ]


def _mock_torch_download(monkeypatch, checkpoint: Path, tokenizer: Path) -> None:
    asset_store = object()
    download_manager = object()
    model_card = SimpleNamespace(
        name="model", field=lambda name: SimpleNamespace(as_uri=lambda: f"https:///{name}")
    )
    tokenizer_card = SimpleNamespace(
        name="tokenizer", field=lambda name: SimpleNamespace(as_uri=lambda: f"https:///{name}")
    )
    store = SimpleNamespace(retrieve_card=lambda name: model_card)
    manager = SimpleNamespace(
        download_model=lambda uri, name: checkpoint,
        download_tokenizer=lambda uri, name: tokenizer,
    )
    dependencies = {asset_store: store, download_manager: manager}
    resolver = SimpleNamespace(resolve=dependencies.__getitem__)
    monkeypatch.setitem(
        sys.modules,
        "fairseq2.assets",
        SimpleNamespace(AssetDownloadManager=download_manager, AssetStore=asset_store),
    )
    monkeypatch.setitem(
        sys.modules,
        "fairseq2.data.tokenizers.ref",
        SimpleNamespace(resolve_tokenizer_reference=lambda store, card: tokenizer_card),
    )
    monkeypatch.setitem(
        sys.modules,
        "fairseq2.runtime.dependency",
        SimpleNamespace(get_dependency_resolver=lambda: resolver),
    )


def test_torch_download_rejects_partial_checkpoint_without_deleting_it(monkeypatch, tmp_path):
    from stt.backends import omniasr_torch

    backend = get_backend("omniasr-torch")("omniASR_CTC_300M_v2")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"partial")
    tokenizer = tmp_path / "tokenizer.model"
    tokenizer.write_bytes(b"x" * omniasr_torch.TOKENIZER_SIZE_BYTES)
    monkeypatch.setattr(omniasr_torch, "_sha256", lambda path: omniasr_torch.TOKENIZER_SHA256)
    _mock_torch_download(monkeypatch, checkpoint, tokenizer)

    with pytest.raises(RuntimeError, match="failed integrity validation: checkpoint"):
        backend.download_weights()

    assert checkpoint.read_bytes() == b"partial"


def test_torch_download_rejects_wrong_tokenizer_hash_without_deleting_it(monkeypatch, tmp_path):
    from stt.backends import omniasr_torch

    backend = get_backend("omniasr-torch")("omniASR_CTC_300M_v2")
    checkpoint = tmp_path / "checkpoint.pt"
    with checkpoint.open("wb") as stream:
        stream.truncate(backend.spec.size_bytes)
    tokenizer = tmp_path / "tokenizer.model"
    tokenizer.write_bytes(b"x" * omniasr_torch.TOKENIZER_SIZE_BYTES)
    _mock_torch_download(monkeypatch, checkpoint, tokenizer)

    with pytest.raises(RuntimeError, match="failed integrity validation: tokenizer"):
        backend.download_weights()

    assert tokenizer.read_bytes() == b"x" * omniasr_torch.TOKENIZER_SIZE_BYTES


def test_torch_load_validates_download_before_pipeline_construction(monkeypatch):
    backend = get_backend("omniasr-torch")("omniASR_CTC_300M_v2")
    calls = []
    pipeline = SimpleNamespace(
        ASRInferencePipeline=lambda **kwargs: calls.append(("pipeline", kwargs)) or object()
    )
    monkeypatch.setitem(sys.modules, "omnilingual_asr.models.inference.pipeline", pipeline)
    monkeypatch.setattr(backend, "download_weights", lambda: calls.append("validate"))
    monkeypatch.setattr(backend, "_resolve_device", lambda: "cpu")
    monkeypatch.setattr(backend, "_resolve_dtype", lambda device: "float32")

    backend.load()

    assert calls == [
        "validate",
        (
            "pipeline",
            {"model_card": backend.model, "device": "cpu", "dtype": "float32"},
        ),
    ]


def test_hf_download_is_pinned_allowlisted_and_validated(monkeypatch, tmp_path):
    import huggingface_hub

    files = [
        "config.json",
        "tokenizer.json",
        "model.safetensors",
        "pytorch_model.bin",
        "flax_model.msgpack",
    ]
    calls: list[tuple[str, object]] = []

    def model_info(repo_id, *, revision, files_metadata):
        calls.append(("info", (repo_id, revision, files_metadata)))
        return SimpleNamespace(siblings=[SimpleNamespace(rfilename=name) for name in files])

    def snapshot_download(*, repo_id, revision, allow_patterns):
        calls.append(("download", (repo_id, revision, allow_patterns)))
        for filename in (
            "config.json",
            "tokenizer.json",
            "model.safetensors",
            "preprocessor_config.json",
        ):
            (tmp_path / filename).write_text("cached")
        return tmp_path

    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", staticmethod(model_info))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)

    backend = get_backend("hf")("whisper-large-v3")
    backend.download_weights()

    assert calls[0] == (
        "info",
        ("openai/whisper-large-v3", backend.spec.revision, True),
    )
    assert calls[1] == (
        "download",
        (
            "openai/whisper-large-v3",
            backend.spec.revision,
            ["config.json", "tokenizer.json", "model.safetensors"],
        ),
    )


def test_hf_download_rejects_an_incomplete_snapshot(monkeypatch, tmp_path):
    import huggingface_hub

    backend = get_backend("hf")("whisper-large-v3")
    monkeypatch.setattr(
        huggingface_hub.HfApi,
        "model_info",
        staticmethod(
            lambda *args, **kwargs: SimpleNamespace(
                siblings=[SimpleNamespace(rfilename="model.safetensors")]
            )
        ),
    )
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: tmp_path)

    with pytest.raises(RuntimeError, match="is incomplete"):
        backend.download_weights()


def test_hf_pipeline_load_is_pinned_and_can_be_cached_only(monkeypatch):
    backend = get_backend("hf")("whisper-my-small", local_files_only=True)
    dtype = object()
    calls = []

    monkeypatch.setattr(backend, "_resolve_device", lambda: "cpu")
    monkeypatch.setattr(backend, "_resolve_dtype", lambda device: dtype)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoProcessor=object(),
            pipeline=lambda task, **kwargs: calls.append((task, kwargs)) or object(),
        ),
    )

    backend.load()

    assert calls == [
        (
            "automatic-speech-recognition",
            {
                "model": backend.spec.repo,
                "revision": backend.spec.revision,
                "model_kwargs": {"local_files_only": True},
                "dtype": dtype,
                "device": "cpu",
            },
        )
    ]


@pytest.mark.parametrize("model_name", ["seamless-m4t-v2", "mms-1b-all"])
def test_hf_manual_loads_pin_every_hub_request(monkeypatch, model_name):
    backend = get_backend("hf")(model_name, local_files_only=True)
    dtype = object()
    calls = []

    class Tokenizer:
        def set_target_lang(self, language):
            calls.append(("language", language))

    class Model:
        def load_adapter(self, language, **kwargs):
            calls.append(("adapter", language, kwargs))

        def to(self, device):
            return self

        def eval(self):
            return self

    processor = SimpleNamespace(tokenizer=Tokenizer(), feature_extractor=object())
    loaded_model = Model()
    monkeypatch.setattr(backend, "_resolve_device", lambda: "cpu")
    monkeypatch.setattr(backend, "_resolve_dtype", lambda device: dtype)
    from_pretrained = staticmethod(
        lambda repo, **kwargs: calls.append(("model", repo, kwargs)) or loaded_model
    )
    fake_transformers = SimpleNamespace(
        AutoProcessor=SimpleNamespace(
            from_pretrained=lambda repo, **kwargs: (
                calls.append(("processor", repo, kwargs)) or processor
            )
        ),
        SeamlessM4Tv2ForSpeechToText=SimpleNamespace(from_pretrained=from_pretrained),
        Wav2Vec2ForCTC=SimpleNamespace(from_pretrained=from_pretrained),
        pipeline=lambda *args, **kwargs: object(),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    backend.load()

    hub_options = {"revision": backend.spec.revision, "local_files_only": True}
    assert calls[:2] == [
        ("processor", backend.spec.repo, hub_options),
        ("model", backend.spec.repo, {"dtype": dtype, **hub_options}),
    ]
    if model_name == "mms-1b-all":
        assert calls[2:] == [
            ("language", "mya"),
            ("adapter", "mya", hub_options),
        ]


def test_hf_cache_probe_uses_only_the_pinned_snapshot(monkeypatch, tmp_path):
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    backend = get_backend("hf")("whisper-large-v3")
    snapshots = tmp_path / "models--openai--whisper-large-v3" / "snapshots"
    snapshots.mkdir(parents=True)
    (snapshots / "incomplete").write_text("partial")
    assert backend.weights_cached() is False

    for revision in ("another-complete-revision", backend.spec.revision):
        complete = snapshots / revision
        complete.mkdir()
        (complete / "config.json").write_text("{}")
        (complete / "model.safetensors").write_text("weights")
        (complete / "preprocessor_config.json").write_text("{}")
        (complete / "tokenizer.json").write_text("{}")
        assert backend.weights_cached() is (revision == backend.spec.revision)


def test_hf_cache_probe_requires_every_indexed_shard(monkeypatch, tmp_path):
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    backend = get_backend("hf")("whisper-large-v3")
    snapshot = tmp_path / "models--openai--whisper-large-v3" / "snapshots" / backend.spec.revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "preprocessor_config.json").write_text("{}")
    (snapshot / "tokenizer.json").write_text("{}")
    shards = {"layer.0": "model-00001.safetensors", "layer.1": "model-00002.safetensors"}
    (snapshot / "model.safetensors.index.json").write_text(json.dumps({"weight_map": shards}))
    (snapshot / "model-00001.safetensors").write_text("partial")
    assert backend.weights_cached() is False

    (snapshot / "model-00002.safetensors").write_text("complete")
    assert backend.weights_cached() is True


def test_mms_cache_probe_requires_burmese_adapter_and_top_level_vocab(monkeypatch, tmp_path):
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    backend = get_backend("hf")("mms-1b-all")
    snapshot = tmp_path / "models--facebook--mms-1b-all" / "snapshots" / backend.spec.revision
    snapshot.mkdir(parents=True)
    for filename in (
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "adapter.mya.safetensors",
    ):
        (snapshot / filename).write_text("cached")
    assert backend.weights_cached() is False

    (snapshot / "vocab.json").write_text("{}")
    assert backend.weights_cached() is True


def test_dolphin_cache_probe_requires_checkpoint_and_sidecars(monkeypatch, tmp_path):
    from stt.backends import dolphin

    monkeypatch.setattr(dolphin, "CACHE_ROOT", tmp_path)
    backend = get_backend("dolphin")("small")
    directory = tmp_path / "small"
    directory.mkdir()
    (directory / "small.pt").write_text("checkpoint")
    assert backend.weights_cached() is False

    for filename in ("train.yaml", "feats_stats.npz", "units.txt", "bpe.model"):
        (directory / filename).write_text("sidecar")
    assert backend.weights_cached() is False

    monkeypatch.setattr(
        dolphin,
        "_sha256",
        lambda path: dict(backend.spec.artifacts)[path.name],
    )
    assert backend.weights_cached() is True


@pytest.mark.parametrize(
    ("size", "repo", "revision"),
    [
        ("base", "DataoceanAI/dolphin-base", "4f498c42abc03065b6a8b088800d08eb342b6e35"),
        ("small", "DataoceanAI/dolphin-small", "1df1f4f848fa1ad4bfe1fbdd1603ae45b5afb2ae"),
    ],
)
def test_dolphin_download_is_pinned_allowlisted_and_validated(
    monkeypatch, tmp_path, size, repo, revision
):
    import huggingface_hub

    from stt.backends import dolphin

    backend = get_backend("dolphin")(size)
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        directory = Path(kwargs["local_dir"])
        directory.mkdir(parents=True)
        for filename in kwargs["allow_patterns"]:
            (directory / filename).write_bytes(b"invalid")

    monkeypatch.setattr(dolphin, "CACHE_ROOT", tmp_path)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)

    with pytest.raises(RuntimeError, match="failed integrity validation") as exc_info:
        backend.download_weights()

    assert all(filename in str(exc_info.value) for filename, _ in backend.spec.artifacts)
    assert calls == [
        {
            "repo_id": repo,
            "revision": revision,
            "local_dir": tmp_path / size,
            "allow_patterns": [filename for filename, _ in backend.spec.artifacts],
        }
    ]


def test_dolphin_load_rejects_invalid_cache_before_upstream(monkeypatch, tmp_path):
    from stt.backends import dolphin

    backend = get_backend("dolphin")("small")
    directory = tmp_path / "small"
    directory.mkdir()
    for filename, _ in backend.spec.artifacts:
        (directory / filename).write_bytes(b"invalid")

    monkeypatch.setattr(dolphin, "CACHE_ROOT", tmp_path)
    monkeypatch.setitem(
        sys.modules,
        "dolphin",
        SimpleNamespace(
            load_model=lambda *args, **kwargs: pytest.fail("upstream load/download path ran")
        ),
    )

    with pytest.raises(RuntimeError, match=r"stt models --backend dolphin --download small"):
        backend.load()


def test_dolphin_load_downloads_a_wholly_absent_cache(monkeypatch, tmp_path):
    from stt.backends import dolphin

    backend = get_backend("dolphin")("small")
    downloads = []
    upstream = SimpleNamespace(load_model=lambda *args, **kwargs: object())
    monkeypatch.setitem(sys.modules, "dolphin", upstream)

    def download():
        downloads.append(backend.model)
        directory = tmp_path / backend.model
        directory.mkdir()
        for filename, _ in backend.spec.artifacts:
            (directory / filename).write_bytes(b"verified by the test hash stub")

    monkeypatch.setattr(dolphin, "CACHE_ROOT", tmp_path)
    monkeypatch.setattr(
        dolphin,
        "_sha256",
        lambda path: dict(backend.spec.artifacts)[path.name],
    )
    monkeypatch.setattr(backend, "_resolve_device", lambda: "cpu")
    monkeypatch.setattr(backend, "download_weights", download)

    backend.load()

    assert downloads == ["small"]
    assert backend._loaded is True
