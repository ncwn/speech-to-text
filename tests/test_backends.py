"""Tests for the backend contract and registry.

These run without either heavy runtime installed — that is the point of
keeping optional imports inside methods.
"""

import pytest

from stt.backends.base import ASRBackend
from stt.registry import all_backends, get_backend
from stt.results import TranscriptionResult, read_jsonl, write_jsonl


def test_both_backends_are_registered():
    names = set(all_backends())
    assert {"omniasr-torch", "omniasr-gguf"} <= names


def test_registry_rejects_unknown_name():
    with pytest.raises(KeyError, match="Unknown backend"):
        get_backend("does-not-exist")


@pytest.mark.parametrize("name", ["omniasr-torch", "omniasr-gguf"])
def test_backend_declares_required_metadata(name):
    cls = get_backend(name)
    assert issubclass(cls, ASRBackend)
    assert cls.name == name
    assert cls.description
    assert cls.install_hint
    ok, reason = cls.is_available()
    assert isinstance(ok, bool)
    assert isinstance(reason, str)


def test_gguf_backend_rejects_unknown_model():
    cls = get_backend("omniasr-gguf")
    with pytest.raises(ValueError, match="Unknown GGUF model"):
        cls("not-a-real-model")


def test_gguf_model_table_is_coherent():
    from stt.backends.omniasr_gguf import DEFAULT_MODEL, MODELS

    assert DEFAULT_MODEL in MODELS
    assert MODELS[DEFAULT_MODEL].unlimited, "default should handle long audio"
    for key, spec in MODELS.items():
        assert spec.url.endswith(spec.filename), f"{key}: url/filename mismatch"
        assert spec.approx_mb > 0


def test_torch_backend_identifies_unlimited_cards():
    cls = get_backend("omniasr-torch")
    assert cls("omniASR_LLM_Unlimited_7B_v2").is_unlimited
    assert not cls("omniASR_LLM_7B_v2").is_unlimited


def test_torch_backend_parses_model_size():
    cls = get_backend("omniasr-torch")
    assert cls("omniASR_LLM_Unlimited_7B_v2")._model_size_tag() == "7B"
    assert cls("omniASR_LLM_Unlimited_300M_v2")._model_size_tag() == "300M"


def test_torch_backend_reports_download_size():
    cls = get_backend("omniasr-torch")
    assert cls("omniASR_LLM_Unlimited_7B_v2").estimated_download_mb() == 31200
    assert cls("some_unrecognised_card").estimated_download_mb() is None


def test_cpu_prefers_float32_because_bfloat16_is_emulated_there():
    """Measured on the 7B: bf16 on CPU is RTF 8.85 against fp32's 2.14, for
    identical text. PyTorch has no native half kernels on CPU, so the only
    reason to pick bf16 is not fitting in memory."""
    torch = pytest.importorskip("torch")
    cls = get_backend("omniasr-torch")

    assert cls("omniASR_LLM_Unlimited_300M_v2")._resolve_dtype("cpu") is torch.float32
    large = cls("omniASR_LLM_Unlimited_7B_v2")
    expected = torch.float32 if large._has_headroom_for_float32() else torch.bfloat16
    assert large._resolve_dtype("cpu") is expected


def test_a_large_card_falls_back_to_bfloat16_on_a_small_machine():
    """fp32 for the big cards needs ~34 GB resident; bf16 is the fit-in-RAM path."""
    torch = pytest.importorskip("torch")
    cls = get_backend("omniasr-torch")
    large = cls("omniASR_LLM_Unlimited_7B_v2")
    large._has_headroom_for_float32 = lambda *a, **k: False
    assert large._resolve_dtype("cpu") is torch.bfloat16


def test_gpu_dtype_follows_the_measured_probe_not_a_constant():
    """Which 16-bit format is fast is a property of the GPU. An M2 Max runs
    float16 at 12,306 GFLOP/s and bfloat16 at 5,797; a later chip may invert
    that, so the backend asks rather than hardcoding either one."""
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("no Metal device")
    from stt.hardware import fastest_dtype

    cls = get_backend("omniasr-torch")
    resolved = cls("omniASR_LLM_Unlimited_7B_v2")._resolve_dtype("mps")
    assert resolved is getattr(torch, fastest_dtype("mps"))
    assert resolved in {torch.float16, torch.bfloat16}


def test_explicit_dtype_overrides_auto():
    torch = pytest.importorskip("torch")
    cls = get_backend("omniasr-torch")
    assert (
        cls("omniASR_LLM_Unlimited_7B_v2", dtype="float32")._resolve_dtype("cpu") is torch.float32
    )


def test_unknown_dtype_is_rejected():
    pytest.importorskip("torch")
    cls = get_backend("omniasr-torch")
    with pytest.raises(ValueError, match="Unknown dtype"):
        cls("omniASR_LLM_Unlimited_7B_v2", dtype="float8")._resolve_dtype("cpu")


def test_metal_is_auto_selected_when_available():
    """Metal was measured on the 7B: identical text to CPU, RTF 0.70 against
    8.85 at the same dtype, and 13.9 GB against 22.3 GB. Faster and smaller
    with no change in output, so auto should take it."""
    torch = pytest.importorskip("torch")
    cls = get_backend("omniasr-torch")
    resolved = cls("omniASR_LLM_Unlimited_7B_v2")._resolve_device()
    if torch.cuda.is_available():
        assert resolved == "cuda"
    elif torch.backends.mps.is_available():
        assert resolved == "mps"
    else:
        assert resolved == "cpu"


def test_an_explicit_device_overrides_auto():
    pytest.importorskip("torch")
    cls = get_backend("omniasr-torch")
    assert cls("omniASR_LLM_Unlimited_7B_v2", device="cpu")._resolve_device() == "cpu"
    assert cls("omniASR_LLM_Unlimited_7B_v2", device="mps")._resolve_device() == "mps"


def test_rtf_is_none_without_timings():
    r = TranscriptionResult(audio_path="a.wav", text="x", backend="b", model="m")
    assert r.rtf is None


def test_rtf_computes_from_timings():
    r = TranscriptionResult(
        audio_path="a.wav",
        text="x",
        backend="b",
        model="m",
        elapsed_s=2.0,
        audio_duration_s=10.0,
    )
    assert r.rtf == 0.2


def test_jsonl_roundtrip_preserves_burmese(tmp_path):
    original = [
        TranscriptionResult(
            audio_path="a.wav",
            text="မြန်မာဘာသာစကား",
            backend="b",
            model="m",
            language="mya_Mymr",
            elapsed_s=1.0,
            audio_duration_s=5.0,
            metadata={"device": "cpu"},
        )
    ]
    path = tmp_path / "out.jsonl"
    write_jsonl(original, path)
    restored = read_jsonl(path)

    assert restored[0].text == original[0].text
    assert restored[0].metadata == {"device": "cpu"}
    assert restored[0].rtf == 0.2


# ------------------------------------------------------- transformers backend


def test_new_backends_are_registered():
    names = set(all_backends())
    assert {"hf", "dolphin"} <= names


def test_hf_backend_rejects_unknown_model():
    cls = get_backend("hf")
    with pytest.raises(ValueError, match="Unknown HF model"):
        cls("not-a-real-model")


def test_hf_model_table_is_coherent():
    from stt.backends.transformers_asr import _CHUNKING, DEFAULT_MODEL, MODELS

    assert DEFAULT_MODEL in MODELS
    for key, spec in MODELS.items():
        assert "/" in spec.repo, f"{key}: repo should be owner/name"
        assert spec.approx_mb > 0
        assert spec.family in _CHUNKING, f"{key}: no chunking policy for {spec.family}"


def test_hf_language_codes_match_each_family():
    """Each family spells Burmese differently; a wrong code decodes silently."""
    from stt.backends.transformers_asr import MODELS

    assert MODELS["whisper-my-large-v3"].lang == "my"  # Whisper's ISO-639-1
    assert MODELS["mms-1b-all"].lang == "mya"  # MMS adapter name
    assert MODELS["seamless-m4t-v2"].lang == "mya"  # Seamless tgt_lang
    assert MODELS["w2v-bert-my"].lang is None  # monolingual, no code needed


def test_hf_prefers_metal_over_cpu():
    torch = pytest.importorskip("torch")
    cls = get_backend("hf")
    resolved = cls("whisper-my-small")._resolve_device()
    if torch.backends.mps.is_available():
        assert resolved == "mps"
    else:
        assert resolved in {"cpu", "cuda"}


def test_hf_defaults_to_float32():
    """Half precision on MPS produces NaNs in some attention kernels."""
    torch = pytest.importorskip("torch")
    cls = get_backend("hf")
    assert cls("whisper-my-small")._resolve_dtype("mps") is torch.float32


def test_hf_uses_saturated_default_only_for_seamless_on_mps():
    cls = get_backend("hf")
    seamless = cls("seamless-m4t-v2")
    seamless.resolved_device = "mps"
    mms = cls("mms-1b-all")
    mms.resolved_device = "mps"
    cpu = cls("seamless-m4t-v2")
    cpu.resolved_device = "cpu"

    assert seamless.preferred_batch_size() == 32
    assert mms.preferred_batch_size() == 1
    assert cpu.preferred_batch_size() == 1


@pytest.mark.parametrize(
    ("version", "dtype_key", "audio_key"),
    [
        ("4.45.0", "torch_dtype", "audios"),
        ("4.56.0", "dtype", "audios"),
        ("4.57.6", "dtype", "audio"),
    ],
)
def test_hf_uses_version_appropriate_transformers_kwargs(
    monkeypatch, version, dtype_key, audio_key
):
    """Keep the declared Transformers 4.45+ range warning-free and usable."""
    import sys
    from types import SimpleNamespace

    from stt.backends import transformers_asr

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(__version__=version))
    assert transformers_asr._dtype_kwargs("sentinel") == {dtype_key: "sentinel"}
    assert transformers_asr._audio_kwarg("sentinel") == {audio_key: "sentinel"}


def test_hf_rejects_non_burmese_language():
    cls = get_backend("hf")
    backend = cls("whisper-my-small")
    backend._check_language("mya_Mymr")  # accepted
    backend._check_language(None)  # accepted
    with pytest.raises(ValueError, match="Burmese only"):
        backend._check_language("tha_Thai")


# ------------------------------------------------------------ dolphin backend


def test_dolphin_backend_rejects_unknown_model():
    cls = get_backend("dolphin")
    with pytest.raises(ValueError, match="Unknown Dolphin model"):
        cls("large")  # real in the paper, never publicly released


def test_dolphin_model_table_is_coherent():
    from stt.backends.dolphin import DEFAULT_MODEL, MODELS

    assert DEFAULT_MODEL in MODELS
    for spec in MODELS.values():
        assert spec.params_m > 0
        assert spec.approx_mb > 0


def test_dolphin_uses_the_two_level_language_scheme():
    """Dolphin needs a language *and* a region token, not one combined code."""
    from stt.backends.dolphin import BURMESE_LANG, BURMESE_REGION

    assert BURMESE_LANG == "my"
    assert BURMESE_REGION == "MM"


def test_dolphin_gives_each_size_its_own_cache_dir():
    """Dolphin writes a shared config.yaml next to the weights and skips files
    that already exist, so one directory per size is the only safe layout."""
    from stt.backends.dolphin import MODELS, cache_dir

    dirs = {size: cache_dir(size) for size in MODELS}
    assert len(set(dirs.values())) == len(MODELS), "sizes must not share a directory"
    for size, path in dirs.items():
        assert path.name == size


def test_dolphin_rejects_non_burmese_language():
    cls = get_backend("dolphin")
    backend = cls("small")
    backend._check_language("mya_Mymr")
    with pytest.raises(ValueError, match="Burmese"):
        backend._check_language("zho_Hans")


# ------------------------------------------------------ silence-aware chunking


def test_split_on_quiet_returns_whole_clip_when_short():
    import numpy as np

    from stt.audio import split_on_quiet

    pcm = np.ones(8000, dtype="float32")
    assert split_on_quiet(pcm, 16000, window_s=30.0) == [(0, 8000)]


def test_split_on_quiet_covers_the_input_without_gaps_or_overlap():
    import numpy as np

    from stt.audio import split_on_quiet

    rng = np.random.default_rng(0)
    pcm = rng.standard_normal(16000 * 95).astype("float32")
    bounds = split_on_quiet(pcm, 16000, window_s=20.0)

    assert bounds[0][0] == 0
    assert bounds[-1][1] == len(pcm)
    for (_, prev_end), (next_start, _) in zip(bounds, bounds[1:], strict=False):
        assert prev_end == next_start, "windows must tile the input exactly"


def test_split_on_quiet_cuts_in_the_silence():
    """A boundary should land in the pause, not in the middle of the tone."""
    import numpy as np

    from stt.audio import split_on_quiet

    rate = 16000
    loud = np.ones(rate, dtype="float32")
    quiet = np.zeros(rate, dtype="float32")
    # 20 s of tone, 1 s of silence, then more tone. Nominal cut is at 20 s;
    # the silence sits at 20-21 s, within the default 2 s search radius.
    pcm = np.concatenate([np.tile(loud, 20), quiet, np.tile(loud, 10)])

    bounds = split_on_quiet(pcm, rate, window_s=20.0, search_s=2.0)
    first_cut = bounds[0][1]
    assert 20 * rate <= first_cut <= 21 * rate, f"cut at {first_cut / rate:.2f}s"


def test_split_on_quiet_still_splits_continuous_speech():
    """With no pause anywhere it must still tile, not return one huge window."""
    import numpy as np

    from stt.audio import split_on_quiet

    pcm = np.ones(16000 * 100, dtype="float32")
    bounds = split_on_quiet(pcm, 16000, window_s=20.0)
    assert len(bounds) >= 4
    assert max(end - start for start, end in bounds) <= 16000 * 23


def test_float64_buffers_are_demoted_for_metal():
    """Metal implements no float64 at all, so one such buffer makes a model
    unloadable. Dolphin has exactly two: the CMVN mean and standard deviation."""
    torch = pytest.importorskip("torch")
    from stt.backends.dolphin import _demote_float64

    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("mean", torch.zeros(4, dtype=torch.float64))
            self.register_buffer("std", torch.ones(4, dtype=torch.float64))
            self.register_buffer("ok", torch.ones(4, dtype=torch.float32))

    class Outer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cmvn = Inner()
            self.linear = torch.nn.Linear(4, 4)

    model = Outer()
    changed = _demote_float64(model)

    assert sorted(changed) == ["cmvn.mean", "cmvn.std"]
    assert all(b.dtype is not torch.float64 for b in model.buffers())
    assert model.cmvn.ok.dtype is torch.float32  # untouched
    assert model.linear.weight.dtype is torch.float32


def test_demoting_a_clean_model_changes_nothing():
    torch = pytest.importorskip("torch")
    from stt.backends.dolphin import _demote_float64

    model = torch.nn.Linear(2, 2)
    assert _demote_float64(model) == []


def test_demoted_values_survive_the_cast():
    """It is a compatibility cast, not a quantisation — the numbers must hold."""
    torch = pytest.importorskip("torch")
    from stt.backends.dolphin import _demote_float64

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("mean", torch.tensor([1.5, -2.25], dtype=torch.float64))

    model = M()
    _demote_float64(model)
    assert torch.allclose(model.mean, torch.tensor([1.5, -2.25]), atol=1e-6)
