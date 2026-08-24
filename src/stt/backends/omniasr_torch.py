"""Meta Omnilingual ASR via the official PyTorch/fairseq2 runtime.

This adapter accepts upstream model card names, including
``omniASR_LLM_Unlimited_7B_v2``.

Device ``auto`` selects CUDA, then MPS, then CPU. A failed MPS decode is
retried on CPU for the rest of the run.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from stt.backends.base import ASRBackend
from stt.registry import register
from stt.results import TranscriptionResult

#: Cards without "Unlimited" in the name reject audio longer than this.
MAX_LIMITED_AUDIO_SEC = 40

DEFAULT_MODEL = "omniASR_LLM_Unlimited_7B_v2"


@dataclass(frozen=True)
class OmniASRModel:
    family: str
    size: str
    size_bytes: int
    unlimited: bool = False

    @property
    def approx_mb(self) -> int:
        return round(self.size_bytes / 1_000_000)


#: Upstream v2 cards and exact checkpoint sizes reported by their origin server.
MODELS: dict[str, OmniASRModel] = {
    "omniASR_CTC_300M_v2": OmniASRModel("CTC", "300M", 1_304_065_508),
    "omniASR_CTC_1B_v2": OmniASRModel("CTC", "1B", 3_902_956_068),
    "omniASR_CTC_3B_v2": OmniASRModel("CTC", "3B", 12_325_920_624),
    "omniASR_CTC_7B_v2": OmniASRModel("CTC", "7B", 26_023_732_143),
    "omniASR_LLM_300M_v2": OmniASRModel("LLM", "300M", 6_526_183_880),
    "omniASR_LLM_1B_v2": OmniASRModel("LLM", "1B", 9_118_733_852),
    "omniASR_LLM_3B_v2": OmniASRModel("LLM", "3B", 17_522_679_843),
    "omniASR_LLM_7B_v2": OmniASRModel("LLM", "7B", 31_220_488_063),
    "omniASR_LLM_Unlimited_300M_v2": OmniASRModel("LLM", "300M", 6_526_216_648, True),
    "omniASR_LLM_Unlimited_1B_v2": OmniASRModel("LLM", "1B", 9_118_766_620, True),
    "omniASR_LLM_Unlimited_3B_v2": OmniASRModel("LLM", "3B", 17_522_712_611, True),
    "omniASR_LLM_Unlimited_7B_v2": OmniASRModel("LLM", "7B", 31_220_520_831, True),
}

TOKENIZER_SIZE_BYTES = 91_481
TOKENIZER_SHA256 = "8aa11a1092142ef472537476ef6e76541123e2f0d789b79f3ebd119008240b1e"
_ASSET_BASE = "https://dl.fbaipublicfiles.com/mms"
_TOKENIZER_FILENAME = "omniASR_tokenizer_written_v2.model"


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


@register
class OmniASRTorchBackend(ASRBackend):
    name: ClassVar[str] = "omniasr-torch"
    description: ClassVar[str] = "Meta Omnilingual ASR, official PyTorch/fairseq2 runtime"
    install_hint: ClassVar[str] = "uv sync --extra omniasr"
    supported_options: ClassVar[frozenset[str]] = frozenset({"device", "dtype"})

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: str = "auto",
        dtype: str = "auto",
        **options: Any,
    ) -> None:
        if model not in MODELS:
            known = ", ".join(MODELS)
            raise ValueError(f"Unknown omniASR model {model!r}. Available: {known}")
        super().__init__(model, **options)
        self.spec = MODELS[model]
        self.device_arg = device
        self.dtype_arg = dtype
        self.pipeline = None
        self.resolved_device: str | None = None
        self.resolved_dtype: str | None = None

    # ------------------------------------------------------------------ setup

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        try:
            import omnilingual_asr  # noqa: F401
            import torch
        except ImportError as exc:
            return False, f"missing dependency: {exc.name}"
        return True, f"torch {torch.__version__}"

    def _resolve_device(self) -> str:
        import torch

        if self.device_arg != "auto":
            return self.device_arg
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            # MPS is the locally validated default; transcribe falls back to CPU.
            return "mps"
        return "cpu"

    def _resolve_dtype(self, device: str) -> Any:
        import torch

        named = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
        }
        if self.dtype_arg != "auto":
            if self.dtype_arg not in named:
                raise ValueError(f"Unknown dtype {self.dtype_arg!r}. Choose from {sorted(named)}")
            return named[self.dtype_arg]

        if device != "cpu":
            # Probe the machine because the fastest 16-bit format varies by GPU.
            from stt.hardware import fastest_dtype

            return named[fastest_dtype(device)]

        # CPU float32 is the fast path; use bfloat16 only when memory is tight.
        if not self._has_headroom_for_float32():
            return torch.bfloat16
        return torch.float32

    #: Resident memory runs to roughly the checkpoint size plus activations and
    #: the buffers used to read it. Derived from the download size rather than
    #: from a per-card constant, so a new card needs no new number here.
    _FLOAT32_OVERHEAD = 1.5

    def _has_headroom_for_float32(self) -> bool:
        """Whether this machine can hold this particular card at float32.

        Asks how big the checkpoint is and how much memory the machine has,
        rather than comparing a model name against a fixed threshold — the same
        card is comfortable on a 64 GB machine and impossible on a 16 GB one.
        """
        from stt.hardware import total_ram_mb

        checkpoint_mb = self.spec.approx_mb
        ram_mb = total_ram_mb()
        if not checkpoint_mb or not ram_mb:
            return True  # unknown card or unknown machine: keep the fast path
        return ram_mb >= checkpoint_mb * self._FLOAT32_OVERHEAD

    @property
    def is_unlimited(self) -> bool:
        return self.spec.unlimited

    def estimated_download_mb(self) -> int | None:
        return self.spec.approx_mb

    def weights_cached(self) -> bool | None:
        root = Path(
            os.environ.get(
                "FAIRSEQ2_CACHE_DIR",
                Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
                / "fairseq2"
                / "assets",
            )
        )
        checkpoint_name = self.model.replace("_", "-") + ".pt"
        checkpoint = self._asset_path(root, checkpoint_name)
        tokenizer = self._asset_path(root, _TOKENIZER_FILENAME)
        return self._valid_checkpoint(checkpoint) and self._valid_tokenizer(tokenizer)

    @staticmethod
    def _asset_path(root: Path, filename: str) -> Path:
        uri = f"{_ASSET_BASE}/{filename}"
        directory = hashlib.sha1(uri.encode(), usedforsecurity=False).hexdigest()[:24]
        return root / directory / filename

    def _valid_checkpoint(self, path: Path) -> bool:
        try:
            return path.is_file() and path.stat().st_size == self.spec.size_bytes
        except OSError:
            return False

    @staticmethod
    def _valid_tokenizer(path: Path) -> bool:
        try:
            return (
                path.is_file()
                and path.stat().st_size == TOKENIZER_SIZE_BYTES
                and _sha256(path) == TOKENIZER_SHA256
            )
        except OSError:
            return False

    def download_weights(self) -> None:
        from fairseq2.assets import AssetDownloadManager, AssetStore
        from fairseq2.data.tokenizers.ref import resolve_tokenizer_reference
        from fairseq2.runtime.dependency import get_dependency_resolver

        resolver = get_dependency_resolver()
        store = resolver.resolve(AssetStore)
        manager = resolver.resolve(AssetDownloadManager)
        card = store.retrieve_card(self.model)
        checkpoint = Path(manager.download_model(card.field("checkpoint").as_uri(), card.name))
        tokenizer = resolve_tokenizer_reference(store, card)
        tokenizer_path = Path(
            manager.download_tokenizer(tokenizer.field("tokenizer").as_uri(), tokenizer.name)
        )

        invalid = []
        if not self._valid_checkpoint(checkpoint):
            invalid.append(f"checkpoint {checkpoint}")
        if not self._valid_tokenizer(tokenizer_path):
            invalid.append(f"tokenizer {tokenizer_path}")
        if invalid:
            raise RuntimeError(
                "omniASR download failed integrity validation: " + ", ".join(invalid)
            )

    def load(self) -> None:
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

        self.download_weights()
        device = self._resolve_device()
        dtype = self._resolve_dtype(device)

        self.resolved_device = device
        self.resolved_dtype = str(dtype).replace("torch.", "")

        self.pipeline = ASRInferencePipeline(model_card=self.model, device=device, dtype=dtype)
        self._loaded = True

    def unload(self) -> None:
        self.pipeline = None
        self._loaded = False
        import gc

        gc.collect()

    # ------------------------------------------------------------- inference

    @staticmethod
    def supported_languages() -> list[str]:
        from omnilingual_asr.models.wav2vec2_llama.lang_ids import supported_langs

        return list(supported_langs)

    def _check_language(self, language: str | None) -> None:
        if language is None:
            return
        supported = self.supported_languages()
        if language not in supported:
            near = [x for x in supported if x.split("_")[0].startswith(language.split("_")[0][:3])]
            hint = f" Did you mean one of: {', '.join(near[:5])}?" if near else ""
            raise ValueError(f"Language {language!r} is not supported by omniASR.{hint}")

    def _transcribe_one(self, path: str, lang_arg: list[str] | None, batch_size: int) -> list[str]:
        """Decode one file, retrying on CPU if Metal fails.

        An unsupported kernel would otherwise fail the run. The switch is
        permanent for this instance so a systematic failure does not repeat the
        MPS attempt for every remaining file.
        """
        assert self.pipeline is not None
        try:
            return self.pipeline.transcribe([path], lang=lang_arg, batch_size=batch_size)
        except Exception as exc:  # noqa: BLE001 - any Metal failure is worth retrying on CPU
            if self.resolved_device != "mps":
                raise
            import logging

            logging.getLogger(__name__).warning(
                "Metal failed (%s: %s); falling back to CPU for the rest of this run.",
                type(exc).__name__,
                exc,
            )
            self.unload()
            self.device_arg = "cpu"
            self.load()
            assert self.pipeline is not None
            return self.pipeline.transcribe([path], lang=lang_arg, batch_size=batch_size)

    def transcribe(
        self,
        audio_paths: list[Path],
        language: str | None = None,
        batch_size: int = 1,
    ) -> list[TranscriptionResult]:
        if self.pipeline is None:
            self.load()

        self._check_language(language)

        from stt.audio import duration_of

        results: list[TranscriptionResult] = []

        # Process one file at a time so a single failure does not lose the batch
        # and so per-file timings are meaningful. batch_size is still honoured
        # inside the pipeline call for machines that can exploit it.
        for path in audio_paths:
            duration = None
            try:
                duration = duration_of(path)
                if not self.is_unlimited and duration > MAX_LIMITED_AUDIO_SEC:
                    raise ValueError(
                        f"{path.name} is {duration:.1f}s but {self.model} accepts at most "
                        f"{MAX_LIMITED_AUDIO_SEC}s. Use an "
                        f"omniASR_LLM_Unlimited_*_v2 card for long audio."
                    )

                started = time.perf_counter()
                lang_arg = [language] if language else None
                texts = self._transcribe_one(str(path), lang_arg, batch_size)
                elapsed = time.perf_counter() - started

                results.append(
                    TranscriptionResult(
                        audio_path=str(path),
                        text=texts[0].strip() if texts else "",
                        backend=self.name,
                        model=self.model,
                        language=language,
                        elapsed_s=elapsed,
                        audio_duration_s=duration,
                        metadata={
                            "device": self.resolved_device,
                            "dtype": self.resolved_dtype,
                        },
                    )
                )
            except Exception as exc:  # noqa: BLE001 - one bad clip must not abort the sweep
                results.append(
                    TranscriptionResult(
                        audio_path=str(path),
                        text="",
                        backend=self.name,
                        model=self.model,
                        language=language,
                        audio_duration_s=duration,
                        error=f"{type(exc).__name__}: {exc}",
                        metadata={
                            "device": self.resolved_device,
                            "dtype": self.resolved_dtype,
                        },
                    )
                )

        return results
