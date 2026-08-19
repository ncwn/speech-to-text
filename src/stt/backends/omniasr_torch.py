"""Meta Omnilingual ASR via the official PyTorch/fairseq2 runtime.

This is the reference implementation — every model card Meta published works
here, including ``omniASR_LLM_Unlimited_7B_v2``, and it is the accuracy
ground truth the faster backends are measured against.

Apple Silicon note: there is no CUDA, and fairseq2 has no validated Metal path,
so this runs on CPU. That is slow but correct. Use the ``omniasr-gguf`` backend
when you want Metal acceleration and can accept a smaller model.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, ClassVar

from stt.backends.base import ASRBackend
from stt.registry import register
from stt.results import TranscriptionResult

#: Cards without "Unlimited" in the name reject audio longer than this.
MAX_LIMITED_AUDIO_SEC = 40

DEFAULT_MODEL = "omniASR_LLM_Unlimited_7B_v2"

#: Approximate fp32 checkpoint download size per card, in MB.
_DOWNLOAD_MB = {"300M": 6500, "1B": 9100, "3B": 17500, "7B": 31200}


@register
class OmniASRTorchBackend(ASRBackend):
    name: ClassVar[str] = "omniasr-torch"
    description: ClassVar[str] = (
        "Meta Omnilingual ASR, official PyTorch/fairseq2 runtime (CPU on macOS)"
    )
    install_hint: ClassVar[str] = "uv sync --extra omniasr"
    accepts_language: ClassVar[bool] = True
    is_local: ClassVar[bool] = True

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: str = "auto",
        dtype: str = "auto",
        **options: Any,
    ) -> None:
        super().__init__(model, **options)
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
            # Metal was measured, not assumed. On the 7B card, five FLEURS clips
            # decoded to text identical to CPU, at RTF 0.70 against 8.85 for the
            # same dtype on CPU and 2.14 for CPU's fastest dtype, using 13.9 GB
            # against 22.3 GB. Faster and smaller with no change in output, so
            # it is the default; `transcribe` falls back to CPU if it fails.
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
            # Which 16-bit format is fastest is a property of the GPU, not of
            # the model: an M2 Max runs float16 at 12,306 GFLOP/s and bfloat16
            # at 5,797, while later chips may invert that. Measured once per
            # machine and cached rather than assumed.
            from stt.hardware import fastest_dtype

            return named[fastest_dtype(device)]

        # On CPU, float32 is the fast path — PyTorch lacks native half-precision
        # kernels there and emulates them. Measured on the 7B: bfloat16 on CPU
        # runs at RTF 8.85 against float32's 2.14, a 4.1x penalty for identical
        # text. So float32 unless the machine cannot hold it: the large cards
        # need roughly 34 GB resident at float32, on top of the checkpoint read.
        size = self._model_size_tag()
        if size in {"3B", "7B"} and not self._has_headroom_for_float32():
            return torch.bfloat16
        return torch.float32

    @staticmethod
    def _has_headroom_for_float32(required_gb: int = 48) -> bool:
        """Whether this machine can hold a large card at float32 comfortably."""
        from stt.telemetry import describe_host

        ram_mb = describe_host().get("ram_mb")
        return bool(ram_mb and ram_mb >= required_gb * 1024)

    def _model_size_tag(self) -> str:
        for tag in ("300M", "1B", "3B", "7B"):
            if f"_{tag}_" in self.model or self.model.endswith(f"_{tag}"):
                return tag
        return "unknown"

    @property
    def is_unlimited(self) -> bool:
        return "Unlimited" in self.model

    def estimated_download_mb(self) -> int | None:
        return _DOWNLOAD_MB.get(self._model_size_tag())

    def weights_cached(self) -> bool | None:
        """Unknowable: fairseq2 stores assets under opaque content hashes.

        Returning None makes callers warn about the download rather than
        wrongly promising it is either cached or not.
        """
        return None

    def load(self) -> None:
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

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

        Metal is the measured-faster default, but fairseq2 does not test it and
        an unimplemented kernel would otherwise turn a slow run into a failed
        one. Falling back costs speed; not falling back costs the transcript.
        The switch is permanent for this instance, so a systematic failure does
        not pay the Metal attempt on every remaining file.
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
