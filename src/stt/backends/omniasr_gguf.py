"""Meta Omnilingual ASR via CrispASR's ggml runtime — Metal GPU on Apple Silicon.

CrispASR is a whisper.cpp/ggml fork with an ``omniasr-llm`` backend. It picks
the best available ggml backend at init (CUDA > Metal > Vulkan > CPU), so on a
Mac this runs on the GPU with no configuration.

The trade-off versus :mod:`stt.backends.omniasr_torch` is coverage: only the
300M and 1B LLM cards have been converted to GGUF. There is no 3B or 7B. Use
this backend to iterate quickly, then confirm on the 7B with the PyTorch one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from stt.audio import join_segments
from stt.backends.base import ASRBackend
from stt.native import suppress_native_output
from stt.registry import register
from stt.results import Segment, TranscriptionResult

_HF = "https://huggingface.co"


@dataclass(frozen=True)
class GgufModel:
    """A downloadable GGUF checkpoint and the CrispASR backend that reads it."""

    filename: str
    url: str
    crisp_backend: str
    approx_mb: int
    unlimited: bool


#: Every omniASR checkpoint currently available as GGUF. Keys are what the user
#: passes to ``-m``. Q4_K is the default quantisation; f16 variants are listed
#: where they exist, for checking how much the quantisation costs on Burmese.
MODELS: dict[str, GgufModel] = {
    "llm-unlimited-300m-v2": GgufModel(
        filename="omniasr-llm-unlimited-300m-v2-q4_k.gguf",
        url=f"{_HF}/cstr/omniasr-llm-unlimited-300m-v2-GGUF/resolve/main/omniasr-llm-unlimited-300m-v2-q4_k.gguf",
        crisp_backend="omniasr-llm-unlimited",
        approx_mb=1025,
        unlimited=True,
    ),
    "llm-unlimited-300m-v2-f16": GgufModel(
        filename="omniasr-llm-unlimited-300m-v2-f16.gguf",
        url=f"{_HF}/cstr/omniasr-llm-unlimited-300m-v2-GGUF/resolve/main/omniasr-llm-unlimited-300m-v2-f16.gguf",
        crisp_backend="omniasr-llm-unlimited",
        approx_mb=3113,
        unlimited=True,
    ),
    "llm-300m-v2": GgufModel(
        filename="omniasr-llm-300m-v2-q4_k.gguf",
        url=f"{_HF}/cstr/omniasr-llm-300m-v2-GGUF/resolve/main/omniasr-llm-300m-v2-q4_k.gguf",
        crisp_backend="omniasr-llm",
        approx_mb=1019,
        unlimited=False,
    ),
    "llm-1b": GgufModel(
        filename="omniasr-llm-1b-q4_k.gguf",
        url=f"{_HF}/cstr/omniasr-llm-1b-GGUF/resolve/main/omniasr-llm-1b-q4_k.gguf",
        crisp_backend="omniasr-llm",
        approx_mb=1376,
        unlimited=False,
    ),
    "ctc-1b-v2": GgufModel(
        filename="omniasr-ctc-1b-v2-q4_k.gguf",
        url=f"{_HF}/cstr/omniASR-CTC-1B-v2-GGUF/resolve/main/omniasr-ctc-1b-v2-q4_k.gguf",
        crisp_backend="omniasr",
        approx_mb=658,
        unlimited=False,
    ),
    "ctc-300m-v2": GgufModel(
        filename="omniasr-ctc-300m-v2-q4_k.gguf",
        url=f"{_HF}/cstr/omniASR-CTC-300M-v2-GGUF/resolve/main/omniasr-ctc-300m-v2-q4_k.gguf",
        crisp_backend="omniasr-300m",
        approx_mb=194,
        unlimited=False,
    ),
}

DEFAULT_MODEL = "llm-unlimited-300m-v2"


def _speech_confidence(no_speech_prob: float) -> float | None:
    """Turn CrispASR's no-speech probability into a confidence, or ``None``.

    The omniasr-llm backend does not compute this and reports ``-1.0`` as a
    sentinel. Passing that through as ``1 - (-1) = 2.0`` would hand downstream
    routing a confidence above 1, so an out-of-range value becomes "unknown".

    Even when it is real this measures how likely the span is to be *silence*,
    not how likely the transcript is to be right — a weaker signal than the
    posterior that forced alignment gives (see :mod:`stt.align`).
    """
    probability = float(no_speech_prob)
    if not 0.0 <= probability <= 1.0:
        return None
    return 1.0 - probability


@register
class OmniASRGgufBackend(ASRBackend):
    name: ClassVar[str] = "omniasr-gguf"
    description: ClassVar[str] = "Meta Omnilingual ASR via CrispASR/ggml (Metal GPU, 300M-1B only)"
    install_hint: ClassVar[str] = "uv sync --extra gguf"
    accepts_language: ClassVar[bool] = True
    is_local: ClassVar[bool] = True

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        n_threads: int | None = None,
        chunk_seconds: int = 0,
        verbose: bool = False,
        **options: Any,
    ) -> None:
        super().__init__(model, **options)
        if model not in MODELS:
            raise ValueError(f"Unknown GGUF model {model!r}. Available: {', '.join(MODELS)}")
        self.spec = MODELS[model]
        # Left to CrispASR unless asked for. Measured on this backend: 4, 8 and
        # 12 threads all give RTF 0.182, because the work is on the GPU and the
        # CPU sits at 0.05 cores. Picking a number here would be noise dressed
        # up as tuning.
        self.n_threads = n_threads
        # 0 lets CrispASR choose its own chunking for long audio.
        self.chunk_seconds = chunk_seconds
        # ggml logs every Metal kernel compile to fd 1/2; off unless asked for.
        self.verbose = verbose
        self.session = None
        self.model_path: str | None = None

    # ------------------------------------------------------------------ setup

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        try:
            import crispasr
        except ImportError:
            return False, "missing dependency: crispasr"
        return True, f"crispasr {getattr(crispasr, '__version__', '?')}"

    def estimated_download_mb(self) -> int | None:
        return self.spec.approx_mb

    def weights_cached(self) -> bool | None:
        from crispasr import cache_dir

        return (Path(cache_dir()) / self.spec.filename).exists()

    def ensure_weights(self, quiet: bool = False) -> Path:
        """Download the GGUF to ``~/.cache/crispasr`` if it is not already there."""
        from crispasr import cache_ensure_file

        path = cache_ensure_file(self.spec.filename, self.spec.url, quiet=quiet)
        if not path:
            raise RuntimeError(f"Failed to download {self.spec.filename} from {self.spec.url}")
        return Path(path)

    def load(self) -> None:
        from crispasr import Session

        # Download outside the suppression block so progress stays visible.
        self.model_path = str(self.ensure_weights())

        with suppress_native_output(not self.verbose) as log:
            try:
                options: dict[str, Any] = {"backend": self.spec.crisp_backend}
                if self.n_threads is not None:
                    options["n_threads"] = self.n_threads
                self.session = Session(self.model_path, **options)
            except Exception as exc:
                raise RuntimeError(f"Failed to open {self.spec.filename}:\n{log.tail()}") from exc
        self._loaded = True

    def unload(self) -> None:
        if self.session is not None:
            with suppress_native_output(not self.verbose):
                self.session.close()
            self.session = None
        self._loaded = False

    # ------------------------------------------------------------- inference

    @staticmethod
    def _read_pcm(path: Path):
        """Read a file as mono float32 at 16 kHz — the format ggml expects."""
        import numpy as np
        import soundfile as sf

        pcm, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
        pcm = pcm.mean(axis=1)  # downmix; CrispASR would otherwise take channel 0
        return np.ascontiguousarray(pcm), sample_rate

    def transcribe(
        self,
        audio_paths: list[Path],
        language: str | None = None,
        batch_size: int = 1,  # noqa: ARG002 - ggml runs one utterance at a time
    ) -> list[TranscriptionResult]:
        if self.session is None:
            self.load()

        results: list[TranscriptionResult] = []

        for path in audio_paths:
            duration = None
            try:
                pcm, sample_rate = self._read_pcm(path)
                duration = len(pcm) / sample_rate

                started = time.perf_counter()
                with suppress_native_output(not self.verbose) as log:
                    try:
                        if self.chunk_seconds or not self.spec.unlimited:
                            segments = self.session.transcribe_chunked(
                                pcm,
                                chunk_seconds=self.chunk_seconds,
                                sample_rate=sample_rate,
                                language=language,
                            )
                        else:
                            segments = self.session.transcribe(
                                pcm, sample_rate=sample_rate, language=language
                            )
                    except Exception as exc:
                        raise RuntimeError(f"{exc}\n{log.tail()}") from exc
                elapsed = time.perf_counter() - started

                # CrispASR already timed every segment, so keep the timings
                # rather than flattening the whole decode down to a string.
                timed = [
                    Segment(
                        text=s.text.strip(),
                        start=float(s.start),
                        end=float(s.end),
                        confidence=_speech_confidence(s.no_speech_prob),
                        source="native",
                    )
                    for s in segments
                    if s.text.strip()
                ]
                text = join_segments(timed)

                results.append(
                    TranscriptionResult(
                        audio_path=str(path),
                        text=text.strip(),
                        backend=self.name,
                        model=self.model,
                        language=language,
                        elapsed_s=elapsed,
                        audio_duration_s=duration,
                        metadata={
                            "gguf": self.spec.filename,
                            "crisp_backend": self.spec.crisp_backend,
                            "n_threads": self.n_threads,  # None means CrispASR chose
                            "n_segments": len(segments),
                        },
                        segments=timed or None,
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
                        metadata={"gguf": self.spec.filename},
                    )
                )

        return results
