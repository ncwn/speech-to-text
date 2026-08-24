"""Meta Omnilingual ASR via CrispASR's ggml runtime.

CrispASR is a whisper.cpp/ggml fork with an ``omniasr-llm`` backend. It picks
the best available ggml backend at init (CUDA > Metal > Vulkan > CPU), so on a
Mac this runs on the GPU with no configuration.

The model table below defines the GGUF conversions exposed by this adapter.
"""

from __future__ import annotations

import hashlib
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
    revision: str
    sha256: str
    crisp_backend: str
    approx_mb: int
    unlimited: bool


#: GGUF checkpoints exposed by this adapter. Keys are accepted by ``-m``.
MODELS: dict[str, GgufModel] = {
    "llm-unlimited-300m-v2": GgufModel(
        filename="omniasr-llm-unlimited-300m-v2-q4_k.gguf",
        url=(
            f"{_HF}/cstr/omniasr-llm-unlimited-300m-v2-GGUF/resolve/"
            "a68f5d040b483506c0966a1602050ee1550a0382/"
            "omniasr-llm-unlimited-300m-v2-q4_k.gguf"
        ),
        revision="a68f5d040b483506c0966a1602050ee1550a0382",
        sha256="56879dd6c0411aa19b225440aac088f11bad9496f64a1c0aa0d293e1450d7203",
        crisp_backend="omniasr-llm-unlimited",
        approx_mb=1075,
        unlimited=True,
    ),
    "llm-unlimited-300m-v2-f16": GgufModel(
        filename="omniasr-llm-unlimited-300m-v2-f16.gguf",
        url=(
            f"{_HF}/cstr/omniasr-llm-unlimited-300m-v2-GGUF/resolve/"
            "a68f5d040b483506c0966a1602050ee1550a0382/"
            "omniasr-llm-unlimited-300m-v2-f16.gguf"
        ),
        revision="a68f5d040b483506c0966a1602050ee1550a0382",
        sha256="e6e2220532ec7f76d0bdb0896109b2ab3d14bd46d639123a6212a57a932b3d8b",
        crisp_backend="omniasr-llm-unlimited",
        approx_mb=3264,
        unlimited=True,
    ),
    "llm-300m-v2": GgufModel(
        filename="omniasr-llm-300m-v2-q4_k.gguf",
        url=(
            f"{_HF}/cstr/omniasr-llm-300m-v2-GGUF/resolve/"
            "f77a5ccffff18e90f8e5605279103cec2b184b46/"
            "omniasr-llm-300m-v2-q4_k.gguf"
        ),
        revision="f77a5ccffff18e90f8e5605279103cec2b184b46",
        sha256="2039697e6d21d27a2394372c972de6f3439a0a74a3575876949130466d87f90c",
        crisp_backend="omniasr-llm",
        approx_mb=1068,
        unlimited=False,
    ),
    "llm-1b": GgufModel(
        filename="omniasr-llm-1b-q4_k.gguf",
        url=(
            f"{_HF}/cstr/omniasr-llm-1b-GGUF/resolve/"
            "7b433b6ab2b211c3cdde8591d8bb23642fe2742f/"
            "omniasr-llm-1b-q4_k.gguf"
        ),
        revision="7b433b6ab2b211c3cdde8591d8bb23642fe2742f",
        sha256="0181ce14efc1197222601c330035ccb7446119c4d8ffb0b0cdb526634c404fdf",
        crisp_backend="omniasr-llm",
        approx_mb=1442,
        unlimited=False,
    ),
    "ctc-1b-v2": GgufModel(
        filename="omniasr-ctc-1b-v2-q4_k.gguf",
        url=(
            f"{_HF}/cstr/omniASR-CTC-1B-v2-GGUF/resolve/"
            "317880194be65674e7b27efba10273be1afeb9f1/"
            "omniasr-ctc-1b-v2-q4_k.gguf"
        ),
        revision="317880194be65674e7b27efba10273be1afeb9f1",
        sha256="fcd75539c542f335877c04a83cbe6d9ccf31deae42ee72c655a8adefd6627036",
        crisp_backend="omniasr",
        approx_mb=691,
        unlimited=False,
    ),
    "ctc-300m-v2": GgufModel(
        filename="omniasr-ctc-300m-v2-q4_k.gguf",
        url=(
            f"{_HF}/cstr/omniASR-CTC-300M-v2-GGUF/resolve/"
            "fc3e3765175be5aaf0c9e75ed25a7e8843ef04d5/"
            "omniasr-ctc-300m-v2-q4_k.gguf"
        ),
        revision="fc3e3765175be5aaf0c9e75ed25a7e8843ef04d5",
        sha256="cac0ae5eef46f146e47a1445e9fb8a4e894fac78c5b90e4d6318dc3734b34808",
        crisp_backend="omniasr-300m",
        approx_mb=204,
        unlimited=False,
    ),
}

DEFAULT_MODEL = "llm-unlimited-300m-v2"


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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
    description: ClassVar[str] = "Meta Omnilingual ASR via CrispASR/ggml (selectable 300M/1B cards)"
    install_hint: ClassVar[str] = "uv sync --extra gguf"
    supported_options: ClassVar[frozenset[str]] = frozenset({"n_threads", "verbose"})

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
        # Leave thread selection to CrispASR unless explicitly overridden.
        self.n_threads = n_threads
        # Unlimited cards decode directly at 0; limited cards use chunked mode.
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

    def _cache_path(self) -> Path:
        from crispasr import cache_dir

        directory = cache_dir()
        if not directory:
            raise RuntimeError("CrispASR did not provide a cache directory")
        return Path(directory) / self.spec.filename

    def _weights_valid(self, path: Path) -> bool:
        try:
            return path.is_file() and _sha256(path) == self.spec.sha256
        except OSError:
            return False

    def weights_cached(self) -> bool | None:
        try:
            return self._weights_valid(self._cache_path())
        except RuntimeError:
            return False

    def download_weights(self) -> None:
        self.ensure_weights()

    def ensure_weights(self, quiet: bool = False) -> Path:
        """Return verified weights, downloading only when the model is absent."""
        from crispasr import cache_ensure_file

        cached = self._cache_path()
        if self._weights_valid(cached):
            return cached
        # The model is authoritative; a stale .src must not trigger replacement.
        if cached.exists() or cached.is_symlink():
            raise RuntimeError(
                f"GGUF cache failed integrity validation: {cached}. Quarantine the model and "
                f"its {cached.name}.src sidecar, then retry."
            )

        path = cache_ensure_file(self.spec.filename, self.spec.url, quiet=quiet)
        if not path:
            raise RuntimeError(f"Failed to download {self.spec.filename} from {self.spec.url}")
        downloaded = Path(path)
        if not self._weights_valid(downloaded):
            raise RuntimeError(f"GGUF download failed integrity validation: {downloaded}")
        return downloaded

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
        """Read mono float32 PCM; callers provide normalized 16 kHz audio."""
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
