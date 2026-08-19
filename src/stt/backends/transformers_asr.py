"""Hugging Face ``transformers`` ASR runtime, accelerated with Metal (MPS).

This is the GPU path for everything that is not omniASR. It covers four model
families behind one interface:

``whisper``    Burmese fine-tunes of Whisper, plus stock Whisper as a baseline.
``mms``        Meta MMS-1B with its per-language adapter (``mya``).
``ctc``        Monolingual Burmese CTC fine-tunes (w2v2-BERT).
``seamless``   SeamlessM4T v2, which transcribes Burmese speech but cannot
               synthesise it — source-side only.

Unlike fairseq2, ``transformers`` runs cleanly on MPS, so these models use the
GPU on Apple Silicon. That does not make them *better* than omniASR: every
Burmese Whisper fine-tune on the Hub was trained on a small read-speech corpus.
See ``docs/model-survey.md`` for what the published numbers actually say.

Licensing: MMS and SeamlessM4T weights are CC-BY-NC-4.0. Fine for evaluation,
not for a commercial product.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from stt.audio import join_segments
from stt.backends.base import ASRBackend
from stt.registry import register
from stt.results import Segment, TranscriptionResult

#: Longest window fed to the model in one go, and how much neighbouring context
#: overlaps it. Whisper was trained on 30 s windows; CTC models have no such
#: constraint but degrade on very long inputs and cost quadratic attention.
_CHUNKING = {
    "whisper": (30.0, 5.0),
    "mms": (20.0, 3.0),
    "ctc": (20.0, 3.0),
    "seamless": (20.0, 0.0),
}


@dataclass(frozen=True)
class HFModel:
    repo: str
    family: str
    approx_mb: int
    #: Language code this family expects, given a Burmese request.
    lang: str | None
    note: str = ""


MODELS: dict[str, HFModel] = {
    "whisper-my-large-v3": HFModel(
        "chuuhtetnaing/whisper-large-v3-myanmar",
        "whisper",
        6174,
        "my",
        "Burmese fine-tune; card reports 54.9 WER on its own eval",
    ),
    "whisper-my-medium": HFModel(
        "chuuhtetnaing/whisper-medium-myanmar",
        "whisper",
        3056,
        "my",
        "same corpus as the large fine-tune",
    ),
    "whisper-my-small": HFModel(
        "chuuhtetnaing/whisper-small-myanmar",
        "whisper",
        967,
        "my",
        "same corpus as the large fine-tune",
    ),
    "whisper-large-v3": HFModel(
        "openai/whisper-large-v3",
        "whisper",
        3087,
        "my",
        "stock Whisper; Burmese is below OpenAI's published quality bar",
    ),
    "mms-1b-all": HFModel(
        "facebook/mms-1b-all",
        "mms",
        3869,
        "mya",
        "CC-BY-NC; very fast; 37.9 WER on third-party medical data",
    ),
    "seamless-m4t-v2": HFModel(
        "facebook/seamless-m4t-v2-large",
        "seamless",
        9237,
        "mya",
        "CC-BY-NC; Burmese is source-speech only",
    ),
    "w2v-bert-my": HFModel(
        "YonaKhine/finetuned-w2v2-bert-burmese-asr",
        "ctc",
        2423,
        None,
        "monolingual Burmese CTC fine-tune; no published score",
    ),
}

# Chosen on measured CER over FLEURS Burmese (see README), not on reputation:
# Seamless beat every Whisper fine-tune by a factor of six. Note its weights
# are CC-BY-NC-4.0 -- fine for evaluation, not for a commercial product.
DEFAULT_MODEL = "seamless-m4t-v2"


@register
class TransformersASRBackend(ASRBackend):
    name: ClassVar[str] = "hf"
    description: ClassVar[str] = (
        "Hugging Face transformers: Whisper / MMS / SeamlessM4T / w2v-BERT (Metal GPU)"
    )
    install_hint: ClassVar[str] = "uv sync --extra hf"
    accepts_language: ClassVar[bool] = True
    is_local: ClassVar[bool] = True

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: str = "auto",
        dtype: str = "auto",
        **options: Any,
    ) -> None:
        if model not in MODELS:
            known = ", ".join(sorted(MODELS))
            raise ValueError(f"Unknown HF model {model!r}. Available: {known}")
        super().__init__(model, **options)
        self.spec = MODELS[model]
        self.device_arg = device
        self.dtype_arg = dtype
        self.pipe: Any = None
        self._seamless: tuple[Any, Any] | None = None
        self.resolved_device: str | None = None
        self.resolved_dtype: str | None = None

    # ------------------------------------------------------------------ setup

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        try:
            import torch
            import transformers
        except ImportError as exc:
            return False, f"missing dependency: {exc.name}"
        gpu = "mps" if torch.backends.mps.is_available() else "cpu"
        return True, f"transformers {transformers.__version__} on {gpu}"

    def _resolve_device(self) -> str:
        import torch

        if self.device_arg != "auto":
            return self.device_arg
        if torch.cuda.is_available():
            return "cuda"
        # Unlike fairseq2, transformers has a working MPS path for these
        # architectures, so Metal is the sensible default on Apple Silicon.
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _resolve_dtype(self, device: str) -> Any:
        import torch

        named = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        if self.dtype_arg != "auto":
            if self.dtype_arg not in named:
                raise ValueError(f"Unknown dtype {self.dtype_arg!r}. Choose from {sorted(named)}")
            return named[self.dtype_arg]
        # float32 everywhere by default, and measurement backs this up rather
        # than mere caution. On five FLEURS clips with SeamlessM4T v2 on Metal:
        #
        #   float32   RTF 0.16   CER 0.0420
        #   float16   RTF 0.26   CER 0.0455   (4/5 transcripts differ)
        #   bfloat16  RTF 0.26   CER 0.0420   (4/5 transcripts differ)
        #
        # Half precision is both slower *and* no more accurate here, so it buys
        # nothing but GPU memory. That is the opposite of omniASR's LLM decoder,
        # which is matmul-bound and gains from float16 — the best dtype is a
        # property of the model as much as of the chip, so this backend does not
        # share omniASR's probe.
        return torch.float32

    def estimated_download_mb(self) -> int | None:
        return self.spec.approx_mb

    def weights_cached(self) -> bool | None:
        from huggingface_hub import constants

        cache = Path(constants.HF_HUB_CACHE)
        folder = cache / f"models--{self.spec.repo.replace('/', '--')}"
        if not folder.is_dir():
            return False
        snapshots = folder / "snapshots"
        return any(snapshots.iterdir()) if snapshots.is_dir() else False

    def load(self) -> None:
        from transformers import AutoProcessor, pipeline

        device = self._resolve_device()
        dtype = self._resolve_dtype(device)
        self.resolved_device = device
        self.resolved_dtype = str(dtype).replace("torch.", "")

        repo = self.spec.repo

        if self.spec.family == "seamless":
            from transformers import SeamlessM4Tv2ForSpeechToText

            processor = AutoProcessor.from_pretrained(repo)
            model = SeamlessM4Tv2ForSpeechToText.from_pretrained(repo, torch_dtype=dtype)
            model.to(device).eval()
            self._seamless = (processor, model)
            self._loaded = True
            return

        if self.spec.family == "mms":
            from transformers import Wav2Vec2ForCTC

            processor = AutoProcessor.from_pretrained(repo)
            model = Wav2Vec2ForCTC.from_pretrained(repo, torch_dtype=dtype)
            # MMS is one shared encoder plus a tiny per-language adapter; both
            # the tokenizer and the model have to be switched to Burmese or you
            # silently decode with the previous language's vocabulary.
            processor.tokenizer.set_target_lang(self.spec.lang)
            model.load_adapter(self.spec.lang)
            model.to(device).eval()
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=dtype,
                device=device,
            )
            self._loaded = True
            return

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=repo,
            torch_dtype=dtype,
            device=device,
        )
        self._loaded = True

    def unload(self) -> None:
        self.pipe = None
        self._seamless = None
        self._loaded = False
        import gc

        gc.collect()
        try:
            import torch

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:  # noqa: BLE001 - cache clearing is best-effort
            pass

    # ------------------------------------------------------------- inference

    def _transcribe_seamless(self, path: Path) -> list[Segment]:
        """SeamlessM4T has no ASR pipeline, so window the audio by hand."""
        import numpy as np
        import soundfile as sf
        import torch

        from stt.audio import windowed

        assert self._seamless is not None
        processor, model = self._seamless
        window, _ = _CHUNKING["seamless"]

        pcm, rate = sf.read(str(path), dtype="float32", always_2d=False)
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=1)

        def decode(chunk: np.ndarray) -> str:
            inputs = processor(
                audios=np.asarray(chunk), sampling_rate=rate, return_tensors="pt"
            ).to(model.device, model.dtype)
            with torch.inference_mode():
                tokens = model.generate(**inputs, tgt_lang=self.spec.lang)
            return processor.decode(tokens[0].tolist(), skip_special_tokens=True).strip()

        return windowed(pcm, rate, window, decode)

    def _transcribe_pipeline(self, path: Path) -> tuple[str, list[Segment]]:
        kwargs: dict[str, Any]
        if self.spec.family == "whisper":
            # Deliberately no chunk_length_s. Whisper carries its own sequential
            # long-form algorithm that conditions each 30 s window on the
            # previous one; the pipeline's naive chunking discards that context
            # and transformers warns it is less accurate. return_timestamps is
            # what switches the sequential path on.
            kwargs = {
                "return_timestamps": True,
                "generate_kwargs": {"language": self.spec.lang, "task": "transcribe"},
            }
        else:
            # CTC models have no cross-window state, so fixed windows with an
            # overlapping stride are the correct approach here.
            window, stride = _CHUNKING[self.spec.family]
            kwargs = {"chunk_length_s": window, "stride_length_s": stride}
        out = self.pipe(str(path), **kwargs)
        if not isinstance(out, dict):
            return str(out).strip(), []
        text = str(out.get("text", "")).strip()

        # Whisper's sequential long-form path returns per-window timestamps.
        # CTC families run without them, so `chunks` is simply absent there.
        segments: list[Segment] = []
        for chunk in out.get("chunks") or []:
            stamp = chunk.get("timestamp") or (None, None)
            piece = str(chunk.get("text", "")).strip()
            # The final chunk's end is None when the decoder hit the audio end.
            if not piece or stamp[0] is None:
                continue
            segments.append(
                Segment(
                    text=piece,
                    start=float(stamp[0]),
                    end=float(stamp[1] if stamp[1] is not None else stamp[0]),
                    source="native",
                )
            )
        return text, segments

    def transcribe(
        self,
        audio_paths: list[Path],
        language: str | None = None,
        batch_size: int = 1,
    ) -> list[TranscriptionResult]:
        if not self._loaded:
            self.load()

        from stt.audio import duration_of

        results: list[TranscriptionResult] = []
        meta = {
            "repo": self.spec.repo,
            "family": self.spec.family,
            "device": self.resolved_device,
            "dtype": self.resolved_dtype,
        }

        for path in audio_paths:
            duration = None
            try:
                self._check_language(language)
                duration = duration_of(path)
                started = time.perf_counter()
                if self.spec.family == "seamless":
                    segments = self._transcribe_seamless(path)
                    text = join_segments(segments)
                else:
                    text, segments = self._transcribe_pipeline(path)
                elapsed = time.perf_counter() - started

                results.append(
                    TranscriptionResult(
                        audio_path=str(path),
                        text=text,
                        backend=self.name,
                        model=self.model,
                        language=language,
                        elapsed_s=elapsed,
                        audio_duration_s=duration,
                        metadata=dict(meta),
                        segments=segments or None,
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
                        metadata=dict(meta),
                    )
                )

        return results

    def _check_language(self, language: str | None) -> None:
        """These models are Burmese-only here; reject anything else loudly.

        A wrong code would otherwise decode silently in the wrong language,
        which is far harder to spot than an exception.
        """
        if language is None:
            return
        if language.split("_")[0] not in {"mya", "my"}:
            raise ValueError(
                f"The {self.name} backend is configured for Burmese only; got {language!r}."
            )
