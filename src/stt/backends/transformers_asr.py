"""Hugging Face ``transformers`` ASR runtime, accelerated with Metal (MPS).

This adapter covers four model families behind one interface:

``whisper``    Burmese fine-tunes of Whisper, plus stock Whisper as a baseline.
``mms``        Meta MMS-1B with its per-language adapter (``mya``).
``ctc``        Monolingual Burmese CTC fine-tunes (w2v2-BERT).
``seamless``   SeamlessM4T v2 with speech/text input and text output. This
               adapter does not expose speech output.

Device ``auto`` selects CUDA, then MPS, then CPU.

Licensing: MMS and SeamlessM4T weights are CC-BY-NC-4.0. Fine for evaluation,
not for a commercial product.
"""

from __future__ import annotations

import json
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
    revision: str
    family: str
    approx_mb: int
    #: Language code this family expects, given a Burmese request.
    lang: str | None
    note: str = ""


MODELS: dict[str, HFModel] = {
    "whisper-my-large-v3": HFModel(
        "chuuhtetnaing/whisper-large-v3-myanmar",
        "c6d3e92a45b561cb5c00724625ca1904f830d887",
        "whisper",
        6174,
        "my",
        "Burmese Whisper fine-tune",
    ),
    "whisper-my-medium": HFModel(
        "chuuhtetnaing/whisper-medium-myanmar",
        "6ddaae5665c80e0d3c322bd352656074b9851566",
        "whisper",
        3056,
        "my",
        "same corpus as the large fine-tune",
    ),
    "whisper-my-small": HFModel(
        "chuuhtetnaing/whisper-small-myanmar",
        "f3de3c167914fec3c0974aad1189eda3fa77d8cd",
        "whisper",
        967,
        "my",
        "same corpus as the large fine-tune",
    ),
    "whisper-large-v3": HFModel(
        "openai/whisper-large-v3",
        "06f233fe06e710322aca913c1bc4249a0d71fce1",
        "whisper",
        3087,
        "my",
        "stock Whisper baseline",
    ),
    "mms-1b-all": HFModel(
        "facebook/mms-1b-all",
        "3d33597edbdaaba14a8e858e2c8caa76e3cec0cd",
        "mms",
        3869,
        "mya",
        "CC-BY-NC; Burmese adapter",
    ),
    "seamless-m4t-v2": HFModel(
        "facebook/seamless-m4t-v2-large",
        "5f8cc790b19fc3f67a61c105133b20b34e3dcb76",
        "seamless",
        9237,
        "mya",
        "CC-BY-NC; speech/text input, text output only",
    ),
    "w2v-bert-my": HFModel(
        "YonaKhine/finetuned-w2v2-bert-burmese-asr",
        "3a0bb058936140acfe7c905171eefc78234e93be",
        "ctc",
        2423,
        None,
        "monolingual Burmese CTC fine-tune",
    ),
}

# Its weights are CC-BY-NC-4.0 and unsuitable for commercial use.
DEFAULT_MODEL = "seamless-m4t-v2"


@register
class TransformersASRBackend(ASRBackend):
    name: ClassVar[str] = "hf"
    description: ClassVar[str] = "Hugging Face transformers: Whisper / MMS / SeamlessM4T / w2v-BERT"
    install_hint: ClassVar[str] = "uv sync --extra hf"
    supported_options: ClassVar[frozenset[str]] = frozenset({"device", "dtype", "local_files_only"})

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: str = "auto",
        dtype: str = "auto",
        local_files_only: bool = False,
        **options: Any,
    ) -> None:
        if model not in MODELS:
            known = ", ".join(sorted(MODELS))
            raise ValueError(f"Unknown HF model {model!r}. Available: {known}")
        super().__init__(model, **options)
        self.spec = MODELS[model]
        self.device_arg = device
        self.dtype_arg = dtype
        self.local_files_only = local_files_only
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
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        return True, f"transformers {transformers.__version__} on {device}"

    def _resolve_device(self) -> str:
        import torch

        if self.device_arg != "auto":
            return self.device_arg
        if torch.cuda.is_available():
            return "cuda"
        # These architectures use MPS by default on Apple Silicon.
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
        # Keep the adapter's default numerics stable across supported devices.
        return torch.float32

    def estimated_download_mb(self) -> int | None:
        return self.spec.approx_mb

    def weights_cached(self) -> bool | None:
        from huggingface_hub import constants

        cache = Path(constants.HF_HUB_CACHE)
        folder = cache / f"models--{self.spec.repo.replace('/', '--')}"
        if not folder.is_dir():
            return False
        return self._snapshot_complete(folder / "snapshots" / self.spec.revision)

    def _snapshot_complete(self, snapshot: Path) -> bool:
        if not (snapshot / "config.json").is_file():
            return False
        weights = any(
            (snapshot / filename).is_file()
            for filename in ("model.safetensors", "pytorch_model.bin")
        )
        for filename in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            index = snapshot / filename
            if not index.is_file():
                continue
            try:
                shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
            except (KeyError, OSError, TypeError, ValueError):
                continue
            if shards and all((snapshot / shard).is_file() for shard in shards):
                weights = True
                break
        if not weights:
            return False
        if not (snapshot / "preprocessor_config.json").is_file():
            return False
        if self.spec.family == "whisper":
            return (snapshot / "tokenizer.json").is_file() or all(
                (snapshot / filename).is_file() for filename in ("vocab.json", "merges.txt")
            )
        if self.spec.family == "mms":
            adapter = any(
                (snapshot / f"adapter.{self.spec.lang}.{suffix}").is_file()
                for suffix in ("safetensors", "bin")
            )
            return adapter and (snapshot / "vocab.json").is_file()
        if self.spec.family == "ctc":
            return (snapshot / "vocab.json").is_file()
        if self.spec.family == "seamless":
            return all(
                (snapshot / filename).is_file()
                for filename in ("tokenizer.model", "sentencepiece.bpe.model")
            )
        return False

    def download_weights(self) -> None:
        from huggingface_hub import HfApi, snapshot_download

        names = [
            file.rfilename
            for file in HfApi()
            .model_info(
                self.spec.repo,
                revision=self.spec.revision,
                files_metadata=True,
            )
            .siblings
        ]
        weights = [
            name
            for name in names
            if name == "model.safetensors"
            or (name.startswith("model-") and name.endswith(".safetensors"))
        ]
        if any(name.startswith("model-") for name in weights):
            weights.append("model.safetensors.index.json")
        if not weights:
            weights = [
                name
                for name in names
                if name == "pytorch_model.bin"
                or (name.startswith("pytorch_model-") and name.endswith(".bin"))
            ]
            if any(name.startswith("pytorch_model-") for name in weights):
                weights.append("pytorch_model.bin.index.json")
        if not weights:
            raise RuntimeError(f"No PyTorch weights found in {self.spec.repo}")
        if self.spec.family == "mms":
            adapters = (
                f"adapter.{self.spec.lang}.safetensors",
                f"adapter.{self.spec.lang}.bin",
            )
            adapter = next((name for name in adapters if name in names), None)
            if adapter is None:
                raise RuntimeError(f"No {self.spec.lang} adapter found in {self.spec.repo}")
            weights.append(adapter)

        metadata_suffixes = (".json", ".txt", ".model", ".yaml", ".yml")
        metadata = [
            name
            for name in names
            if name.endswith(metadata_suffixes)
            and not name.endswith(".index.json")
            and not (self.spec.family == "mms" and "/" in name)
        ]
        snapshot = Path(
            snapshot_download(
                repo_id=self.spec.repo,
                revision=self.spec.revision,
                allow_patterns=metadata + weights,
            )
        )
        if not self._snapshot_complete(snapshot):
            raise RuntimeError(
                f"Downloaded snapshot for {self.spec.repo}@{self.spec.revision} is incomplete"
            )

    def load(self) -> None:
        from transformers import AutoProcessor, pipeline

        device = self._resolve_device()
        dtype = self._resolve_dtype(device)
        self.resolved_device = device
        self.resolved_dtype = str(dtype).replace("torch.", "")

        repo = self.spec.repo
        hub_options = {
            "revision": self.spec.revision,
            "local_files_only": self.local_files_only,
        }

        if self.spec.family == "seamless":
            from transformers import SeamlessM4Tv2ForSpeechToText

            processor = AutoProcessor.from_pretrained(repo, **hub_options)
            model = SeamlessM4Tv2ForSpeechToText.from_pretrained(repo, dtype=dtype, **hub_options)
            model.to(device).eval()
            self._seamless = (processor, model)
            self._loaded = True
            return

        if self.spec.family == "mms":
            from transformers import Wav2Vec2ForCTC

            processor = AutoProcessor.from_pretrained(repo, **hub_options)
            model = Wav2Vec2ForCTC.from_pretrained(repo, dtype=dtype, **hub_options)
            # MMS is one shared encoder plus a tiny per-language adapter; both
            # the tokenizer and the model have to be switched to Burmese or you
            # silently decode with the previous language's vocabulary.
            processor.tokenizer.set_target_lang(self.spec.lang)
            model.load_adapter(self.spec.lang, **hub_options)
            model.to(device).eval()
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                dtype=dtype,
                device=device,
            )
            self._loaded = True
            return

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=repo,
            revision=self.spec.revision,
            model_kwargs={"local_files_only": self.local_files_only},
            dtype=dtype,
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
