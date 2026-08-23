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

import hashlib
import math
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
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

# Seamless normally permits 256 generated tokens for every window. A missed EOS
# in one batch member then holds the entire autoregressive batch open. The
# trusted FLEURS test corpus stayed below 8 output tokens/s across 140 windows;
# retain that measured ceiling with a floor for short tails and the model's
# existing upper bound. Batch 32 is the measured Metal throughput knee: batch
# 64 regresses corpus RTF and GPU occupancy while tripling Metal allocation.
_SEAMLESS_OUTPUT_TOKENS_PER_SECOND = 8.0
_SEAMLESS_MIN_NEW_TOKENS = 32
_SEAMLESS_MAX_NEW_TOKENS = 256
_SEAMLESS_MPS_MAX_BATCH_SIZE = 32


def _seamless_max_new_tokens(sample_count: int, sample_rate: int) -> int:
    """Return the bounded decoder budget for one Seamless audio window."""
    if sample_count < 0 or sample_rate <= 0:
        raise ValueError(
            "Seamless token budgeting requires non-negative samples and a positive rate"
        )
    scaled = math.ceil(sample_count / sample_rate * _SEAMLESS_OUTPUT_TOKENS_PER_SECOND)
    return min(_SEAMLESS_MAX_NEW_TOKENS, max(_SEAMLESS_MIN_NEW_TOKENS, scaled))


@dataclass(frozen=True)
class HFModel:
    repo: str
    family: str
    approx_mb: int
    #: Language code this family expects, given a Burmese request.
    lang: str | None
    note: str = ""


@dataclass(frozen=True)
class _SeamlessBatchDecode:
    segments: list[list[Segment]]
    mode: str
    per_file_elapsed_s: tuple[float, ...] | None = None


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

# Chosen on measured CER over FLEURS Burmese, not on reputation: Seamless beat
# every Whisper fine-tune by a factor of six. docs/findings.md#baseline
# Its weights are CC-BY-NC-4.0 -- evaluation only, never a product.
DEFAULT_MODEL = "seamless-m4t-v2"


def _transformers_version() -> tuple[int, int]:
    """Return the major/minor Transformers version for API feature checks.

    The HF extra intentionally supports a range of Transformers releases.  A
    tiny local parser avoids importing another package merely to compare
    versions, and treats an unrecognised version conservatively as the oldest
    supported API.
    """
    import transformers

    match = re.match(r"(\d+)\.(\d+)", getattr(transformers, "__version__", ""))
    return (int(match.group(1)), int(match.group(2))) if match else (0, 0)


def _dtype_kwargs(dtype: Any) -> dict[str, Any]:
    """Build model/pipeline dtype kwargs across Transformers API versions.

    ``torch_dtype`` was renamed to ``dtype`` in Transformers 4.56.  Keep the
    old spelling for the project's declared 4.45+ compatibility range while
    using the non-deprecated spelling everywhere it is supported.
    """
    key = "dtype" if _transformers_version() >= (4, 56) else "torch_dtype"
    return {key: dtype}


def _audio_kwarg(audio: Any) -> dict[str, Any]:
    """Build the Seamless processor audio kwarg across API versions.

    The processor accepted ``audios`` through 4.56; ``audio`` was added in
    4.57 and the former spelling is scheduled for removal in 4.59.
    """
    key = "audio" if _transformers_version() >= (4, 57) else "audios"
    return {key: audio}


def _bound_hf_snapshot(binding: Any) -> Path:
    """Return the exact local directory containing a validated HF binding.

    The provenance preflight records paths for every file selected from the
    commit-addressed snapshot. Passing the Hub repo id plus a revision here
    would still let ``transformers`` consult another cache entry, so bound
    workers must use the worker-local paths themselves. A common parent keeps
    nested artifacts (for example ``vocabs/mya.txt``) in the same snapshot
    while preserving their individual digest validation.
    """
    from stt.paths import cache_dir
    from stt.provenance import _cache_lock, artifact_manifest_sha256, validate_binding

    issues = validate_binding(binding)
    if issues:
        raise RuntimeError("bound Hugging Face artifacts are invalid: " + "; ".join(issues))

    declared = tuple(binding.provenance.artifacts)
    by_name = {item.name: item for item in binding.paths}
    if len(by_name) != len(declared) or any(
        artifact.name not in by_name or not by_name[artifact.name].path for artifact in declared
    ):
        missing = sorted({artifact.name for artifact in declared} - set(by_name))
        raise RuntimeError("bound Hugging Face artifacts have no local path: " + ", ".join(missing))

    # ``ModelBinding.validate`` guarantees this one-to-one lookup has the same
    # descriptor identity that ``validate_binding`` hashed above; do not select
    # a second path by iterating an untrusted list.
    paths = [by_name[artifact.name] for artifact in declared]
    files = [Path(item.path).expanduser().resolve(strict=True) for item in paths]
    if not declared:
        raise RuntimeError("bound Hugging Face provenance declares no artifacts")

    # Hub snapshots normally expose these names through symlinks into a
    # content-addressed ``blobs`` directory.  A worker binding deliberately
    # records the resolved blob path, while Transformers expects the
    # snapshot-relative names (especially for adapters and vocabularies).  Do
    # not infer names from the blob parent; materialize the declared view.
    relative_names: list[PurePosixPath] = []
    for artifact in declared:
        name = artifact.name
        if (
            not name
            or "\x00" in name
            or "\\" in name
            or PurePosixPath(name).is_absolute()
            or PureWindowsPath(name).drive
            or any(part in {"", ".", ".."} for part in name.split("/"))
        ):
            raise RuntimeError(f"bound Hugging Face artifact name escapes its snapshot: {name!r}")
        relative_names.append(PurePosixPath(name))

    # Include the source paths in the key because worker-local bindings may
    # point at different copies of the same validated blobs.  This keeps each
    # immutable view self-contained instead of racing to retarget a shared
    # symlink tree, while repeated calls for one binding reuse the same view.
    manifest_key = artifact_manifest_sha256(tuple(declared))
    source_key = hashlib.sha256(
        "\n".join(
            f"{name.as_posix()}\0{path}" for name, path in zip(relative_names, files, strict=True)
        ).encode("utf-8")
    ).hexdigest()
    cache_parent = cache_dir("huggingface") / "bound-snapshots"
    cache_parent.mkdir(parents=True, exist_ok=True)
    snapshot = cache_parent / f"{manifest_key}-{source_key}"
    lock_path = cache_parent / f".{snapshot.name}.lock"

    def valid_view() -> bool:
        if not snapshot.is_dir() or snapshot.is_symlink():
            return False
        expected_files = {Path(relative) for relative in relative_names}
        expected_dirs = {Path(".")}
        for relative in expected_files:
            expected_dirs.update(Path(parent) for parent in relative.parents)
        for relative, source in zip(relative_names, files, strict=True):
            destination = snapshot / relative
            if not destination.is_symlink():
                return False
            try:
                if destination.resolve(strict=True) != source:
                    return False
            except OSError:
                return False
        try:
            entries = snapshot.rglob("*")
        except OSError:
            return False
        for entry in entries:
            relative = entry.relative_to(snapshot)
            if relative in expected_files:
                if not entry.is_symlink():
                    return False
            elif relative in expected_dirs:
                if entry.is_symlink() or not entry.is_dir():
                    return False
            else:
                return False
        return all((snapshot / relative).is_symlink() for relative in expected_files)

    with _cache_lock(lock_path):
        if valid_view():
            return snapshot
        if snapshot.exists() or snapshot.is_symlink():
            raise RuntimeError(f"bound Hugging Face snapshot cache is invalid: {snapshot}")

        staging = Path(tempfile.mkdtemp(prefix=f".{snapshot.name}.", dir=cache_parent))
        try:
            for relative, source in zip(relative_names, files, strict=True):
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.symlink_to(source)
            os.replace(staging, snapshot)
        except FileExistsError:
            # A non-POSIX worker may not honour flock.  If another creator won
            # the race, accept its complete view; otherwise fail closed.
            if not valid_view():
                raise RuntimeError(
                    f"bound Hugging Face snapshot cache is invalid: {snapshot}"
                ) from None
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        if not valid_view():
            raise RuntimeError(f"bound Hugging Face snapshot cache is invalid: {snapshot}")
        return snapshot


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
        self.resolved_revision: str | None = None

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
        # Half precision is both slower *and* no more accurate for Seamless, so
        # it buys nothing but GPU memory — the opposite of omniASR's LLM
        # decoder. The best dtype is a property of the model as much as of the
        # chip, so this backend does not share omniASR's probe.
        # docs/findings.md#precision
        return torch.float32

    def estimated_download_mb(self) -> int | None:
        return self.spec.approx_mb

    def preferred_batch_size(self) -> int:
        """Use the measured saturation batch only for Seamless on Metal."""
        device = self.resolved_device or self._resolve_device()
        if self.spec.family == "seamless" and device == "mps":
            return _SEAMLESS_MPS_MAX_BATCH_SIZE
        return 1

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

        binding = self.model_binding
        revision = binding.provenance.upstream_revision if binding else None
        self.resolved_revision = revision
        snapshot = _bound_hf_snapshot(binding) if binding is not None else None
        source = str(snapshot) if snapshot is not None else self.spec.repo

        def load_kwargs() -> dict[str, Any]:
            return {"local_files_only": True} if snapshot is not None else {}

        if self.spec.family == "seamless":
            from transformers import SeamlessM4Tv2ForSpeechToText

            processor = AutoProcessor.from_pretrained(source, **load_kwargs())
            model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
                source, **load_kwargs(), **_dtype_kwargs(dtype)
            )
            model.to(device).eval()
            self._seamless = (processor, model)
            self._loaded = True
            return

        if self.spec.family == "mms":
            from transformers import Wav2Vec2ForCTC

            processor = AutoProcessor.from_pretrained(source, **load_kwargs())
            model = Wav2Vec2ForCTC.from_pretrained(source, **load_kwargs(), **_dtype_kwargs(dtype))
            # MMS is one shared encoder plus a tiny per-language adapter; both
            # the tokenizer and the model have to be switched to Burmese or you
            # silently decode with the previous language's vocabulary.
            processor.tokenizer.set_target_lang(self.spec.lang)
            model.load_adapter(self.spec.lang, **load_kwargs())
            model.to(device).eval()
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                **_dtype_kwargs(dtype),
                device=device,
            )
            self._loaded = True
            return

        pipeline_kwargs: dict[str, Any] = {
            "model": source,
            **_dtype_kwargs(dtype),
            "device": device,
        }
        if snapshot is not None:
            pipeline_kwargs["model_kwargs"] = load_kwargs()
            # Keep tokenizer/feature-extractor resolution on the same pinned,
            # offline snapshot as the model weights. Pipeline forwards
            # ``model_kwargs`` to these Auto* loaders on supported versions.
            pipeline_kwargs["tokenizer"] = source
            pipeline_kwargs["feature_extractor"] = source
        self.pipe = pipeline("automatic-speech-recognition", **pipeline_kwargs)
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
                **_audio_kwarg(np.asarray(chunk)), sampling_rate=rate, return_tensors="pt"
            ).to(model.device, model.dtype)
            with torch.inference_mode():
                tokens = model.generate(
                    **inputs,
                    tgt_lang=self.spec.lang,
                    max_new_tokens=_seamless_max_new_tokens(len(chunk), rate),
                )
            return processor.decode(tokens[0].tolist(), skip_special_tokens=True).strip()

        return windowed(pcm, rate, window, decode)

    def _transcribe_seamless_batch(
        self, paths: list[Path], batch_size: int
    ) -> _SeamlessBatchDecode:
        """Decode Seamless windows in batches while preserving file order."""
        import numpy as np
        import soundfile as sf
        import torch

        from stt.audio import split_on_quiet

        assert self._seamless is not None
        processor, model = self._seamless
        window, _ = _CHUNKING["seamless"]
        loaded: list[tuple[np.ndarray, int]] = []
        per_file_elapsed = [0.0] * len(paths)
        for path in paths:
            file_started = time.perf_counter()
            pcm, rate = sf.read(str(path), dtype="float32", always_2d=False)
            if pcm.ndim > 1:
                pcm = pcm.mean(axis=1)
            loaded.append((pcm, rate))
            per_file_elapsed[len(loaded) - 1] += time.perf_counter() - file_started

        # The normal CLI path converts everything to 16 kHz. If a caller uses
        # the backend directly with mixed rates, retain the old per-file path.
        if len({rate for _, rate in loaded}) != 1:
            decoded = []
            for index, path in enumerate(paths):
                item_started = time.perf_counter()
                decoded.append(self._transcribe_seamless(path))
                per_file_elapsed[index] += time.perf_counter() - item_started
            return _SeamlessBatchDecode(
                decoded,
                "serial-mixed-rate",
                tuple(per_file_elapsed),
            )

        rate = loaded[0][1]
        pending: list[tuple[int, int, int, np.ndarray]] = []
        segments: list[list[Segment]] = [[] for _ in paths]
        for file_index, (pcm, _) in enumerate(loaded):
            split_started = time.perf_counter()
            for start, end in split_on_quiet(pcm, rate, window):
                if end - start < rate * 0.2:
                    continue
                pending.append((file_index, start, end, pcm[start:end]))
            per_file_elapsed[file_index] += time.perf_counter() - split_started

        def decode_single(chunk: np.ndarray) -> str:
            inputs = processor(
                **_audio_kwarg(np.asarray(chunk)),
                sampling_rate=rate,
                return_tensors="pt",
            ).to(model.device, model.dtype)
            with torch.inference_mode():
                tokens = model.generate(
                    **inputs,
                    tgt_lang=self.spec.lang,
                    max_new_tokens=_seamless_max_new_tokens(len(chunk), rate),
                )
            return processor.decode(tokens[0].tolist(), skip_special_tokens=True).strip()

        # Padding a very short tail changes its decoder context on some Metal
        # kernels. Keep those tails on the exact serial path; full windows still
        # carry the throughput gain from list batching.
        batchable = []
        for item in pending:
            file_index, start, end, chunk = item
            if len(chunk) < rate * 2.0:
                item_started = time.perf_counter()
                text = decode_single(chunk)
                per_file_elapsed[file_index] += time.perf_counter() - item_started
                if text:
                    segments[file_index].append(
                        Segment(
                            text=text,
                            start=start / rate,
                            end=end / rate,
                            source="chunk",
                        )
                    )
            else:
                batchable.append(item)

        for offset in range(0, len(batchable), max(1, batch_size)):
            group = batchable[offset : offset + max(1, batch_size)]
            chunks = [item[3] for item in group]
            inputs = processor(
                **_audio_kwarg(chunks),
                sampling_rate=rate,
                return_tensors="pt",
                padding=True,
            ).to(model.device, model.dtype)
            with torch.inference_mode():
                tokens = model.generate(
                    **inputs,
                    tgt_lang=self.spec.lang,
                    max_new_tokens=_seamless_max_new_tokens(
                        max(len(chunk) for chunk in chunks), rate
                    ),
                )
            texts = [
                processor.decode(row.tolist(), skip_special_tokens=True).strip() for row in tokens
            ]
            if len(texts) != len(group):
                raise RuntimeError(
                    f"Seamless returned {len(texts)} transcript(s) for {len(group)} windows"
                )
            for (file_index, start, end, _), text in zip(group, texts, strict=True):
                if text:
                    segments[file_index].append(
                        Segment(
                            text=text,
                            start=start / rate,
                            end=end / rate,
                            source="chunk",
                        )
                    )
        for items in segments:
            items.sort(key=lambda item: (item.start, item.end))
        if not batchable:
            return _SeamlessBatchDecode(
                segments,
                "serial-short-tails",
                tuple(per_file_elapsed),
            )
        mode = "batched-with-serial-tails" if len(batchable) != len(pending) else "batched"
        return _SeamlessBatchDecode(segments, mode)

    def _pipeline_kwargs(self) -> dict[str, Any]:
        if self.spec.family == "whisper":
            # Whisper's sequential long-form path conditions each window on the
            # previous one; keep this path serial and let the caller preserve
            # that state rather than using naive list batching.
            return {
                "return_timestamps": True,
                "generate_kwargs": {"language": self.spec.lang, "task": "transcribe"},
            }
        window, stride = _CHUNKING[self.spec.family]
        return {"chunk_length_s": window, "stride_length_s": stride}

    @staticmethod
    def _decode_pipeline_output(out: Any) -> tuple[str, list[Segment]]:
        if not isinstance(out, dict):
            return str(out).strip(), []
        text = str(out.get("text", "")).strip()
        segments: list[Segment] = []
        for chunk in out.get("chunks") or []:
            stamp = chunk.get("timestamp") or (None, None)
            piece = str(chunk.get("text", "")).strip()
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

    def _transcribe_pipeline(self, path: Path) -> tuple[str, list[Segment]]:
        out = self.pipe(str(path), **self._pipeline_kwargs())
        return self._decode_pipeline_output(out)

    def parity_adapter_trace(self, path: Path, *, language: str | None = None):
        """Observe the tensors used by the real Transformers pipeline path."""
        if self.spec.family == "seamless":
            return self._parity_seamless_trace(path, language=language, adapter=True)
        if self.spec.family != "mms" or self.pipe is None:
            raise RuntimeError(f"adapter parity is not implemented for {self.model}")
        self._check_language(language)
        import numpy as np
        from transformers.pipelines.audio_utils import ffmpeg_read

        from stt.parity import ParityTrace

        decoded_pcm = ffmpeg_read(path.read_bytes(), self.pipe.feature_extractor.sampling_rate)
        model_inputs: list[np.ndarray] = []
        logits: list[np.ndarray] = []

        def capture_inputs(module, args, kwargs):
            del module, args
            value = kwargs.get(self.pipe.model.main_input_name)
            if value is None:
                raise RuntimeError("pipeline parity could not observe model input values")
            model_inputs.append(value.detach().cpu().numpy())

        def capture_logits(module, args, kwargs, output):
            del module, args, kwargs
            logits.append(output.logits.detach().cpu().numpy())

        before = self.pipe.model.register_forward_pre_hook(capture_inputs, with_kwargs=True)
        after = self.pipe.model.register_forward_hook(capture_logits, with_kwargs=True)
        try:
            output = self.pipe(str(path), **self._pipeline_kwargs())
        finally:
            before.remove()
            after.remove()
        if len(model_inputs) != 1 or len(logits) != 1:
            raise RuntimeError(
                "MMS parity expects one pipeline forward; use audio shorter than its chunk window"
            )
        tokens = np.asarray(logits[0]).argmax(axis=-1)
        raw = str(output.get("text", "")) if isinstance(output, dict) else str(output)
        return ParityTrace(
            decoded_pcm=decoded_pcm,
            features=model_inputs[0],
            logits_or_encoder=logits[0],
            token_ids=tokens,
            raw_transcript=raw,
            final_transcript=raw.strip(),
            metadata={
                "entrypoint": "transformers-pipeline",
                "decoder": "transformers.pipelines.audio_utils.ffmpeg_read",
                "pipeline_manages_inference_mode": True,
            },
        )

    def parity_reference_trace(self, path: Path, *, language: str | None = None):
        """Run the official processor/model/tokenizer components directly."""
        if self.spec.family == "seamless":
            return self._parity_seamless_trace(path, language=language, adapter=False)
        if self.spec.family != "mms" or self.pipe is None:
            raise RuntimeError(f"reference parity is not implemented for {self.model}")
        self._check_language(language)
        import numpy as np
        import soundfile as sf
        import torch

        from stt.parity import ParityTrace

        decoded_pcm, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
        if decoded_pcm.ndim > 1:
            decoded_pcm = decoded_pcm.mean(axis=1)
        processed = self.pipe.feature_extractor(
            decoded_pcm,
            sampling_rate=sample_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )
        processed = processed.to(device=self.pipe.device)
        input_values = processed[self.pipe.model.main_input_name].to(dtype=self.pipe.model.dtype)
        attention_mask = processed.get("attention_mask")
        with torch.inference_mode():
            output = self.pipe.model(
                **{
                    self.pipe.model.main_input_name: input_values,
                    "attention_mask": attention_mask,
                }
            )
        logits = output.logits
        tokens = logits.argmax(dim=-1)
        raw = self.pipe.tokenizer.decode(tokens[0].tolist(), skip_special_tokens=False)
        return ParityTrace(
            decoded_pcm=np.asarray(decoded_pcm),
            features=input_values.detach().cpu().numpy(),
            logits_or_encoder=logits.detach().cpu().numpy(),
            token_ids=tokens.detach().cpu().numpy(),
            raw_transcript=raw,
            final_transcript=raw.strip(),
            metadata={
                "entrypoint": "feature-extractor-model-tokenizer",
                "decoder": "soundfile",
                "torch_inference_mode": True,
            },
        )

    def _parity_seamless_trace(
        self,
        path: Path,
        *,
        language: str | None,
        adapter: bool,
    ):
        """Capture the single-window Seamless processor/generate path."""
        if self._seamless is None:
            raise RuntimeError("Seamless parity requires a loaded processor and model")
        self._check_language(language)
        import numpy as np
        import soundfile as sf
        import torch

        from stt.parity import ParityTrace

        processor, model = self._seamless
        decoded_pcm, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
        if decoded_pcm.ndim > 1:
            decoded_pcm = decoded_pcm.mean(axis=1)
        if len(decoded_pcm) >= sample_rate * _CHUNKING["seamless"][0]:
            raise RuntimeError("Seamless parity fixture must be shorter than one adapter window")

        features: list[np.ndarray] = []
        encoder_outputs: list[np.ndarray] = []

        def capture_features(module, args, kwargs):
            del module, args
            value = kwargs.get("input_features")
            if value is None:
                raise RuntimeError("Seamless parity could not observe input features")
            features.append(value.detach().cpu().numpy())

        def capture_encoder(module, args, kwargs, output):
            del module, args, kwargs
            value = output[0] if isinstance(output, tuple) else output.last_hidden_state
            encoder_outputs.append(value.detach().cpu().numpy())

        before = model.speech_encoder.register_forward_pre_hook(capture_features, with_kwargs=True)
        after = model.speech_encoder.register_forward_hook(capture_encoder, with_kwargs=True)
        generated: list[Any] = []
        try:
            if adapter:
                original_generate = model.generate

                def capture_generate(*args, **kwargs):
                    tokens = original_generate(*args, **kwargs)
                    generated.append(tokens.detach().cpu())
                    return tokens

                model.generate = capture_generate
                try:
                    segments = self._transcribe_seamless(path)
                finally:
                    model.generate = original_generate
                final = join_segments(segments)
            else:
                inputs = processor(
                    **_audio_kwarg(np.asarray(decoded_pcm)),
                    sampling_rate=sample_rate,
                    return_tensors="pt",
                ).to(model.device, model.dtype)
                with torch.inference_mode():
                    tokens = model.generate(
                        **inputs,
                        tgt_lang=self.spec.lang,
                        max_new_tokens=_seamless_max_new_tokens(len(decoded_pcm), sample_rate),
                    )
                generated.append(tokens.detach().cpu())
                final = processor.decode(tokens[0].tolist(), skip_special_tokens=True).strip()
        finally:
            before.remove()
            after.remove()
        if len(features) != 1 or len(encoder_outputs) != 1 or len(generated) != 1:
            raise RuntimeError("Seamless parity expected one processor/encoder/generate call")
        raw = processor.decode(generated[0][0].tolist(), skip_special_tokens=True)
        return ParityTrace(
            decoded_pcm=np.asarray(decoded_pcm),
            features=features[0],
            logits_or_encoder=encoder_outputs[0],
            token_ids=generated[0].numpy(),
            raw_transcript=raw,
            final_transcript=final,
            metadata={
                "entrypoint": (
                    "repository-single-window" if adapter else "processor-generate-direct"
                ),
                "decoder": "soundfile",
                "torch_inference_mode": True,
            },
        )

    def _transcribe_pipeline_batch(
        self, paths: list[Path], batch_size: int
    ) -> list[tuple[str, list[Segment]]]:
        """Use Transformers' list-input batching for non-Whisper pipelines."""
        if self.spec.family == "whisper":
            return [self._transcribe_pipeline(path) for path in paths]
        out = self.pipe(
            [str(path) for path in paths],
            batch_size=max(1, batch_size),
            **self._pipeline_kwargs(),
        )
        if not isinstance(out, list) or len(out) != len(paths):
            raise RuntimeError(
                f"Transformers returned {len(out) if isinstance(out, list) else 1} "
                f"transcript(s) for {len(paths)} input(s)"
            )
        return [self._decode_pipeline_output(item) for item in out]

    def transcribe(
        self,
        audio_paths: list[Path],
        language: str | None = None,
        batch_size: int = 1,
    ) -> list[TranscriptionResult]:
        if not self._loaded:
            self.load()

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        self._check_language(language)

        from stt.audio import duration_of

        slots: list[TranscriptionResult | None] = [None] * len(audio_paths)
        meta = {
            "repo": self.spec.repo,
            "family": self.spec.family,
            "device": self.resolved_device,
            "dtype": self.resolved_dtype,
        }
        if self.spec.family == "seamless":
            meta.update(
                {
                    "max_new_tokens_per_second": _SEAMLESS_OUTPUT_TOKENS_PER_SECOND,
                    "min_new_tokens": _SEAMLESS_MIN_NEW_TOKENS,
                    "max_new_tokens": _SEAMLESS_MAX_NEW_TOKENS,
                    "mps_batch_size_cap": _SEAMLESS_MPS_MAX_BATCH_SIZE,
                }
            )

        def failed(path: Path, duration: float | None, exc: Exception) -> TranscriptionResult:
            return TranscriptionResult(
                audio_path=str(path),
                text="",
                backend=self.name,
                model=self.model,
                language=language,
                audio_duration_s=duration,
                error=f"{type(exc).__name__}: {exc}",
                metadata=dict(meta),
            )

        valid_inputs: list[tuple[int, Path, float]] = []
        for index, path in enumerate(audio_paths):
            duration: float | None = None
            try:
                duration = duration_of(path)
                valid_inputs.append((index, path, duration))
            except Exception as exc:  # noqa: BLE001 - preserve one-result-per-input
                slots[index] = failed(path, duration, exc)

        # Seamless padding cost follows the longest item. Duration ordering
        # keeps each group homogeneous without changing the positional result
        # contract. Other families retain caller order.
        effective_batch_size = (
            min(batch_size, _SEAMLESS_MPS_MAX_BATCH_SIZE)
            if self.spec.family == "seamless" and self.resolved_device == "mps"
            else batch_size
        )
        processing_order = (
            sorted(valid_inputs, key=lambda item: (item[2], item[0]))
            if self.spec.family == "seamless" and batch_size > 1
            else valid_inputs
        )
        for batch_group, offset in enumerate(range(0, len(processing_order), effective_batch_size)):
            valid = processing_order[offset : offset + effective_batch_size]

            paths = [path for _, path, _ in valid]
            can_batch = (
                effective_batch_size > 1 and len(paths) > 1 and self.spec.family != "whisper"
            )
            measured: list[float] = []
            started = time.perf_counter()
            try:
                if can_batch and self.spec.family == "seamless":
                    seamless_decode = self._transcribe_seamless_batch(paths, effective_batch_size)
                    decoded = [(join_segments(items), items) for items in seamless_decode.segments]
                    if seamless_decode.per_file_elapsed_s is not None:
                        measured.extend(seamless_decode.per_file_elapsed_s)
                elif can_batch:
                    decoded = self._transcribe_pipeline_batch(paths, batch_size)
                else:
                    # Serial decode: each file is its own forward pass, so time
                    # them individually. Whisper always lands here, and copying
                    # the whole loop's wall onto every item made a five-file
                    # chunk report five times the work it did.
                    decoded = []
                    for path in paths:
                        item_started = time.perf_counter()
                        if self.spec.family == "seamless":
                            items = self._transcribe_seamless(path)
                            decoded.append((join_segments(items), items))
                        else:
                            decoded.append(self._transcribe_pipeline(path))
                        measured.append(time.perf_counter() - item_started)
                elapsed = time.perf_counter() - started
                total_duration = sum(duration for _, _, duration in valid)
                # One aggregate call can only be apportioned; a serial loop was
                # actually measured per item.
                weights = (
                    [duration / total_duration for _, _, duration in valid]
                    if can_batch and total_duration > 0
                    else [1.0] * len(valid)
                )
                per_item = measured if len(measured) == len(valid) else None
                for position, ((index, path, duration), (text, segments), weight) in enumerate(
                    zip(valid, decoded, weights, strict=True)
                ):
                    slots[index] = TranscriptionResult(
                        audio_path=str(path),
                        text=text,
                        backend=self.name,
                        model=self.model,
                        language=language,
                        elapsed_s=(
                            per_item[position] if per_item is not None else elapsed * weight
                        ),
                        audio_duration_s=duration,
                        metadata={
                            **meta,
                            **(
                                {"seamless_decode_mode": seamless_decode.mode}
                                if can_batch and self.spec.family == "seamless"
                                else {}
                            ),
                            **(
                                {
                                    "batch_size": batch_size,
                                    "effective_batch_size": effective_batch_size,
                                    "batch_items": len(paths),
                                    "batch_group": batch_group,
                                    "batch_elapsed_s": elapsed,
                                    "elapsed_s_source": "batch_proportional",
                                }
                                if can_batch and per_item is None
                                else {
                                    "batch_size": batch_size,
                                    "effective_batch_size": effective_batch_size,
                                    "batch_items": len(paths),
                                    "batch_group": batch_group,
                                    "batch_internal_serial": can_batch,
                                    "elapsed_s_source": (
                                        "measured" if per_item is not None else "serial_share"
                                    ),
                                }
                            ),
                        },
                        segments=segments or None,
                    )
            except Exception as batch_exc:  # noqa: BLE001 - retry individually
                import logging

                if can_batch:
                    logging.getLogger(__name__).warning(
                        "HF batch of %d file(s) failed (%s: %s); retrying individually.",
                        len(paths),
                        type(batch_exc).__name__,
                        batch_exc,
                    )
                for index, path, duration in valid:
                    single_started = time.perf_counter()
                    try:
                        if self.spec.family == "seamless":
                            segments = self._transcribe_seamless(path)
                            text = join_segments(segments)
                        else:
                            text, segments = self._transcribe_pipeline(path)
                        single_elapsed = time.perf_counter() - single_started
                        slots[index] = TranscriptionResult(
                            audio_path=str(path),
                            text=text,
                            backend=self.name,
                            model=self.model,
                            language=language,
                            elapsed_s=single_elapsed,
                            audio_duration_s=duration,
                            metadata={
                                **meta,
                                "batch_size": batch_size,
                                "effective_batch_size": effective_batch_size,
                                "batch_items": len(paths),
                                "batch_group": batch_group,
                                "batch_fallback": can_batch,
                                "elapsed_s_source": "measured",
                            },
                            segments=segments or None,
                        )
                    except Exception as exc:  # noqa: BLE001 - preserve one-result contract
                        slots[index] = failed(path, duration, exc)
            finally:
                if self.resolved_device == "mps":
                    import torch

                    torch.mps.synchronize()
                    torch.mps.empty_cache()

        assert all(result is not None for result in slots)
        return [result for result in slots if result is not None]

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
