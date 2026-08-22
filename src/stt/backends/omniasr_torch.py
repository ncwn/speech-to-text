"""Meta Omnilingual ASR via the official PyTorch/fairseq2 runtime.

This is the reference implementation — every model card Meta published works
here, including ``omniASR_LLM_Unlimited_7B_v2``, and it is the accuracy ground
truth the faster backends are measured against.

fairseq2's MPS path runs every card on Metal, which is the default on Apple
Silicon. See ``docs/findings.md#device-defaults``.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from stt.backends.base import ASRBackend
from stt.registry import register
from stt.results import TranscriptionResult

#: Cards without "Unlimited" in the name reject audio longer than this.
MAX_LIMITED_AUDIO_SEC = 40


@dataclass(frozen=True)
class OmniASRModel:
    approx_mb: int
    unlimited: bool
    note: str = ""


#: The common cards. Any name from facebookresearch/omnilingual-asr also works —
#: size and length limit are then derived from the name, so a new card needs no
#: entry here.
MODELS: dict[str, OmniASRModel] = {
    "omniASR_LLM_Unlimited_300M_v2": OmniASRModel(6500, True),
    "omniASR_LLM_Unlimited_1B_v2": OmniASRModel(9100, True),
    "omniASR_LLM_Unlimited_3B_v2": OmniASRModel(17500, True),
    "omniASR_LLM_Unlimited_7B_v2": OmniASRModel(31200, True),
    "omniASR_LLM_7B_v2": OmniASRModel(31200, False),
    "omniASR_CTC_7B_v2": OmniASRModel(30000, False, "no language hint"),
}

DEFAULT_MODEL = "omniASR_LLM_Unlimited_7B_v2"

#: Approximate fp32 checkpoint download size by parameter count, in MB. Keyed by
#: size tag so an unlisted card still resolves.
_DOWNLOAD_MB = {"300M": 6500, "1B": 9100, "3B": 17500, "7B": 31200}


@register
class OmniASRTorchBackend(ASRBackend):
    name: ClassVar[str] = "omniasr-torch"
    description: ClassVar[str] = (
        "Meta Omnilingual ASR, official PyTorch/fairseq2 runtime (Metal on macOS)"
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
            # Faster and smaller than CPU for identical text; `transcribe` falls
            # back to CPU if a kernel fails. docs/findings.md#device-defaults
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
            # Which 16-bit format is fastest is a property of the GPU, and it
            # differs by generation — so measure, do not assume.
            # docs/findings.md#precision
            from stt.hardware import fastest_dtype

            return named[fastest_dtype(device)]

        # On CPU, float32 is the fast path — PyTorch emulates half precision
        # there. float32 unless the machine cannot hold the card.
        # docs/findings.md#precision
        if not self._has_headroom_for_float32():
            return torch.bfloat16
        return torch.float32

    #: Resident memory runs to roughly the checkpoint size plus activations and
    #: the buffers used to read it, so a new card needs no new constant here.
    _FLOAT32_OVERHEAD = 1.5

    def _has_headroom_for_float32(self) -> bool:
        """Whether this machine can hold this particular card at float32."""
        from stt.hardware import total_ram_mb

        checkpoint_mb = _DOWNLOAD_MB.get(self._model_size_tag())
        ram_mb = total_ram_mb()
        if not checkpoint_mb or not ram_mb:
            return True  # unknown card or unknown machine: keep the fast path
        return ram_mb >= checkpoint_mb * self._FLOAT32_OVERHEAD

    def _model_size_tag(self) -> str:
        for tag in ("300M", "1B", "3B", "7B"):
            if f"_{tag}_" in self.model or self.model.endswith(f"_{tag}"):
                return tag
        return "unknown"

    @property
    def is_unlimited(self) -> bool:
        spec = MODELS.get(self.model)
        return spec.unlimited if spec else "Unlimited" in self.model

    def estimated_download_mb(self) -> int | None:
        spec = MODELS.get(self.model)
        return spec.approx_mb if spec else _DOWNLOAD_MB.get(self._model_size_tag())

    def weights_cached(self) -> bool | None:
        """Unknowable: fairseq2 stores assets under opaque content hashes."""
        return None

    def load(self) -> None:
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

        device = self._resolve_device()
        dtype = self._resolve_dtype(device)

        self.resolved_device = device
        self.resolved_dtype = str(dtype).replace("torch.", "")

        binding = self.model_binding
        if binding is None:
            self.pipeline = ASRInferencePipeline(model_card=self.model, device=device, dtype=dtype)
        else:
            import torch
            from fairseq2.assets import AssetCard, get_asset_store
            from fairseq2.data.tokenizers.hub import load_tokenizer
            from fairseq2.models.hub import load_model

            # fairseq2's loader expects the torch device API (it reads
            # ``device.type``), while provenance keeps the resolved name as a
            # string.  ASRInferencePipeline forwards this device to the
            # already-bound model as well.
            runtime_device = torch.device(device)

            weights = [item for item in binding.paths if item.role == "weights" and item.path]
            tokenizers = [item for item in binding.paths if item.role == "tokenizer" and item.path]
            if len(weights) != 1 or len(tokenizers) != 1:
                raise RuntimeError(
                    "omniASR binding must contain exactly one checkpoint and tokenizer"
                )
            store = get_asset_store()
            source_card = store.retrieve_card(self.model)
            tokenizer_name = source_card.field("tokenizer_ref").as_(str)
            source_tokenizer = store.retrieve_card(tokenizer_name)
            captured_card = next(
                (item for item in binding.provenance.artifacts if item.role == "card-metadata"),
                None,
            )
            live_semantics = {
                "name": source_card.name,
                "model_family": source_card.field("model_family").as_(str),
                "model_arch": source_card.field("model_arch").as_(str),
                "checkpoint": str(source_card.field("checkpoint").as_uri()),
                "tokenizer_ref": tokenizer_name,
                "tokenizer_family": source_tokenizer.field("tokenizer_family").as_(str),
                "tokenizer": str(source_tokenizer.field("tokenizer").as_uri()),
            }
            live_digest = hashlib.sha256(
                json.dumps(
                    live_semantics,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode("ascii")
            ).hexdigest()
            if captured_card is None or live_digest != captured_card.sha256:
                raise RuntimeError("omniASR asset-card semantics changed after preflight")
            model_metadata = {
                "model_family": live_semantics["model_family"],
                "model_arch": live_semantics["model_arch"],
                "checkpoint": Path(str(weights[0].path)).resolve().as_uri(),
                "tokenizer_ref": tokenizer_name,
            }
            tokenizer_metadata = {
                "tokenizer_family": live_semantics["tokenizer_family"],
                "tokenizer": Path(str(tokenizers[0].path)).resolve().as_uri(),
            }
            bound_model = load_model(
                AssetCard(f"{self.model}.stt-bound", model_metadata),
                device=runtime_device,
                dtype=dtype,
                progress=False,
            )
            bound_tokenizer = load_tokenizer(
                AssetCard(f"{tokenizer_name}.stt-bound", tokenizer_metadata),
                progress=False,
            )
            self.pipeline = ASRInferencePipeline(
                model_card=None,
                model=bound_model,
                tokenizer=bound_tokenizer,
                device=runtime_device,
                dtype=dtype,
            )
        self._loaded = True

    def unload(self) -> None:
        self.pipeline = None
        self._loaded = False
        import gc

        gc.collect()

    def _parity_trace(self, path: Path, *, language: str | None, adapter: bool):
        """Compare the repository pipeline call with its direct fairseq2 path."""
        if self.pipeline is None:
            raise RuntimeError("omniASR parity requires a loaded pipeline")
        if "Unlimited" not in self.model:
            raise RuntimeError("omniASR parity currently covers unlimited LLM cards only")

        import numpy as np
        import torch

        from stt.parity import ParityTrace

        pipeline = self.pipeline
        model = pipeline.model
        if not hasattr(model, "embed_audio") or pipeline.beam_search_generator is None:
            raise RuntimeError("omniASR parity requires the Wav2Vec2Llama model path")

        waveforms: list[np.ndarray] = []
        embeddings: list[np.ndarray] = []
        logits: list[np.ndarray] = []
        generated: list[tuple[np.ndarray, list[int]]] = []

        original_build = pipeline._build_audio_wavform_pipeline
        original_embed = model.embed_audio
        original_generate = pipeline.beam_search_generator.generate_hypotheses

        def capture_waveform(value):
            detached = value.detach().cpu()
            waveforms.append(detached.numpy().copy())
            return value

        def capture_build(inp):
            return original_build(inp).map(capture_waveform)

        def capture_embed(*args, **kwargs):
            result = original_embed(*args, **kwargs)
            embeddings.append(result[0].detach().cpu().numpy().copy())
            return result

        def capture_generate(*args, **kwargs):
            tokens, lengths = original_generate(*args, **kwargs)
            generated.append((tokens.detach().cpu().numpy().copy(), list(lengths)))
            return tokens, lengths

        def capture_logits(module, args, output):
            del module, args
            value = output[0] if isinstance(output, tuple) else output
            logits.append(value.detach().cpu().numpy().copy())

        projection_hook = model.final_proj.register_forward_hook(capture_logits)
        pipeline._build_audio_wavform_pipeline = capture_build
        model.embed_audio = capture_embed
        pipeline.beam_search_generator.generate_hypotheses = capture_generate
        try:
            if adapter:
                texts = pipeline.transcribe([path], lang=[language], batch_size=1)
                entrypoint = "repository-pipeline-transcribe"
            else:
                prepared = list(pipeline._build_audio_wavform_pipeline([path]).and_return())
                if len(prepared) != 1:
                    raise RuntimeError("omniASR parity expected one prepared waveform")
                batch = pipeline._create_batch_simple([(prepared[0], language)])
                texts = pipeline._apply_model(batch)
                entrypoint = "fairseq2-batch-model-beam-search"
        finally:
            pipeline._build_audio_wavform_pipeline = original_build
            model.embed_audio = original_embed
            pipeline.beam_search_generator.generate_hypotheses = original_generate
            projection_hook.remove()

        if len(waveforms) != 1 or len(embeddings) != 1:
            raise RuntimeError("omniASR parity expected one prepared waveform and audio embedding")
        if len(generated) != 1 or not logits:
            raise RuntimeError("omniASR parity expected one generation trace")
        token_array, token_lengths = generated[0]
        token_count = int(token_lengths[0])
        token_row = token_array[0, :token_count]
        logit_array = np.concatenate(logits, axis=1)
        raw_decoder = pipeline.tokenizer.create_decoder(skip_special_tokens=False)
        raw = str(raw_decoder(torch.from_numpy(token_row)))
        final = str(pipeline.token_decoder(torch.from_numpy(token_row))).strip()
        if not texts or str(texts[0]).strip() != final:
            raise RuntimeError("omniASR parity direct and pipeline transcript decoders disagree")
        return ParityTrace(
            decoded_pcm=waveforms[0],
            features=embeddings[0],
            logits_or_encoder=logit_array,
            token_ids=token_row,
            raw_transcript=raw,
            final_transcript=final,
            metadata={
                "entrypoint": entrypoint,
                "decoder": "fairseq2 AudioDecoder + add_waveform_processing",
                "feature_stage": "Wav2Vec2LlamaModel.embed_audio",
                "logit_stage": "Wav2Vec2LlamaModel.final_proj during beam search",
                "torch_inference_mode": True,
            },
        )

    def parity_adapter_trace(self, path: Path, *, language: str | None = None):
        return self._parity_trace(path, language=language, adapter=True)

    def parity_reference_trace(self, path: Path, *, language: str | None = None):
        return self._parity_trace(path, language=language, adapter=False)

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

    def _transcribe_paths(
        self, paths: list[str], lang_arg: list[str] | None, batch_size: int
    ) -> list[str]:
        """Decode one or more files, retrying on CPU if Metal fails.

        Metal is the measured-faster default, but fairseq2 does not test it and
        an unimplemented kernel would otherwise turn a slow run into a failed
        one. Falling back costs speed; not falling back costs the transcript.
        The switch is permanent for this instance, so a systematic failure does
        not pay the Metal attempt on every remaining file.
        """
        assert self.pipeline is not None
        try:
            texts = self.pipeline.transcribe(paths, lang=lang_arg, batch_size=batch_size)
            if len(texts) != len(paths):
                raise RuntimeError(
                    f"omniASR returned {len(texts)} transcript(s) for {len(paths)} input(s)"
                ) from None
            return texts
        except Exception as exc:  # noqa: BLE001 - any Metal failure is worth retrying on CPU
            if self.resolved_device != "mps":
                raise
            import logging

            logging.getLogger(__name__).warning(
                "Metal failed (%s: %s); falling back to CPU for the rest of this run.",
                type(exc).__name__,
                exc,
            )
            self.record_fallback(
                from_device=self.resolved_device,
                to_device="cpu",
                error=f"{type(exc).__name__}: {exc}",
            )
            self.unload()
            self.device_arg = "cpu"
            self.load()
            assert self.pipeline is not None
            texts = self.pipeline.transcribe(paths, lang=lang_arg, batch_size=batch_size)
            if len(texts) != len(paths):
                raise RuntimeError(
                    f"omniASR returned {len(texts)} transcript(s) for {len(paths)} input(s)"
                ) from None
            return texts

    def _transcribe_one(self, path: str, lang_arg: list[str] | None, batch_size: int) -> list[str]:
        """Decode one file, retaining the old helper's CPU fallback behavior."""
        return self._transcribe_paths([path], lang_arg, batch_size)

    def transcribe(
        self,
        audio_paths: list[Path],
        language: str | None = None,
        batch_size: int = 1,
    ) -> list[TranscriptionResult]:
        if self.pipeline is None:
            self.load()

        self._check_language(language)

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        from stt.audio import duration_of

        def _result_metadata() -> dict[str, Any]:
            # Resolve at result creation time because an MPS failure can switch
            # the backend to CPU during the decode call.
            return {"device": self.resolved_device, "dtype": self.resolved_dtype}

        # Keep slots so validation failures and decoder failures cannot change
        # the positional contract when a batch is retried one file at a time.
        slots: list[TranscriptionResult | None] = [None] * len(audio_paths)
        valid: list[tuple[int, Path, float]] = []

        for index, path in enumerate(audio_paths):
            duration: float | None = None
            try:
                duration = duration_of(path)
                if not self.is_unlimited and duration > MAX_LIMITED_AUDIO_SEC:
                    raise ValueError(
                        f"{path.name} is {duration:.1f}s but {self.model} accepts at most "
                        f"{MAX_LIMITED_AUDIO_SEC}s. Use an "
                        f"omniASR_LLM_Unlimited_*_v2 card for long audio."
                    )
                valid.append((index, path, duration))
            except Exception as exc:  # noqa: BLE001 - one bad clip must not abort the sweep
                slots[index] = TranscriptionResult(
                    audio_path=str(path),
                    text="",
                    backend=self.name,
                    model=self.model,
                    language=language,
                    audio_duration_s=duration,
                    error=f"{type(exc).__name__}: {exc}",
                    metadata=_result_metadata(),
                )

        if valid:
            paths = [str(path) for _, path, _ in valid]
            lang_arg = [language] * len(paths) if language else None
            started = time.perf_counter()
            try:
                texts = self._transcribe_paths(paths, lang_arg, batch_size)
                elapsed = time.perf_counter() - started
                total_duration = sum(duration for _, _, duration in valid)
                # The pipeline reports one aggregate wall time. Allocate it by
                # duration so corpus RTF remains meaningful, and mark it as an
                # estimate rather than pretending it was measured per file.
                weights = (
                    [duration / total_duration for _, _, duration in valid]
                    if total_duration > 0
                    else [1.0 / len(valid)] * len(valid)
                )
                for (index, path, duration), text, weight in zip(
                    valid, texts, weights, strict=True
                ):
                    result_metadata = {
                        **_result_metadata(),
                        "batch_size": batch_size,
                        "batch_items": len(valid),
                        "batch_elapsed_s": elapsed,
                        "elapsed_s_source": "batch_proportional",
                    }
                    slots[index] = TranscriptionResult(
                        audio_path=str(path),
                        text=text.strip(),
                        backend=self.name,
                        model=self.model,
                        language=language,
                        elapsed_s=elapsed * weight,
                        audio_duration_s=duration,
                        metadata=result_metadata,
                    )
            except Exception as batch_exc:  # noqa: BLE001 - retry per-file below
                # A single malformed clip, unsupported batch shape, or OOM must
                # not discard successful transcripts from the other inputs.
                import logging

                logging.getLogger(__name__).warning(
                    "omniASR batch of %d file(s) failed (%s: %s); retrying individually.",
                    len(valid),
                    type(batch_exc).__name__,
                    batch_exc,
                )
                for index, path, duration in valid:
                    single_started = time.perf_counter()
                    try:
                        texts = self._transcribe_one(
                            str(path), [language] if language else None, batch_size
                        )
                        text = texts[0]
                        single_elapsed = time.perf_counter() - single_started
                        slots[index] = TranscriptionResult(
                            audio_path=str(path),
                            text=text.strip(),
                            backend=self.name,
                            model=self.model,
                            language=language,
                            elapsed_s=single_elapsed,
                            audio_duration_s=duration,
                            metadata={
                                **_result_metadata(),
                                "batch_size": batch_size,
                                "batch_items": len(valid),
                                "batch_fallback": True,
                                "elapsed_s_source": "measured",
                            },
                        )
                    except Exception as exc:  # noqa: BLE001 - preserve per-file contract
                        slots[index] = TranscriptionResult(
                            audio_path=str(path),
                            text="",
                            backend=self.name,
                            model=self.model,
                            language=language,
                            audio_duration_s=duration,
                            error=f"{type(exc).__name__}: {exc}",
                            metadata={
                                **_result_metadata(),
                                "batch_size": batch_size,
                                "batch_items": len(valid),
                                "batch_fallback": True,
                            },
                        )

        # Validation and decoder paths above fill every slot; the assertion is
        # a guard against accidentally weakening the one-result-per-input API.
        assert all(result is not None for result in slots)
        return [result for result in slots if result is not None]
