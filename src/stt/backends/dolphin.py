"""DataoceanAI Dolphin — multilingual ASR for 40 Eastern languages.

Dolphin is a Whisper-style encoder-decoder trained on East, South and
Southeast Asian speech. Burmese is in its language table as ``my`` with region
``MM``, which makes it one of the few general-purpose multilingual recognisers
that covers Burmese at all.

Only the small (372M) and base (140M) checkpoints were released publicly; the
medium and large rows in the paper are not downloadable. No Burmese-specific
error rate has been published — the 25.2 WER quoted for ``small`` is averaged
across all 40 languages, so treat it as a reason to test, not a prediction.

Language table:
https://github.com/DataoceanAI/Dolphin/blob/main/languages.md
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from stt.audio import join_segments
from stt.backends.base import ASRBackend
from stt.native import suppress_native_output
from stt.paths import cache_dir
from stt.registry import register
from stt.results import Segment, TranscriptionResult

#: Dolphin inherits Whisper's fixed 30-second input window.
WINDOW_SEC = 30.0

#: Burmese, as Dolphin's two-level language/region scheme spells it.
BURMESE_LANG = "my"
BURMESE_REGION = "MM"


@dataclass(frozen=True)
class DolphinModel:
    size: str
    params_m: int
    approx_mb: int


MODELS: dict[str, DolphinModel] = {
    "small": DolphinModel("small", 372, 1500),
    "base": DolphinModel("base", 140, 570),
}

DEFAULT_MODEL = "small"

#: Dolphin writes ``config.yaml`` and ``train.yaml`` alongside the ``.pt`` under
#: whatever directory it is handed, and skips files that already exist. Sharing
#: one directory across sizes therefore leaves the *first* model's config next to
#: a later model's weights, and the load fails with a shape mismatch. Give each
#: size its own directory.
CACHE_ROOT = cache_dir("dolphin", create=False)


def cache_dir(size: str) -> Path:
    return CACHE_ROOT / size


def _demote_float64(module: Any) -> list[str]:
    """Cast a model's float64 buffers to float32, in place.

    Metal does not implement float64 at all, so a single such tensor makes the
    whole model unloadable. In Dolphin's case there are exactly two, both tiny:
    ``encoder.global_cmvn.mean`` and ``.std``, the 80-dimensional mean and
    standard deviation used to normalise filterbank features. All 819 real
    parameters are already float32.

    Normalisation statistics do not need more than float32's seven significant
    digits, so this is a compatibility cast rather than a quantisation — but it
    is a change to the numbers, so it is applied only when the device demands
    it, and the names it touched are returned for the caller to record.

    Returns the qualified names that were cast.
    """
    import torch

    changed: list[str] = []
    for name, buffer in list(module.named_buffers()):
        if buffer.dtype is not torch.float64:
            continue
        owner = module
        *path, attribute = name.split(".")
        for part in path:
            owner = getattr(owner, part)
        setattr(owner, attribute, buffer.float())
        changed.append(name)
    return changed


@register
class DolphinBackend(ASRBackend):
    name: ClassVar[str] = "dolphin"
    description: ClassVar[str] = "DataoceanAI Dolphin, 40 Eastern languages incl. Burmese"
    install_hint: ClassVar[str] = "uv sync --extra dolphin"
    accepts_language: ClassVar[bool] = True
    is_local: ClassVar[bool] = True

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: str = "auto",
        **options: Any,
    ) -> None:
        if model not in MODELS:
            known = ", ".join(sorted(MODELS))
            raise ValueError(f"Unknown Dolphin model {model!r}. Available: {known}")
        super().__init__(model, **options)
        self.spec = MODELS[model]
        self.device_arg = device
        self.engine: Any = None
        self.resolved_device: str | None = None
        #: Dolphin's checkpoint is float32 throughout; the only precision
        #: decision this adapter makes is the float64 compatibility cast in
        #: `_demote_float64`, which is recorded as a fallback rather than
        #: folded into this value.
        self.resolved_dtype: str | None = None

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        try:
            import dolphin  # noqa: F401
        except ImportError as exc:
            return False, f"missing dependency: {exc.name}"
        return True, "dataoceanai-dolphin"

    def _resolve_device(self) -> str:
        import torch

        if self.device_arg != "auto":
            return self.device_arg
        if torch.cuda.is_available():
            return "cuda"
        # Metal works here — see `_demote_float64` for what it took — but is
        # slower for this model, so CPU is the default. `--device mps` is still
        # worth having: it frees the CPU almost entirely.
        # docs/findings.md#device-defaults
        return "cpu"

    def _resolve_dtype(self, device: str) -> str:
        """Return Dolphin's native precision for provenance preflight.

        Dolphin's checkpoint parameters are float32 on every device.  MPS may
        additionally demote its float64 CMVN buffers at load time; that change
        is recorded as a fallback there, while the resolved model dtype remains
        the same value reported by preflight.
        """
        del device
        return "float32"

    def estimated_download_mb(self) -> int | None:
        return self.spec.approx_mb

    def weights_cached(self) -> bool | None:
        return (cache_dir(self.spec.size) / f"{self.spec.size}.pt").is_file()

    def load(self) -> None:
        import dolphin

        binding = self.model_binding
        if binding is not None:
            bound_files = [Path(str(item.path)) for item in binding.paths if item.path]
            directories = {path.resolve().parent for path in bound_files}
            if len(directories) != 1:
                raise RuntimeError("Dolphin binding artifacts must share one loader directory")
            directory = next(iter(directories))
        else:
            directory = cache_dir(self.spec.size)
        directory.mkdir(parents=True, exist_ok=True)
        self.resolved_device = self._resolve_device()
        self.resolved_dtype = self._resolve_dtype(self.resolved_device)

        with suppress_native_output(not self.options.get("verbose")):
            if self.resolved_device == "mps":
                # Metal has no float64 at all, and `load_model` moves the model
                # to the device itself, so the cast has to happen in between.
                self.engine = dolphin.load_model(self.spec.size, str(directory), "cpu")
                demoted = _demote_float64(self.engine)
                if demoted:
                    # A change to the numbers, so it is provenance, not a
                    # detail: the run is comparable only to other runs that
                    # made the same cast.
                    self.record_fallback(
                        reason="metal has no float64",
                        cast="float64->float32",
                        tensors=demoted,
                    )
                self.engine = self.engine.to("mps")
                # `dolphin.transcribe` places its inputs with `model.device`,
                # a plain string set when the model was built. Moving the
                # module does not update it, so it has to be corrected or every
                # decode fails on a cpu/mps tensor mismatch.
                self.engine.device = "mps"
            else:
                self.engine = dolphin.load_model(
                    self.spec.size, str(directory), self.resolved_device
                )
        self._loaded = True

    def unload(self) -> None:
        self.engine = None
        self._loaded = False
        import gc

        gc.collect()

    def _check_language(self, language: str | None) -> None:
        if language is None:
            return
        if language.split("_")[0] not in {"mya", "my"}:
            raise ValueError(
                f"The {self.name} backend is wired for Burmese (my/MM); got {language!r}."
            )

    def _transcribe_file(self, path: Path) -> list[Segment]:
        """Window the audio at 30 s, since Dolphin decodes in a single pass.

        ``dolphin.transcribe`` does no chunking of its own and the model
        inherits Whisper's fixed 30-second input, so anything longer has to be
        split here. It takes a *path* rather than samples, so each window is
        staged as a temporary wav.
        """
        import tempfile

        import dolphin
        import soundfile as sf

        from stt.audio import windowed

        pcm, rate = sf.read(str(path), dtype="float32", always_2d=False)
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=1)

        with tempfile.TemporaryDirectory(prefix="stt-dolphin-") as tmp:
            counter = itertools.count()

            def decode(chunk) -> str:
                # Dolphin's Python API only accepts a path, so each window has
                # to be written out before it can be decoded.
                window_path = Path(tmp) / f"w{next(counter):05d}.wav"
                sf.write(str(window_path), chunk, rate)
                out = dolphin.transcribe(
                    self.engine,
                    str(window_path),
                    lang_sym=BURMESE_LANG,
                    region_sym=BURMESE_REGION,
                )
                # `text` still carries the <my><MM> control tokens; the
                # nospecial variant is the actual transcript.
                return (getattr(out, "text_nospecial", None) or "").strip()

            return windowed(pcm, rate, WINDOW_SEC, decode)

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
        meta = {"size": self.spec.size, "device": self.resolved_device}

        for path in audio_paths:
            duration = None
            try:
                self._check_language(language)
                duration = duration_of(path)
                started = time.perf_counter()
                with suppress_native_output(not self.options.get("verbose")):
                    segments = self._transcribe_file(path)
                text = join_segments(segments)
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
