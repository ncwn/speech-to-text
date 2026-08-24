"""DataoceanAI Dolphin — multilingual ASR for 40 Eastern languages.

Dolphin is an encoder-decoder trained on East, South and Southeast Asian speech.
Burmese is in its language table as ``my`` with region ``MM``.

This adapter exposes the public base and small checkpoints. Upstream language
coverage is linked from ``docs/model-survey.md``.
"""

from __future__ import annotations

import hashlib
import itertools
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from stt.audio import join_segments
from stt.backends.base import ASRBackend
from stt.native import suppress_native_output
from stt.registry import register
from stt.results import Segment, TranscriptionResult

#: Match upstream's 30-second segmentation and timestamp scope.
WINDOW_SEC = 30.0

#: Burmese, as Dolphin's two-level language/region scheme spells it.
BURMESE_LANG = "my"
BURMESE_REGION = "MM"


@dataclass(frozen=True)
class DolphinModel:
    size: str
    params_m: int
    approx_mb: int
    repo: str
    revision: str
    artifacts: tuple[tuple[str, str], ...]


MODELS: dict[str, DolphinModel] = {
    "small": DolphinModel(
        "small",
        372,
        1500,
        "DataoceanAI/dolphin-small",
        "1df1f4f848fa1ad4bfe1fbdd1603ae45b5afb2ae",
        (
            ("small.pt", "4a0c6c636657121ec2a2b656e97e45b29a8b29c92fa3998006e02ab146d8ac51"),
            ("train.yaml", "e2765fc748f683710546fc61a840f49dda7b2189afaa6aed38ec09df5a1a7f08"),
            ("feats_stats.npz", "5a37d00c07d595dbc2479b31be42b3c75de422469a947ce4b7bda193c3b1de7f"),
            ("units.txt", "c3788261a51df1899ea4b210b552cd42139204de72c0ad60f6cebb199078872e"),
            ("bpe.model", "4b9102181ef1a2a3c42ce8fbca8a545ea4a55bce47ba7a5222951ab5bb21bb3c"),
        ),
    ),
    "base": DolphinModel(
        "base",
        140,
        570,
        "DataoceanAI/dolphin-base",
        "4f498c42abc03065b6a8b088800d08eb342b6e35",
        (
            ("base.pt", "688f0cdb26da2684a4eec200a432091920287585e8e332507cbe9c1ab6d77401"),
            ("train.yaml", "8f1ea59adf48e47696e0f71b8f421a8201b464644cdcce2eb4efbec8dd7e2c93"),
            ("feats_stats.npz", "5a37d00c07d595dbc2479b31be42b3c75de422469a947ce4b7bda193c3b1de7f"),
            ("units.txt", "c3788261a51df1899ea4b210b552cd42139204de72c0ad60f6cebb199078872e"),
            ("bpe.model", "4b9102181ef1a2a3c42ce8fbca8a545ea4a55bce47ba7a5222951ab5bb21bb3c"),
        ),
    ),
}

DEFAULT_MODEL = "small"

#: Checkpoints and train.yaml are size-specific, so each model gets its own cache.
CACHE_ROOT = Path(os.environ.get("DOLPHIN_CACHE_DIR", Path.home() / ".cache" / "dolphin"))


def cache_dir(size: str) -> Path:
    return CACHE_ROOT / size


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _demote_float64(module: Any) -> list[str]:
    """Cast a model's float64 buffers to float32, in place.

    Metal does not implement float64. Dolphin stores its CMVN mean and standard
    deviation buffers in that dtype, so they must be demoted before an MPS move.

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
    description: ClassVar[str] = "DataoceanAI Dolphin with Burmese my/MM support"
    install_hint: ClassVar[str] = "uv sync --extra dolphin"
    supported_options: ClassVar[frozenset[str]] = frozenset({"device", "verbose"})

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

    # ------------------------------------------------------------------ setup

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
        # CPU is the automatic default; MPS remains available explicitly.
        return "cpu"

    def estimated_download_mb(self) -> int | None:
        return self.spec.approx_mb

    def _invalid_artifacts(self) -> tuple[str, ...]:
        directory = cache_dir(self.spec.size)
        invalid: list[str] = []
        for filename, expected in self.spec.artifacts:
            path = directory / filename
            try:
                if not path.is_file() or _sha256(path) != expected:
                    invalid.append(filename)
            except OSError:
                invalid.append(filename)
        return tuple(invalid)

    def _has_any_artifact(self) -> bool:
        directory = cache_dir(self.spec.size)
        return any(
            (directory / filename).exists() or (directory / filename).is_symlink()
            for filename, _ in self.spec.artifacts
        )

    def weights_cached(self) -> bool | None:
        return not self._invalid_artifacts()

    def download_weights(self) -> None:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=self.spec.repo,
            revision=self.spec.revision,
            local_dir=cache_dir(self.spec.size),
            allow_patterns=[filename for filename, _ in self.spec.artifacts],
        )
        if invalid := self._invalid_artifacts():
            raise RuntimeError(
                f"Dolphin {self.spec.size} download failed integrity validation: "
                f"{', '.join(invalid)}"
            )

    def load(self) -> None:
        directory = cache_dir(self.spec.size)
        if invalid := self._invalid_artifacts():
            if not self._has_any_artifact():
                self.download_weights()
            else:
                raise RuntimeError(
                    f"Dolphin {self.spec.size} cache is incomplete or failed integrity validation "
                    f"({', '.join(invalid)}). Run `stt models --backend dolphin "
                    f"--download {self.spec.size}`."
                )

        # train.yaml is hash-verified above before the locked dependency's YAML loader sees it.
        import dolphin

        self.resolved_device = self._resolve_device()

        with suppress_native_output(not self.options.get("verbose")):
            if self.resolved_device == "mps":
                # Metal has no float64 at all, and `load_model` moves the model
                # to the device itself, so the cast has to happen in between.
                self.engine = dolphin.load_model(self.spec.size, str(directory), "cpu")
                _demote_float64(self.engine)
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

    # ------------------------------------------------------------- inference

    def _check_language(self, language: str | None) -> None:
        if language is None:
            return
        if language.split("_")[0] not in {"mya", "my"}:
            raise ValueError(
                f"The {self.name} backend is wired for Burmese (my/MM); got {language!r}."
            )

    def _transcribe_file(self, path: Path) -> list[Segment]:
        """Window the audio at 30 s, since Dolphin decodes in a single pass.

        The upstream long-form path uses VAD capped to 30-second segments. This
        adapter uses the same scope without adding a second VAD pass. It takes a
        *path* rather than samples, so each window is staged as a temporary wav.
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
