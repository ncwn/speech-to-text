"""The interface every ASR backend implements.

Adding a new engine (Dolphin, ElevenLabs Scribe v2, Google Chirp 3) means
writing one module in this package that subclasses :class:`ASRBackend` and
calls :func:`stt.registry.register`. Nothing else in the codebase changes.

Backends must not import heavy or optional dependencies at module scope —
do it inside ``load()`` or ``is_available()`` so that ``stt backends`` keeps
working when a given runtime is not installed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from stt.results import TranscriptionResult


class ASRBackend(ABC):
    """A speech-to-text engine that can transcribe a batch of audio files."""

    #: Short identifier used on the command line, e.g. ``omniasr-torch``.
    name: ClassVar[str]

    #: One-line description shown by ``stt backends``.
    description: ClassVar[str] = ""

    #: How to install this backend, shown when it is unavailable.
    install_hint: ClassVar[str] = ""

    #: Whether this backend accepts a language code such as ``mya_Mymr``.
    accepts_language: ClassVar[bool] = True

    #: Whether the engine runs locally (affects whether audio leaves the machine).
    is_local: ClassVar[bool] = True

    def __init__(self, model: str, **options: Any) -> None:
        self.model = model
        self.options = options
        self._loaded = False

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        """Return ``(available, reason)``.

        ``reason`` explains what is missing when ``available`` is False, and may
        carry a useful detail (such as a resolved binary path) when it is True.
        """
        return True, ""

    def estimated_download_mb(self) -> int | None:
        """Approximate download size for this model, or None if not known.

        Used to warn before a command kicks off a multi-gigabyte fetch.
        """
        return None

    def weights_cached(self) -> bool | None:
        """Whether weights are already on disk.

        ``None`` means the backend cannot tell — callers should treat that as
        "possibly not cached" and warn rather than assume either way.
        """
        return None

    @abstractmethod
    def load(self) -> None:
        """Load weights / establish a session. Called once before transcribing."""

    @abstractmethod
    def transcribe(
        self,
        audio_paths: list[Path],
        language: str | None = None,
        batch_size: int = 1,
    ) -> list[TranscriptionResult]:
        """Transcribe files, returning one result per input in the same order.

        A per-file failure must be reported as a result with ``error`` set
        rather than raised, so one bad clip does not abort a long sweep.
        """

    def unload(self) -> None:
        """Release weights. Default is a no-op; override when it matters."""
        self._loaded = False

    def __enter__(self) -> ASRBackend:
        if not self._loaded:
            self.load()
            self._loaded = True
        return self

    def __exit__(self, *exc: object) -> None:
        self.unload()
