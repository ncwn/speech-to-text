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
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

from stt.provenance import (
    ModelBinding,
    ModelProvenance,
    ProvenanceError,
    collect_runtime_provenance,
    unresolved_execution_issues,
)
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

    #: Optional compatibility names accepted in a preflight binding.  The
    #: canonical value remains ``name``; aliases are explicit so a binding
    #: cannot silently identify a different backend.
    provenance_backend_aliases: ClassVar[tuple[str, ...]] = ()

    #: Optional compatibility model names accepted in a preflight binding.
    provenance_model_aliases: ClassVar[tuple[str, ...]] = ()

    def __init__(self, model: str, **options: Any) -> None:
        self.model = model
        self.options = options
        self._loaded = False
        self.model_binding: ModelBinding | None = None
        self._fallback_history: list[dict[str, Any]] = []
        self._execution_settings: dict[str, Any] = {}

    def _bound_requested_settings(self) -> dict[str, Any]:
        """Return constructor settings as they exist in this runtime instance."""
        current = dict(self.options)
        current.update(self._execution_settings)
        for key, attribute in (
            ("device", "device_arg"),
            ("dtype", "dtype_arg"),
            ("chunk_seconds", "chunk_seconds"),
            ("n_threads", "n_threads"),
        ):
            value = getattr(self, attribute, None)
            if value is not None:
                current.setdefault(key, value)
        return current

    def _bound_resolved_settings(self, provenance: ModelProvenance) -> dict[str, Any]:
        """Capture resolved settings and fail closed when a bound value is unknown."""
        del provenance
        current: dict[str, Any] = {}
        for key, attribute in (
            ("device", "resolved_device"),
            ("dtype", "resolved_dtype"),
            ("n_threads", "n_threads"),
            ("chunk_seconds", "chunk_seconds"),
        ):
            value = getattr(self, attribute, None)
            # CrispASR leaves its native thread default as None; preflight
            # records the effective default as four threads.
            if key == "n_threads" and value is None and self.name == "omniasr-gguf":
                value = 4
            current[key] = value

        spec = getattr(self, "spec", None)
        for key in ("family", "lang", "crisp_backend", "size", "unlimited"):
            if not hasattr(spec, key):
                continue
            current[key] = getattr(spec, key, None)
        return current

    @staticmethod
    def _identity_aliases(
        instance: ASRBackend,
        canonical: object,
        attribute_names: tuple[str, ...],
    ) -> frozenset[str]:
        values = {str(canonical)}
        for attribute_name in attribute_names:
            aliases = getattr(instance, attribute_name, ())
            if isinstance(aliases, str):
                aliases = (aliases,)
            try:
                values.update(str(alias) for alias in aliases)
            except TypeError:
                continue
        return frozenset(values)

    def _binding_identity_issues(self, provenance: ModelProvenance) -> tuple[str, ...]:
        expected_backends = self._identity_aliases(
            self,
            getattr(self, "name", self.__class__.__name__),
            ("provenance_backend_aliases", "backend_aliases"),
        )
        expected_models = self._identity_aliases(
            self,
            getattr(self, "model", ""),
            ("provenance_model_aliases", "model_aliases"),
        )
        issues: list[str] = []
        if provenance.backend not in expected_backends:
            issues.append(
                "model binding backend does not match consumer: "
                f"{provenance.backend!r} not in {sorted(expected_backends)!r}"
            )
        if provenance.requested_model not in expected_models:
            issues.append(
                "model binding requested model does not match consumer: "
                f"{provenance.requested_model!r} not in {sorted(expected_models)!r}"
            )
        return tuple(issues)

    def _binding_setting_issues(
        self,
        provenance: ModelProvenance,
        requested_settings: dict[str, Any],
        resolved_settings: dict[str, Any],
    ) -> tuple[str, ...]:
        issues: list[str] = []
        requested_expected = set(provenance.requested_settings)
        requested_actual = set(requested_settings)
        if requested_expected != requested_actual:
            missing = sorted(requested_expected - requested_actual)
            extra = sorted(requested_actual - requested_expected)
            issues.append(
                "requested setting keys changed from preflight: "
                f"missing={missing!r}, extra={extra!r}"
            )
        for key, expected in provenance.requested_settings.items():
            actual = requested_settings.get(key)
            if actual != expected:
                issues.append(
                    f"requested setting {key!r} changed from preflight: {expected!r} -> {actual!r}"
                )
        resolved_expected = set(provenance.resolved_settings)
        resolved_actual = set(resolved_settings)
        if resolved_expected != resolved_actual:
            missing = sorted(resolved_expected - resolved_actual)
            extra = sorted(resolved_actual - resolved_expected)
            issues.append(
                "resolved setting keys changed from preflight: "
                f"missing={missing!r}, extra={extra!r}"
            )
        for key, expected in provenance.resolved_settings.items():
            actual = resolved_settings.get(key)
            if actual != expected:
                issues.append(
                    f"resolved setting {key!r} changed from preflight: {expected!r} -> {actual!r}"
                )
        return tuple(issues)

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

    def preferred_batch_size(self) -> int:
        """Return the measured default for an ordinary multi-file call."""
        return 1

    def weights_cached(self) -> bool | None:
        """Whether weights are already on disk.

        ``None`` means the backend cannot tell — callers should treat that as
        "possibly not cached" and warn rather than assume either way.
        """
        return None

    def bind_model(self, binding: ModelBinding | None) -> None:
        """Attach an orchestrator-resolved model binding before ``load``."""
        if binding is not None:
            binding.validate()
            identity_issues = self._binding_identity_issues(binding.provenance)
            if identity_issues:
                raise ProvenanceError("; ".join(identity_issues))
        self.model_binding = binding

    def set_execution_settings(self, **settings: Any) -> None:
        """Record call-time settings for comparison with a model binding."""
        self._execution_settings = {
            key: value for key, value in settings.items() if value is not None
        }

    def model_provenance(self) -> ModelProvenance:
        """Return model identity; concrete runtimes should override this."""
        binding = self.model_binding
        if binding is not None:
            provenance = ModelProvenance.from_dict(binding.provenance.to_dict())
            requested_settings = self._bound_requested_settings()
            resolved_settings = self._bound_resolved_settings(provenance)
            issues = [*provenance.issues, *self._binding_identity_issues(provenance)]
            issues.extend(
                self._binding_setting_issues(provenance, requested_settings, resolved_settings)
            )
            if self._fallback_history:
                issues.append("runtime fallback occurred")
            issues.extend(unresolved_execution_issues(resolved_settings))
            return replace(
                provenance,
                requested_settings=requested_settings,
                resolved_settings=resolved_settings,
                fallback_history=tuple(self._fallback_history),
                issues=tuple(dict.fromkeys(issues)),
            ).finalized()
        environment = getattr(self, "_provenance_environment", None)
        return collect_runtime_provenance(self, environment)

    def record_fallback(self, **details: Any) -> None:
        """Retain device/runtime fallback history for provenance."""
        self._fallback_history.append(dict(details))

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
