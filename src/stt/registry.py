"""Backend registry.

Backends register themselves at import time; :mod:`stt.backends` imports every
backend module so that a single ``import stt.backends`` populates the registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stt.backends.base import ASRBackend

_REGISTRY: dict[str, type[ASRBackend]] = {}


def register(cls: type[ASRBackend]) -> type[ASRBackend]:
    """Class decorator that adds a backend to the registry."""
    if not getattr(cls, "name", None):
        raise ValueError(f"{cls.__name__} must define a `name` class attribute")
    if cls.name in _REGISTRY:
        raise ValueError(f"Backend name {cls.name!r} is already registered")
    _REGISTRY[cls.name] = cls
    return cls


def get_backend(name: str) -> type[ASRBackend]:
    import stt.backends  # noqa: F401  (populates the registry)

    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown backend {name!r}. Available: {known}")
    return _REGISTRY[name]


def all_backends() -> dict[str, type[ASRBackend]]:
    import stt.backends  # noqa: F401

    return dict(sorted(_REGISTRY.items()))
