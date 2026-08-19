"""Backend implementations.

Importing this package registers every backend. Modules here must keep their
heavy or optional imports inside methods so that importing the package never
fails because a runtime is not installed.
"""

from stt.backends import (  # noqa: F401
    dolphin,
    omniasr_gguf,
    omniasr_torch,
    transformers_asr,
)
from stt.backends.base import ASRBackend

__all__ = ["ASRBackend"]
