"""Burmese-focused speech-to-text evaluation harness."""

from stt.paths import configure_environment

__version__ = "0.1.0"

# Must run before any runtime is imported: huggingface_hub reads HF_HOME once,
# at its own import time. See stt.paths.
configure_environment()
