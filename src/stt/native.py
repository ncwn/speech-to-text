"""Silencing chatty native runtimes.

ggml logs every Metal kernel it compiles, straight to fd 1/2 from C. Python's
``contextlib.redirect_stdout`` cannot see that, so we swap the file descriptors
themselves. Captured output is kept so it can be replayed when something fails —
suppressing logs must never mean losing the traceback that explains a crash.
"""

from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager


class CapturedOutput:
    """Holds whatever the native layer wrote while suppressed."""

    def __init__(self) -> None:
        self.text: str = ""

    def tail(self, lines: int = 20) -> str:
        return "\n".join(self.text.splitlines()[-lines:])


@contextmanager
def suppress_native_output(enabled: bool = True) -> Iterator[CapturedOutput]:
    """Redirect fds 1 and 2 to a temp file for the duration of the block.

    Yields a :class:`CapturedOutput` whose ``text`` is populated on exit, so a
    caller handling an exception can surface the native log alongside it.
    """
    captured = CapturedOutput()
    if not enabled:
        yield captured
        return

    sys.stdout.flush()
    sys.stderr.flush()

    saved_out, saved_err = os.dup(1), os.dup(2)
    with tempfile.TemporaryFile(mode="w+b") as sink:
        try:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            yield captured
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
            os.close(saved_out)
            os.close(saved_err)
            try:
                sink.seek(0)
                captured.text = sink.read().decode("utf-8", errors="replace")
            except OSError:  # pragma: no cover - defensive
                captured.text = ""
