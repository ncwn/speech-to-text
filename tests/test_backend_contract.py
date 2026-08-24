"""The promises in `backends/base.py`, made checkable.

Every bug this repo has hit in the wild produced a *success-shaped result*
rather than an exception, so the contract worth pinning is not "does it raise"
but "does it lie": one result per input, in order, and a per-file failure
reported as a result with ``error`` set rather than as an empty transcript that
scores like a bad model.

`conformance_failures` is the shared check. Offline it runs against stubs — one
correct, several deliberately broken, to prove the check has teeth. With weights
present (``pytest -m weights``) it runs against every installed backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytest

from stt.backends.base import ASRBackend
from stt.registry import all_backends
from stt.results import TranscriptionResult


def conformance_failures(
    results: list[TranscriptionResult], inputs: list[Path], backend: str
) -> list[str]:
    """Ways ``results`` breaks the :class:`ASRBackend` contract."""
    problems: list[str] = []

    if len(results) != len(inputs):
        problems.append(f"{len(results)} result(s) for {len(inputs)} input(s)")
        return problems  # positional checks below would be meaningless

    for got, wanted in zip(results, inputs, strict=True):
        if Path(got.audio_path).name != wanted.name:
            problems.append(f"out of order: {Path(got.audio_path).name} for {wanted.name}")

    for result in results:
        name = Path(result.audio_path).name
        if result.error is None and not result.text.strip():
            # The dangerous case: scores as a catastrophically bad model rather
            # than as the failure it is.
            problems.append(f"{name}: empty transcript with no error set")
        if result.error is not None and result.text.strip():
            problems.append(f"{name}: reports an error but also returned text")
        if result.backend != backend:
            problems.append(f"{name}: backend is {result.backend!r}, expected {backend!r}")
        if result.segments is not None and any(s.end < s.start for s in result.segments):
            problems.append(f"{name}: a segment ends before it starts")

    return problems


class StubBackend(ASRBackend):
    """A well-behaved backend, and the base for the misbehaving ones."""

    name: ClassVar[str] = "stub"

    def load(self) -> None:
        self._loaded = True

    def transcribe(
        self, audio_paths: list[Path], language: str | None = None, batch_size: int = 1
    ) -> list[TranscriptionResult]:
        return [
            TranscriptionResult(
                audio_path=str(p), text=f"transcript for {p.name}", backend="stub", model="stub"
            )
            for p in audio_paths
        ]


def _paths(tmp_path: Path, count: int = 3) -> list[Path]:
    made = []
    for i in range(count):
        p = tmp_path / f"clip{i}.wav"
        p.touch()
        made.append(p)
    return made


def test_a_correct_backend_passes(tmp_path):
    paths = _paths(tmp_path)
    assert conformance_failures(StubBackend("stub").transcribe(paths), paths, "stub") == []


def test_dropping_a_file_is_caught(tmp_path):
    """The failure mode where a bad clip vanishes instead of being reported."""

    class Dropping(StubBackend):
        def transcribe(self, audio_paths, language=None, batch_size=1):
            return super().transcribe(audio_paths[:-1])

    paths = _paths(tmp_path)
    problems = conformance_failures(Dropping("stub").transcribe(paths), paths, "stub")
    assert problems == ["2 result(s) for 3 input(s)"]


def test_reordering_is_caught(tmp_path):
    class Shuffled(StubBackend):
        def transcribe(self, audio_paths, language=None, batch_size=1):
            return super().transcribe(list(reversed(audio_paths)))

    paths = _paths(tmp_path)
    problems = conformance_failures(Shuffled("stub").transcribe(paths), paths, "stub")
    assert any("out of order" in p for p in problems)


def test_a_silent_empty_transcript_is_caught(tmp_path):
    """This is the shape that scores as a terrible model instead of a failure."""

    class Silent(StubBackend):
        def transcribe(self, audio_paths, language=None, batch_size=1):
            results = super().transcribe(audio_paths)
            results[1].text = ""
            return results

    paths = _paths(tmp_path)
    problems = conformance_failures(Silent("stub").transcribe(paths), paths, "stub")
    assert problems == ["clip1.wav: empty transcript with no error set"]


def test_an_error_result_is_not_a_failure(tmp_path):
    """Reporting a per-file failure is correct behaviour, not a violation."""

    class Failing(StubBackend):
        def transcribe(self, audio_paths, language=None, batch_size=1):
            results = super().transcribe(audio_paths)
            results[0].text = ""
            results[0].error = "unreadable audio"
            return results

    paths = _paths(tmp_path)
    assert conformance_failures(Failing("stub").transcribe(paths), paths, "stub") == []


def test_backwards_segments_are_caught(tmp_path):
    from stt.results import Segment

    class Backwards(StubBackend):
        def transcribe(self, audio_paths, language=None, batch_size=1):
            results = super().transcribe(audio_paths)
            results[0].segments = [Segment(text="x", start=5.0, end=1.0)]
            return results

    paths = _paths(tmp_path)
    problems = conformance_failures(Backwards("stub").transcribe(paths), paths, "stub")
    assert any("ends before it starts" in p for p in problems)


# --- against the real engines, only when weights are on disk -----------------


def _installed() -> list[str]:
    return [name for name, cls in all_backends().items() if cls.is_available()[0]]


@pytest.mark.weights
@pytest.mark.parametrize("backend_name", _installed() or ["none-installed"])
def test_installed_backends_keep_the_contract(backend_name, tmp_path):
    """One real clip and one corrupt file, through each installed engine."""
    if backend_name == "none-installed":
        pytest.skip("no ASR runtime installed")

    import soundfile as sf

    from stt.registry import get_backend

    clips = sorted(Path("data/fleurs/audio").glob("*.wav"))
    if not clips:
        pytest.skip("no clips; run stt fetch-fleurs")

    # A real clip, plus a file that is a .wav in name only.
    good = clips[0]
    bad = tmp_path / "corrupt.wav"
    bad.write_bytes(b"this is not audio")
    inputs = [good, bad]

    cls = get_backend(backend_name)
    backend: Any = cls()
    backend.load()
    try:
        results = backend.transcribe(inputs, language="mya_Mymr")
    finally:
        backend.unload()

    assert conformance_failures(results, inputs, backend_name) == []
    assert results[1].error is not None, "a corrupt file must be reported, not transcribed"
    assert sf is not None
