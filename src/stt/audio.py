"""Audio discovery and normalisation.

Every omniASR variant expects 16 kHz mono PCM. Rather than let each backend
reinvent resampling, we normalise once here and hand backends a path to a file
they can read directly.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import struct
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf

TARGET_SAMPLE_RATE = 16_000
CONVERSION_RECIPE_VERSION = "pcm16-mono-16000-v1"
_HASH_BLOCK_SIZE = 1024 * 1024
_PCM_BLOCK_FRAMES = 65_536

if TYPE_CHECKING:
    from stt.results import Segment

#: Containers ffmpeg can decode audio from. A directory scan keeps only these,
#: so anything missing here is dropped — which is why the scan reports what it
#: skipped rather than leaving the caller to notice a short file count.
AUDIO_SUFFIXES = {
    ".wav",
    ".flac",
    ".mp3",
    ".m4a",
    ".m4b",
    ".ogg",
    ".opus",
    ".aac",
    ".wma",
    ".aiff",
    ".aif",
    ".caf",
    ".amr",
    ".mp4",
    ".webm",
    ".mkv",
    ".mov",
    ".avi",
    ".3gp",
}

#: Never worth reporting as "skipped audio" — every directory has some of these.
_IGNORED = {".txt", ".tsv", ".csv", ".json", ".jsonl", ".md", ".srt", ".vtt", ".ds_store", ""}


@dataclass(frozen=True, slots=True)
class PreparedAudio:
    """A source file and the exact waveform handed to an ASR backend."""

    source_path: Path
    prepared_path: Path
    reference_id: str
    source_sha256: str
    audio_id: str


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(_HASH_BLOCK_SIZE):
            digest.update(block)
    return digest.hexdigest()


def canonical_audio_id(path: Path) -> str:
    """Identify decoded audio by its PCM16 samples and stream dimensions.

    Container metadata and the source path deliberately do not participate, so
    two encodings of the same prepared waveform join across independent runs.
    Samples are hashed as little-endian, interleaved signed PCM16.
    """
    digest = hashlib.sha256()
    with sf.SoundFile(str(path)) as audio:
        sample_rate = int(audio.samplerate)
        channels = int(audio.channels)
        frames = int(audio.frames)
        digest.update(struct.pack(">IIQ", sample_rate, channels, frames))

        decoded_frames = 0
        for block in audio.blocks(
            blocksize=_PCM_BLOCK_FRAMES,
            dtype="int16",
            always_2d=True,
        ):
            decoded_frames += len(block)
            little_endian = np.asarray(block, dtype="<i2", order="C")
            digest.update(little_endian.tobytes(order="C"))

    if decoded_frames != frames:
        raise RuntimeError(
            f"Could not read all audio frames from {path}: expected {frames}, got {decoded_frames}"
        )
    return f"pcm16:{sample_rate}:{channels}:{digest.hexdigest()}"


def is_valid_audio_id(value: str | None) -> bool:
    """Return whether ``value`` has the canonical PCM identity wire format."""
    if not value:
        return False
    parts = value.split(":")
    if len(parts) != 4 or parts[0] != "pcm16":
        return False
    try:
        sample_rate, channels = int(parts[1]), int(parts[2])
    except ValueError:
        return False
    digest = parts[3]
    return (
        sample_rate > 0
        and channels > 0
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
    )


def find_audio(paths: list[Path]) -> tuple[list[Path], list[Path]]:
    """Expand files and directories into ``(audio_files, skipped)``.

    A file named explicitly is always taken, whatever its extension — the caller
    asked for it. Inside a directory only known containers are picked, and
    anything else that is plausibly media comes back as ``skipped`` so the
    caller can say so instead of silently transcribing fewer files than the user
    has.
    """
    found: list[Path] = []
    skipped: list[Path] = []
    for p in paths:
        if p.is_dir():
            for child in sorted(p.rglob("*")):
                if not child.is_file() or child.name.startswith("."):
                    continue
                if child.suffix.lower() in AUDIO_SUFFIXES:
                    found.append(child)
                elif child.suffix.lower() not in _IGNORED:
                    skipped.append(child)
        elif p.is_file():
            found.append(p)
        else:
            raise FileNotFoundError(f"No such file or directory: {p}")
    return found, skipped


def probe(path: Path) -> tuple[int, int, float]:
    """Return ``(sample_rate, channels, duration_seconds)`` for an audio file.

    Falls back to ffprobe for formats libsndfile cannot open (mp3, m4a, ...).
    """
    try:
        info = sf.info(str(path))
        return info.samplerate, info.channels, info.duration
    except Exception:
        return _ffprobe(path)


def _ffprobe(path: Path) -> tuple[int, int, float]:
    if shutil.which("ffprobe") is None:
        raise RuntimeError(
            f"Cannot read {path.name}: libsndfile does not support this format and "
            "ffprobe is not installed. Install it with `brew install ffmpeg`."
        )
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels:format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return int(out[0]), int(out[1]), float(out[2])


def needs_conversion(path: Path) -> bool:
    """True unless the file already decodes as 16 kHz mono signed PCM16."""
    if path.suffix.lower() not in {".wav", ".flac"}:
        return True
    try:
        info = sf.info(str(path))
    except sf.LibsndfileError:
        return True
    return info.samplerate != TARGET_SAMPLE_RATE or info.channels != 1 or info.subtype != "PCM_16"


def _valid_cached_conversion(path: Path) -> bool:
    """Return whether ``path`` is a complete 16 kHz mono PCM16 WAV."""
    try:
        info = sf.info(str(path))
        if (
            info.format != "WAV"
            or info.subtype != "PCM_16"
            or info.samplerate != TARGET_SAMPLE_RATE
            or info.channels != 1
            or info.frames <= 0
        ):
            return False
        frames = 0
        with sf.SoundFile(str(path)) as audio:
            for block in audio.blocks(blocksize=_PCM_BLOCK_FRAMES, dtype="int16"):
                frames += len(block)
        return frames == info.frames
    except (OSError, RuntimeError, sf.LibsndfileError):
        return False


def _convert_atomic(path: Path, out: Path, source_sha256: str) -> None:
    """Convert and atomically publish one validated cache entry."""
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{out.stem}.", suffix=".tmp.wav", dir=out.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-i",
                str(path),
                "-ac",
                "1",
                "-ar",
                str(TARGET_SAMPLE_RATE),
                "-c:a",
                "pcm_s16le",
                str(temporary),
            ],
            check=True,
        )
        if file_sha256(path) != source_sha256:
            raise RuntimeError(f"Source audio changed while it was being converted: {path}")
        if not _valid_cached_conversion(temporary):
            raise RuntimeError(f"ffmpeg produced an invalid 16 kHz mono PCM16 WAV for {path}")
        os.replace(temporary, out)
    finally:
        temporary.unlink(missing_ok=True)


def to_16k_mono(
    path: Path,
    cache_dir: Path,
    *,
    source_sha256: str | None = None,
) -> Path:
    """Return a path to a 16 kHz mono WAV version of ``path``.

    Already-conforming files are returned untouched. Converted files are cached
    under a recipe version and source-byte digest, then reused on subsequent
    runs. A benchmark sweep across several backends therefore pays the ffmpeg
    cost once without allowing stale or colliding source paths to share a
    conversion.
    """
    if not needs_conversion(path):
        return path

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            f"{path.name} needs resampling to 16 kHz mono but ffmpeg is not installed. "
            "Install it with `brew install ffmpeg`."
        )

    source_sha256 = source_sha256 or file_sha256(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{CONVERSION_RECIPE_VERSION}__{source_sha256}.wav"
    if out.exists() and _valid_cached_conversion(out):
        return out

    _convert_atomic(path, out, source_sha256)
    return out


def prepare_audio(
    path: Path,
    cache_dir: Path,
    *,
    convert: bool = True,
    reference_id: str | None = None,
) -> PreparedAudio:
    """Prepare one input and bind its source and waveform identities together."""
    source_sha256 = file_sha256(path)
    prepared_path = to_16k_mono(path, cache_dir, source_sha256=source_sha256) if convert else path
    audio_id = canonical_audio_id(prepared_path)
    if file_sha256(path) != source_sha256:
        raise RuntimeError(f"Source audio changed while it was being prepared: {path}")
    return PreparedAudio(
        source_path=path,
        prepared_path=prepared_path,
        reference_id=reference_id if reference_id is not None else path.stem,
        source_sha256=source_sha256,
        audio_id=audio_id,
    )


def duration_of(path: Path) -> float:
    """Duration in seconds, used for real-time-factor reporting."""
    return probe(path)[2]


def split_on_quiet(
    pcm: np.ndarray,
    sample_rate: int,
    window_s: float,
    search_s: float = 2.0,
    frame_ms: int = 20,
) -> list[tuple[int, int]]:
    """Split a waveform into ~``window_s`` slices, cutting where it is quietest.

    Models with a fixed input length (Whisper, Dolphin, SeamlessM4T) have to be
    fed in windows. Cutting on a plain timer slices words in half at every
    boundary, and each truncated word tends to produce a wrong token on both
    sides of the cut. Nudging each boundary to the quietest frame within
    ``search_s`` usually lands it in a pause instead, which costs nothing and
    removes most of those errors.

    This is an energy heuristic, not voice-activity detection: over continuous
    speech with no pause it simply cuts at the least-bad point available.

    Returns a list of ``(start, end)`` sample offsets covering the whole input.
    """
    total = len(pcm)
    window = int(window_s * sample_rate)
    if total <= window:
        return [(0, total)]

    frame = max(1, int(frame_ms * sample_rate / 1000))
    search = int(search_s * sample_rate)

    # Used only to rank candidate cut points, never as a speech/silence verdict.
    usable = (total // frame) * frame
    frames = pcm[:usable].reshape(-1, frame)
    energy = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))

    bounds: list[tuple[int, int]] = []
    start = 0
    while start < total:
        nominal = start + window
        if nominal >= total:
            bounds.append((start, total))
            break

        lo = max(start + window // 2, nominal - search)
        hi = min(total, nominal + search)
        lo_f, hi_f = lo // frame, min(len(energy), hi // frame)
        if hi_f > lo_f:
            cut = (lo_f + int(np.argmin(energy[lo_f:hi_f]))) * frame
        else:
            cut = nominal

        bounds.append((start, cut))
        start = cut

    return bounds


def windowed(
    pcm: np.ndarray,
    sample_rate: int,
    window_s: float,
    decode: Callable[[np.ndarray], str],
    min_s: float = 0.2,
) -> list[Segment]:
    """Decode a waveform window by window, keeping each window's timing.

    Seamless and Dolphin both have a fixed input length, so both have to slice
    long audio and stitch the pieces back together. Sharing the loop here means
    the sample offsets :func:`split_on_quiet` already computed survive into the
    result instead of being thrown away, which is what makes subtitles and
    per-region confidence possible for those backends.

    Windows shorter than ``min_s`` are skipped: a sub-200 ms tail carries no
    intelligible speech and models tend to hallucinate a token for it.

    ``decode`` is called once per window and returns that window's transcript.
    """
    from stt.results import Segment

    segments: list[Segment] = []
    for start, end in split_on_quiet(pcm, sample_rate, window_s):
        chunk = pcm[start:end]
        if len(chunk) < sample_rate * min_s:
            continue
        text = decode(chunk).strip()
        if not text:
            continue
        segments.append(
            Segment(
                text=text,
                start=start / sample_rate,
                end=end / sample_rate,
                source="chunk",
            )
        )
    return segments


def join_segments(segments: list[Segment]) -> str:
    """Concatenate segment texts into one transcript."""
    return " ".join(s.text for s in segments if s.text).strip()
