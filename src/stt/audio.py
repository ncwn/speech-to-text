"""Audio discovery and normalisation.

Every omniASR variant expects 16 kHz mono PCM. Rather than let each backend
reinvent resampling, we normalise once here and hand backends a path to a file
they can read directly.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SAMPLE_RATE = 16_000

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac", ".mp4", ".webm"}


def find_audio(paths: list[Path]) -> list[Path]:
    """Expand a mix of files and directories into a sorted list of audio files."""
    found: list[Path] = []
    for p in paths:
        if p.is_dir():
            found.extend(
                child
                for child in sorted(p.rglob("*"))
                if child.is_file() and child.suffix.lower() in AUDIO_SUFFIXES
            )
        elif p.is_file():
            found.append(p)
        else:
            raise FileNotFoundError(f"No such file or directory: {p}")
    return found


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
    """True if the file is not already 16 kHz mono in a libsndfile-native format."""
    if path.suffix.lower() not in {".wav", ".flac"}:
        return True
    sample_rate, channels, _ = probe(path)
    return sample_rate != TARGET_SAMPLE_RATE or channels != 1


def to_16k_mono(path: Path, cache_dir: Path) -> Path:
    """Return a path to a 16 kHz mono WAV version of ``path``.

    Already-conforming files are returned untouched. Converted files are cached
    in ``cache_dir`` and reused on subsequent runs, so a benchmark sweep across
    several backends only pays the ffmpeg cost once.
    """
    if not needs_conversion(path):
        return path

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            f"{path.name} needs resampling to 16 kHz mono but ffmpeg is not installed. "
            "Install it with `brew install ffmpeg`."
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    # Include the parent directory name so same-named files in different folders
    # do not collide in the flat cache.
    out = cache_dir / f"{path.parent.name}__{path.stem}.16k.wav"
    if out.exists():
        return out

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
            str(out),
        ],
        check=True,
    )
    return out


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

    # Root-mean-square energy per frame, used only to rank candidate cut points.
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
