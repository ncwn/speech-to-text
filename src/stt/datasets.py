"""Fetch public Burmese evaluation audio.

FLEURS (``google/fleurs``, config ``my_mm``) is the default source: read speech
with human transcripts in Unicode. The dev split is the default for short
iteration runs.

Downloads land in ``data/fleurs/`` and are gitignored.
"""

from __future__ import annotations

import csv
import tarfile
from pathlib import Path

FLEURS_REPO = "google/fleurs"
BURMESE_CONFIG = "my_mm"

#: Column layout of a FLEURS split TSV.
_COLUMNS = [
    "id",
    "file_name",
    "raw_transcription",
    "transcription",
    "char_split",
    "num_samples",
    "gender",
]


def fetch_fleurs(
    dest: Path,
    split: str = "dev",
    limit: int | None = 20,
    config: str = BURMESE_CONFIG,
    transcript_column: str = "transcription",
) -> tuple[Path, Path]:
    """Download FLEURS audio plus references.

    Args:
        dest: Directory to populate, e.g. ``data/fleurs``.
        split: ``dev`` (smallest), ``test``, or ``train``.
        limit: Keep only the first N clips; ``None`` extracts the whole split.
        config: FLEURS language config; ``my_mm`` is Burmese.
        transcript_column: ``transcription`` (normalised, the usual eval target)
            or ``raw_transcription`` (with punctuation and casing).

    Returns:
        ``(audio_dir, references_tsv)``.
    """
    from huggingface_hub import hf_hub_download

    if transcript_column not in {"transcription", "raw_transcription"}:
        raise ValueError(f"Unknown transcript column: {transcript_column!r}")

    dest.mkdir(parents=True, exist_ok=True)
    audio_dir = dest / "audio"
    audio_dir.mkdir(exist_ok=True)

    tsv_path = Path(
        hf_hub_download(
            repo_id=FLEURS_REPO,
            filename=f"data/{config}/{split}.tsv",
            repo_type="dataset",
        )
    )

    wanted: dict[str, str] = {}
    text_index = _COLUMNS.index(transcript_column)
    with tsv_path.open(encoding="utf-8", newline="") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) <= text_index:
                continue
            wanted[row[1]] = row[text_index].strip().strip('"')
            if limit is not None and len(wanted) >= limit:
                break

    tar_path = Path(
        hf_hub_download(
            repo_id=FLEURS_REPO,
            filename=f"data/{config}/audio/{split}.tar.gz",
            repo_type="dataset",
        )
    )

    extracted: set[str] = set()
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = Path(member.name).name
            if name not in wanted or name in extracted:
                continue
            src = tar.extractfile(member)
            if src is None:
                continue
            (audio_dir / name).write_bytes(src.read())
            extracted.add(name)
            if len(extracted) == len(wanted):
                break

    refs_path = dest / "references.tsv"
    with refs_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["audio_id", "transcript"])
        for name in sorted(extracted):
            writer.writerow([Path(name).stem, wanted[name]])

    missing = set(wanted) - extracted
    if missing:
        raise RuntimeError(
            f"{len(missing)} clip(s) listed in {split}.tsv were absent from the "
            f"archive, e.g. {sorted(missing)[:3]}"
        )

    return audio_dir, refs_path
