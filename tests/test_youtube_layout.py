from pathlib import Path

from stt.evaluate import load_references

YOUTUBE_ROOT = Path("data/YouTube")
DATASET_ENTRIES = {"README.md", "audio", "references.tsv", "transcripts"}


def test_youtube_datasets_use_canonical_layout():
    assert YOUTUBE_ROOT.is_dir()
    assert all(path.is_dir() for path in YOUTUBE_ROOT.iterdir())

    datasets = [
        dataset
        for channel in YOUTUBE_ROOT.iterdir()
        for dataset in channel.iterdir()
        if dataset.is_dir()
    ]
    assert datasets

    for dataset in datasets:
        entries = {path.name for path in dataset.iterdir()}
        assert entries <= DATASET_ENTRIES, dataset
        # Ignored empty audio directories are absent from clean Git checkouts.
        assert DATASET_ENTRIES - {"audio"} <= entries, dataset

        transcript_dir = dataset / "transcripts"
        transcripts = {path.stem for path in transcript_dir.iterdir() if path.suffix == ".txt"}
        assert transcripts
        assert all(path.suffix == ".txt" for path in transcript_dir.iterdir())

        references = load_references(dataset / "references.tsv")
        assert references.keys() == transcripts

        audio_dir = dataset / "audio"
        if audio_dir.is_dir():
            assert {path.stem for path in audio_dir.iterdir()} <= transcripts
