"""Tests for FLEURS source parsing and tracked reference integrity."""

import csv
from pathlib import Path

import pytest

from stt.datasets import _COLUMNS, _clean_fleurs_field, _read_fleurs_rows


def test_fleurs_parser_keeps_quoted_transcript_in_its_column(tmp_path):
    source = tmp_path / "dev.tsv"
    source.write_text(
        "\t".join(_COLUMNS)
        + "\n"
        + "\t".join(
            [
                "1",
                "clip.wav",
                '"raw with an "" embedded quote"',
                '"normalized with an "" embedded quote"',
                '"token | token"',
                "16000",
                "f",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = list(_read_fleurs_rows(source))

    assert len(rows) == 1
    assert len(rows[0]) == len(_COLUMNS) == 7
    assert _clean_fleurs_field(rows[0][3]) == 'normalized with an " embedded quote'
    assert "token | token" not in _clean_fleurs_field(rows[0][3])


def test_fleurs_parser_rejects_rows_with_wrong_column_count(tmp_path):
    source = tmp_path / "dev.tsv"
    source.write_text("\t".join(_COLUMNS[:-1]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"has 6 columns; expected 7"):
        list(_read_fleurs_rows(source))


@pytest.mark.parametrize(
    "path",
    [Path("data/fleurs/references.tsv"), Path("data/fleurs-test/references.tsv")],
)
def test_tracked_fleurs_references_are_two_column_tsvs(path):
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))

    assert rows
    assert rows[0] == ["audio_id", "transcript"]
    assert all(len(row) == 2 for row in rows)
    assert all("\t" not in row[1] and "|" not in row[1] for row in rows[1:])
