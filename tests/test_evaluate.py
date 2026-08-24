"""Tests for scoring behaviour, including the encoding-mismatch guard."""

import pytest

from stt.evaluate import load_references, score_results
from stt.results import TranscriptionResult

UNICODE_REF = "မြန်မာဘာသာစကား"
ZAWGYI_REF = "ျမန္မာဘာသာစကား"


def _result(text: str, path: str = "clip1.wav", error: str | None = None):
    return TranscriptionResult(
        audio_path=path,
        text=text,
        backend="test",
        model="test-model",
        error=error,
        elapsed_s=1.0,
        audio_duration_s=10.0,
    )


def test_perfect_match_scores_zero():
    score = score_results([_result(UNICODE_REF)], {"clip1": UNICODE_REF})
    assert score.cer == 0.0
    assert score.n_failed == 0


def test_spacing_differences_do_not_count_as_errors():
    """The core reason Burmese needs CER-with-whitespace-stripped."""
    score = score_results([_result("မြန်မာ ဘာသာ စကား")], {"clip1": UNICODE_REF})
    assert score.cer == 0.0


def test_encoding_mismatch_is_refused_not_scored():
    """A Zawgyi hypothesis against a Unicode reference must not yield a number."""
    score = score_results([_result(ZAWGYI_REF)], {"clip1": UNICODE_REF})
    assert score.n_failed == 1
    assert "encoding mismatch" in score.items[0].error


def test_encoding_check_can_be_disabled():
    """Without the guard, a cross-encoding comparison yields a plausible-looking
    number that is really measuring encoding, not recognition — which is exactly
    why the guard is on by default."""
    score = score_results([_result(ZAWGYI_REF)], {"clip1": UNICODE_REF}, check_encoding=False)
    assert score.n_failed == 0
    assert score.cer > 0  # visually equivalent encodings still differ in code points


def test_missing_reference_is_reported():
    score = score_results([_result(UNICODE_REF, path="unknown.wav")], {"clip1": UNICODE_REF})
    assert score.n_failed == 1
    assert "no reference" in score.items[0].error


def test_backend_error_propagates():
    score = score_results([_result("", error="boom")], {"clip1": UNICODE_REF})
    assert score.n_failed == 1
    assert score.items[0].error == "boom"


def test_converted_audio_suffix_still_matches_reference():
    """stt.audio appends '.16k' when it resamples; lookup must survive that."""
    score = score_results([_result(UNICODE_REF, path="clip1.16k.wav")], {"clip1": UNICODE_REF})
    assert score.n_failed == 0
    assert score.cer == 0.0


def test_corpus_cer_is_length_weighted():
    """A short bad clip must not outweigh a long good one."""
    long_ref = UNICODE_REF * 10
    results = [
        _result(long_ref, path="long.wav"),  # perfect, many chars
        _result("မမမမ", path="short.wav"),  # wrong, few chars
    ]
    score = score_results(results, {"long": long_ref, "short": UNICODE_REF})
    # Unweighted averaging would give ~0.5; weighting keeps it far lower.
    assert score.cer < 0.15


def test_load_references_skips_header(tmp_path):
    p = tmp_path / "refs.tsv"
    p.write_text(f"audio_id\ttranscript\nclip1\t{UNICODE_REF}\n", encoding="utf-8")
    refs = load_references(p)
    assert refs == {"clip1": UNICODE_REF}


def test_load_references_accepts_paths_as_keys(tmp_path):
    p = tmp_path / "refs.tsv"
    p.write_text(f"data/audio/clip1.wav\t{UNICODE_REF}\n", encoding="utf-8")
    assert load_references(p) == {"clip1": UNICODE_REF}


def test_empty_run_yields_nan_not_crash():
    score = score_results([], {})
    assert score.cer != score.cer  # NaN


@pytest.mark.parametrize("bad", ["", "   "])
def test_empty_hypothesis_scores_total_error(bad):
    score = score_results([_result(bad)], {"clip1": UNICODE_REF})
    assert score.cer == 1.0
