"""Tests for scoring behaviour, including the encoding-mismatch guard."""

import hashlib
import json
from pathlib import Path

import pytest

from stt.evaluate import corpus_rtf, load_references, mean_rtf, score_results
from stt.results import TranscriptionResult, read_jsonl

UNICODE_REF = "မြန်မာဘာသာစကား"
ZAWGYI_REF = "ျမန္မာဘာသာစကား"


def _result(
    text: str,
    path: str = "clip1.wav",
    error: str | None = None,
    *,
    trusted: bool = True,
    reference_id: str | None = None,
):
    ref_id = reference_id or Path(path).stem.removesuffix(".16k")
    return TranscriptionResult(
        audio_path=path,
        text=text,
        backend="test",
        model="test-model",
        error=error,
        elapsed_s=1.0,
        audio_duration_s=10.0,
        audio_id=f"pcm16:16000:1:{hashlib.sha256(ref_id.encode()).hexdigest()}",
        source_path=path,
        source_sha256=f"source-{ref_id}",
        reference_id=ref_id,
        trusted=trusted,
    )


def test_perfect_match_scores_zero():
    score = score_results([_result(UNICODE_REF)], {"clip1": UNICODE_REF})
    assert score.cer == 0.0
    assert score.rtf == 0.1
    assert score.complete
    assert score.coverage == 1.0
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
    assert score.cer > 0  # identical audio content, non-zero CER: pure artefact


def test_missing_reference_is_reported():
    score = score_results([_result(UNICODE_REF, path="unknown.wav")], {"clip1": UNICODE_REF})
    assert score.n_failed == 2  # unexpected result plus missing expected result
    assert score.n_scored == 0
    assert "no reference" in score.items[0].error
    assert score.cer is None


def test_empty_reference_is_distinguished_from_missing_reference():
    score = score_results([_result(UNICODE_REF)], {"clip1": ""})
    assert score.n_failed == 1
    assert score.items[0].error == "empty reference after normalisation"


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
    assert score.cer is None
    assert score.partial_cer is None
    assert not score.complete


@pytest.mark.parametrize("bad", ["", "   "])
def test_empty_hypothesis_scores_total_error(bad):
    score = score_results([_result(bad)], {"clip1": UNICODE_REF})
    assert score.cer == 1.0


def test_corpus_rtf_weights_by_audio_duration():
    """Long clips must contribute proportionally more than short clips."""
    results = [
        _result(UNICODE_REF, path="short.wav"),
        _result(UNICODE_REF, path="long.wav"),
    ]
    results[0].elapsed_s = 1.0
    results[0].audio_duration_s = 1.0
    results[1].elapsed_s = 10.0
    results[1].audio_duration_s = 100.0

    # Corpus RTF is 11 seconds / 101 seconds, not the unweighted mean of
    # per-file ratios (1.0 and 0.1).
    assert corpus_rtf(results) == pytest.approx(11 / 101)
    assert mean_rtf(results) == pytest.approx(11 / 101)


def test_corpus_rtf_skips_failed_and_incomplete_results():
    results = [
        _result(UNICODE_REF, path="ok.wav"),
        _result(UNICODE_REF, path="failed.wav", error="boom"),
        _result(UNICODE_REF, path="unknown-duration.wav"),
    ]
    results[0].elapsed_s = 2.0
    results[0].audio_duration_s = 4.0
    results[2].elapsed_s = 99.0
    results[2].audio_duration_s = None

    assert corpus_rtf(results) == pytest.approx(0.5)


def test_legacy_result_is_diagnostic_only():
    legacy = TranscriptionResult(
        audio_path="clip1.wav",
        text=UNICODE_REF,
        backend="test",
        model="legacy",
        elapsed_s=1.0,
        audio_duration_s=10.0,
    )
    score = score_results([legacy], {"clip1": UNICODE_REF})

    assert score.partial_cer == 0.0
    assert score.partial_rtf == 0.1
    assert score.cer is None
    assert score.rtf is None
    assert not score.identity_complete
    assert "missing or invalid audio_id" in score.trust_issues


def test_malformed_trusted_value_is_diagnostic_only_after_jsonl_load(tmp_path):
    record = _result(UNICODE_REF).to_dict()
    record["trusted"] = "false"
    path = tmp_path / "malformed-trusted.jsonl"
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    (loaded,) = read_jsonl(path)
    score = score_results([loaded], {"clip1": UNICODE_REF})

    assert loaded.trusted is False
    assert "trusted field must be a JSON boolean" in loaded.trust_issues
    assert score.partial_cer == 0.0
    assert score.cer is None


def test_missing_expected_result_makes_coverage_incomplete():
    score = score_results(
        [_result(UNICODE_REF)],
        {"clip1": UNICODE_REF, "clip2": UNICODE_REF},
    )

    assert score.total == 2
    assert score.n_scored == 1
    assert score.n_failed == 1
    assert score.coverage == 0.5
    assert score.partial_cer == 0.0
    assert score.cer is None


def test_failed_result_is_excluded_from_all_partial_metric_populations():
    ok = _result(UNICODE_REF, path="ok.wav")
    failed = _result("", path="failed.wav", error="decoder failed")
    failed.elapsed_s = 99.0
    failed.audio_duration_s = 1.0

    score = score_results([ok, failed], {"ok": UNICODE_REF, "failed": UNICODE_REF})

    assert score.n_scored == 1
    assert score.partial_cer == 0.0
    assert score.partial_rtf == 0.1
    assert score.cer is None
    assert score.rtf is None


def test_encoding_exclusion_uses_the_same_partial_cer_and_rtf_population():
    ok = _result(UNICODE_REF, path="ok.wav")
    mismatch = _result(ZAWGYI_REF, path="mismatch.wav")
    mismatch.elapsed_s = 99.0
    mismatch.audio_duration_s = 1.0

    score = score_results([ok, mismatch], {"ok": UNICODE_REF, "mismatch": UNICODE_REF})

    assert score.n_scored == 1
    assert score.partial_cer == 0.0
    assert score.partial_rtf == 0.1
    assert score.cer is None


def test_missing_timing_suppresses_cer_and_rtf_together():
    result = _result(UNICODE_REF)
    result.elapsed_s = None
    score = score_results([result], {"clip1": UNICODE_REF})

    assert score.partial_cer == 0.0
    assert score.partial_rtf is None
    assert score.cer is None
    assert score.rtf is None
    assert not score.timing_complete


def test_wer_is_corpus_weighted_not_a_macro_average():
    # Long item: 1 error / 10 words. Short item: 1 / 1. Corpus WER is 2/11;
    # the old macro-average would report 0.55.
    long_ref = "one two three four five six seven eight nine ten"
    long_hyp = "zero two three four five six seven eight nine ten"
    short_ref = "word"
    short_hyp = "wrong"
    score = score_results(
        [
            _result(long_hyp, path="long.wav"),
            _result(short_hyp, path="short.wav"),
        ],
        {"long": long_ref, "short": short_ref},
        options=None,
        check_encoding=False,
    )

    assert score.wer == pytest.approx(2 / 11)
    assert score.wer != pytest.approx((0.1 + 1.0) / 2)
