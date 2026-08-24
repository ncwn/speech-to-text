"""Tests for ROVER-style hypothesis voting."""

import json

import pytest

from stt.results import TranscriptionResult, read_jsonl
from stt.vote import (
    DEFAULT_WEIGHTS,
    VoteInputError,
    prepare_vote_groups,
    rover,
    verified_audio_id,
)


def _result(
    label: str,
    audio_id: str | None,
    *,
    path: str = "audio.wav",
    elapsed_s: float | None = 1.0,
    error: str | None = None,
    trusted: bool = True,
) -> TranscriptionResult:
    result = TranscriptionResult(
        audio_path=path,
        text=f"text-{label}" if error is None else "",
        backend="test",
        model=label,
        elapsed_s=elapsed_s,
        error=error,
    )
    # Assigned separately so these tests remain compatible while the result
    # schema migration is implemented alongside this module.
    result.audio_id = audio_id
    result.trusted = trusted
    return result


def _id(digit: str = "a") -> str:
    return f"pcm16:16000:1:{digit * 64}"


def test_unanimous_agreement_is_returned_unchanged():
    h = {"a": "မြန်မာ", "b": "မြန်မာ", "c": "မြန်မာ"}
    assert rover(h, "a") == "မြန်မာ"


def test_majority_overrules_the_pivot():
    """The whole point: two systems agreeing beat the pivot's own guess."""
    h = {"pivot": "ကမ", "x": "ကဗ", "y": "ကဗ"}
    assert rover(h, "pivot") == "ကဗ"


def test_pivot_wins_a_tie():
    """Ties must favour the pivot, so voting never degrades a lone disagreement."""
    h = {"pivot": "ကမ", "x": "ကဗ"}
    assert rover(h, "pivot") == "ကမ"


def test_weights_can_outvote_a_numeric_majority():
    h = {"good": "ကဗ", "bad1": "ကမ", "bad2": "ကမ"}
    assert rover(h, "good") == "ကမ"
    assert rover(h, "good", {"good": 3.0}) == "ကဗ"


def test_majority_can_delete_a_character_the_pivot_hallucinated():
    """An empty column is a real candidate, so voting can shorten the pivot."""
    h = {"pivot": "ကခဂ", "x": "ကဂ", "y": "ကဂ"}
    assert rover(h, "pivot") == "ကဂ"


def test_unknown_pivot_is_rejected():
    with pytest.raises(KeyError, match="pivot"):
        rover({"a": "က"}, "missing")


def test_empty_pivot_returns_empty():
    assert rover({"a": "", "b": "ကခ"}, "a") == ""


def test_prepare_removes_subword_spacing_before_aligning():
    """SeamlessM4T spaces every sub-word; omniASR spaces nothing. Without a
    shared spacing convention the aligner matches spaces instead of letters."""
    h = {"pivot": "မြန်မာစကား", "x": "မြန် မာ စကား", "y": "မြန် မာ စကား"}
    assert rover(h, "pivot") == "မြန်မာစကား"
    # With normalisation disabled the spaces survive into the vote.
    assert " " in rover(h, "pivot", prepare=None)


def test_default_weights_rank_by_measured_accuracy():
    """Order matters more than the values; a mis-ordered table silently hurts."""
    assert DEFAULT_WEIGHTS["omniASR_LLM_Unlimited_7B_v2"] > DEFAULT_WEIGHTS["seamless-m4t-v2"]
    assert DEFAULT_WEIGHTS["seamless-m4t-v2"] > DEFAULT_WEIGHTS["omniASR_LLM_Unlimited_3B_v2"]
    assert DEFAULT_WEIGHTS["omniASR_LLM_Unlimited_300M_v2"] > DEFAULT_WEIGHTS["small"]
    assert DEFAULT_WEIGHTS["small"] > DEFAULT_WEIGHTS["mms-1b-all"]


def test_voting_preserves_burmese_sentence_delimiters():
    """tidy_spacing must not eat ၊ and ။ — they are the only structure omniASR lacks."""
    h = {"pivot": "ကောင်းတယ်။ ဟုတ်တယ်။", "x": "ကောင်းတယ်။ ဟုတ်တယ်။"}
    assert "။" in rover(h, "pivot")


# --- verified run preparation -----------------------------------------------


def test_verified_audio_id_never_falls_back_to_a_path():
    legacy = _result("legacy", None, path="same.wav")
    malformed = _result("bad", "same.wav", path="same.wav")

    assert verified_audio_id(legacy) is None
    assert verified_audio_id(malformed) is None
    with pytest.raises(VoteInputError, match="verified audio_id"):
        prepare_vote_groups({"pivot": [legacy], "other": [malformed]}, "pivot")


def test_runs_join_by_verified_audio_id_even_when_paths_differ():
    audio_id = _id()
    groups = prepare_vote_groups(
        {
            "pivot": [_result("pivot", audio_id, path="converted.wav")],
            "other": [_result("other", audio_id, path="original.flac")],
        },
        "pivot",
    )

    assert len(groups) == 1
    assert set(groups[0].results) == {"pivot", "other"}
    assert groups[0].complete


def test_identical_paths_with_different_audio_ids_do_not_join():
    with pytest.raises(VoteInputError, match="coverage differs"):
        prepare_vote_groups(
            {
                "pivot": [_result("pivot", _id("a"), path="same.wav")],
                "other": [_result("other", _id("b"), path="same.wav")],
            },
            "pivot",
        )


def test_strict_preflight_rejects_failures_and_untrusted_inputs():
    audio_id = _id()
    with pytest.raises(VoteInputError) as exc:
        prepare_vote_groups(
            {
                "pivot": [_result("pivot", audio_id, trusted=False)],
                "failed": [_result("failed", audio_id, error="decoder failed")],
            },
            "pivot",
        )

    assert "untrusted" in str(exc.value)
    assert "decoder failed" in str(exc.value)


def test_strict_preflight_rejects_trusted_inputs_with_trust_issues():
    audio_id = _id()
    pivot = _result("pivot", audio_id)
    pivot.trust_issues = ["stale provenance"]

    with pytest.raises(VoteInputError, match="stale provenance"):
        prepare_vote_groups(
            {"pivot": [pivot], "other": [_result("other", audio_id)]},
            "pivot",
        )


def test_malformed_trusted_value_is_rejected_after_jsonl_load(tmp_path):
    audio_id = _id()
    record = _result("pivot", audio_id).to_dict()
    record["trusted"] = "false"
    path = tmp_path / "malformed-trusted.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    (pivot,) = read_jsonl(path)

    with pytest.raises(VoteInputError, match="trusted field must be a JSON boolean"):
        prepare_vote_groups(
            {"pivot": [pivot], "other": [_result("other", audio_id)]},
            "pivot",
        )


def test_partial_vote_propagates_constituent_trust_issues():
    audio_id = _id()
    pivot = _result("pivot", audio_id)
    pivot.trust_issues = ["stale provenance"]

    group = prepare_vote_groups(
        {"pivot": [pivot], "other": [_result("other", audio_id)]},
        "pivot",
        allow_partial=True,
    )[0]

    assert not group.complete
    assert "trust issue from voter pivot: stale provenance" in group.trust_issues


def test_partial_vote_omits_failed_and_missing_nonpivot_voters():
    audio_id = _id()
    group = prepare_vote_groups(
        {
            "pivot": [_result("pivot", audio_id)],
            "failed": [_result("failed", audio_id, error="decoder failed")],
            "missing": [],
        },
        "pivot",
        allow_partial=True,
    )[0]

    assert set(group.usable_results) == {"pivot"}
    assert group.failed_voters == ("failed",)
    assert group.missing_voters == ("missing",)
    assert not group.complete
    assert group.provenance() == {
        "pivot": "pivot",
        "expected_voters": ["pivot", "failed", "missing"],
        "used_voters": ["pivot"],
        "missing_voters": ["missing"],
        "failed_voters": ["failed"],
    }


def test_partial_vote_preserves_a_failed_pivot_for_the_caller():
    audio_id = _id()
    group = prepare_vote_groups(
        {
            "pivot": [_result("pivot", audio_id, error="pivot failed")],
            "other": [_result("other", audio_id)],
        },
        "pivot",
        allow_partial=True,
    )[0]

    assert group.pivot.error == "pivot failed"
    assert "pivot" not in group.usable_results
    assert group.failed_voters == ("pivot",)


def test_vote_elapsed_is_unknown_if_any_used_timing_is_missing():
    audio_id = _id()
    group = prepare_vote_groups(
        {
            "pivot": [_result("pivot", audio_id, elapsed_s=1.5)],
            "other": [_result("other", audio_id, elapsed_s=None)],
        },
        "pivot",
    )[0]
    assert group.elapsed_s is None


def test_vote_elapsed_preserves_a_real_zero():
    audio_id = _id()
    group = prepare_vote_groups(
        {
            "pivot": [_result("pivot", audio_id, elapsed_s=0.0)],
            "other": [_result("other", audio_id, elapsed_s=0.0)],
        },
        "pivot",
    )[0]
    assert group.elapsed_s == 0.0


def test_duplicate_audio_ids_are_rejected_even_in_partial_mode():
    audio_id = _id()
    duplicate = _result("pivot", audio_id)
    with pytest.raises(VoteInputError, match="duplicate audio_id"):
        prepare_vote_groups(
            {"pivot": [duplicate, duplicate], "other": [_result("other", audio_id)]},
            "pivot",
            allow_partial=True,
        )
