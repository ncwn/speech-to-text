"""Tests for ROVER-style hypothesis voting."""

import pytest

from stt.vote import DEFAULT_WEIGHTS, rover


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


def test_default_weights_keep_the_historical_ranking():
    """Order matters more than the values; a mis-ordered table silently hurts."""
    assert DEFAULT_WEIGHTS["omniASR_LLM_Unlimited_7B_v2"] > DEFAULT_WEIGHTS["seamless-m4t-v2"]
    assert DEFAULT_WEIGHTS["seamless-m4t-v2"] > DEFAULT_WEIGHTS["omniASR_LLM_Unlimited_3B_v2"]
    assert DEFAULT_WEIGHTS["omniASR_LLM_Unlimited_300M_v2"] > DEFAULT_WEIGHTS["small"]
    assert DEFAULT_WEIGHTS["small"] > DEFAULT_WEIGHTS["mms-1b-all"]


def test_voting_preserves_burmese_sentence_delimiters():
    """tidy_spacing must preserve Burmese phrase and sentence delimiters."""
    h = {"pivot": "ကောင်းတယ်။ ဟုတ်တယ်။", "x": "ကောင်းတယ်။ ဟုတ်တယ်။"}
    assert "။" in rover(h, "pivot")
