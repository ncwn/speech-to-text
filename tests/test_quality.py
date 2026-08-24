"""Reference-free defect detection.

These guard the one failure mode CER barely penalises: a decoder that gets
stuck. A clause emitted four times costs a few percent of CER but ruins the
transcript, so this has to be caught independently of scoring.
"""

from __future__ import annotations

from stt.quality import find_loops, loop_summary

# Long enough to exceed the default 30-character n-gram window.
CLAUSE = "ဤသည်မှာ အလွန်ရှည်လျားသော စာကြောင်းတစ်ခုဖြစ်ပါသည်"
OTHER = "နောက်ထပ် မတူညီသော စာကြောင်းတစ်ခု ဤနေရာတွင် ရှိပါသည်"


def test_clean_output_reports_nothing():
    assert find_loops(CLAUSE + OTHER) == []
    assert loop_summary([]) == ""


def test_a_repeated_clause_is_found():
    sites = find_loops(CLAUSE + CLAUSE + CLAUSE)
    assert sites
    assert sites[0].count >= 2


def test_overlapping_ngrams_from_one_loop_collapse_to_one_site():
    """Four copies of a clause is one defect, not a dozen findings."""
    assert len(find_loops(CLAUSE * 4)) == 1


def test_a_tight_syllable_loop_is_found():
    """A short repeated syllable sequence is still detectable."""
    assert find_loops("ဖြစ်တဲ့" * 20)


def test_repetition_the_reference_also_has_is_not_a_defect():
    """A genuine refrain must not be reported just because it recurs."""
    assert find_loops(CLAUSE * 3, reference=CLAUSE * 3) == []


def test_repetition_beyond_the_reference_is_still_reported():
    sites = find_loops(CLAUSE * 5, reference=CLAUSE * 2)
    assert sites
    assert sites[0].excess > 0


def test_text_shorter_than_the_window_is_never_a_loop():
    assert find_loops("ကမ္ဘာ") == []


def test_summary_counts_sites_and_excess_characters():
    summary = loop_summary(find_loops(CLAUSE * 4))
    assert "1 repeated span" in summary
    assert "excess characters" in summary
