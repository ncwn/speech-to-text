"""Tests for Burmese text handling — the part most likely to silently
produce a wrong accuracy number."""

from stt.burmese import (
    NormalizeOptions,
    describe_encoding,
    has_myanmar,
    is_zawgyi,
    normalize,
    zawgyi_probability,
)

# Unicode Burmese: "Myanmar language"
UNICODE_TEXT = "မြန်မာဘာသာစကား"

# The same words encoded in Zawgyi. Visually identical, different code points.
ZAWGYI_TEXT = "ျမန္မာဘာသာစကား"


def test_detects_myanmar_script():
    assert has_myanmar(UNICODE_TEXT)
    assert not has_myanmar("hello world")
    assert not has_myanmar("")


def test_detects_myanmar_extended_a_boundaries():
    """Myanmar Extended-A (U+A9E0-U+A9FF) is part of the script."""
    assert has_myanmar("\ua9e0")
    assert has_myanmar("\ua9ff")
    assert not has_myanmar("\ua9df")
    assert not has_myanmar("\uaa00")


def test_non_myanmar_text_is_not_zawgyi():
    assert zawgyi_probability("plain ascii") == 0.0
    assert not is_zawgyi("plain ascii")


def test_distinguishes_zawgyi_from_unicode():
    unicode_p = zawgyi_probability(UNICODE_TEXT)
    zawgyi_p = zawgyi_probability(ZAWGYI_TEXT)
    assert zawgyi_p > unicode_p
    assert not is_zawgyi(UNICODE_TEXT)
    assert is_zawgyi(ZAWGYI_TEXT)


def test_describe_encoding_labels():
    assert describe_encoding(UNICODE_TEXT).startswith("unicode")
    assert describe_encoding(ZAWGYI_TEXT).startswith("zawgyi")
    assert describe_encoding("no burmese here") == "no-myanmar"


def test_whitespace_stripped_by_default():
    """Burmese spacing is inconsistent, so it must not affect CER."""
    spaced = "မြန်မာ ဘာသာ စကား"
    unspaced = "မြန်မာဘာသာစကား"
    assert normalize(spaced) == normalize(unspaced)


def test_whitespace_preserved_when_requested():
    opts = NormalizeOptions(strip_whitespace=False)
    assert " " in normalize("မြန်မာ ဘာသာ", opts)


def test_burmese_punctuation_stripped():
    assert normalize("မြန်မာ။") == normalize("မြန်မာ")
    assert normalize("မြန်မာ၊ဘာသာ") == normalize("မြန်မာဘာသာ")


def test_myanmar_digits_map_to_ascii():
    assert normalize("၂၀၂၅") == "2025"


def test_digits_untouched_when_disabled():
    opts = NormalizeOptions(normalize_digits=False)
    assert normalize("၂၀၂၅", opts) == "၂၀၂၅"


def test_nfc_normalisation_is_applied():
    """Decomposed and composed forms must compare equal."""
    import unicodedata

    decomposed = unicodedata.normalize("NFD", UNICODE_TEXT)
    assert normalize(decomposed) == normalize(UNICODE_TEXT)


# ------------------------------------------------------------ spacing artifacts


def test_tidy_spacing_removes_seamless_subword_spaces():
    from stt.burmese import tidy_spacing

    got = tidy_spacing("လူသား တွေ သေဆုံး ပြီး ရင်")
    assert got == "လူသားတွေသေဆုံးပြီးရင်"


def test_tidy_spacing_keeps_one_space_after_burmese_delimiters():
    from stt.burmese import tidy_spacing

    got = tidy_spacing("ကောင်း တာ လုပ် ရင် ၊ ဆိုး တာ လုပ် ရင် ။")
    assert got == "ကောင်းတာလုပ်ရင်၊ ဆိုးတာလုပ်ရင်။"


def test_tidy_spacing_preserves_spaces_around_latin_and_digits():
    """Only spaces with Burmese on both sides are artifacts."""
    from stt.burmese import tidy_spacing

    assert tidy_spacing("အသက် 80 ကျော် COVID 19 ဖြစ် တယ်") == "အသက် 80 ကျော် COVID 19 ဖြစ်တယ်"


def test_tidy_spacing_leaves_already_correct_text_alone():
    from stt.burmese import tidy_spacing

    text = "လူသားတွေသေဆုံးပြီးရင်ဘာဆက်ဖြစ်မလဲ။"
    assert tidy_spacing(text) == text


def test_tidy_spacing_does_not_change_the_score():
    """CER strips whitespace, so tidying must be scoring-neutral."""
    from stt.burmese import normalize, tidy_spacing

    raw = "လူသား တွေ သေဆုံး ပြီး ရင်"
    assert normalize(tidy_spacing(raw)) == normalize(raw)
