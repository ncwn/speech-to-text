"""Burmese (မြန်မာ) text handling for fair ASR scoring.

Three things will silently wreck an accuracy number if ignored: Burmese has no
word delimiters (so CER, not WER, and whitespace is stripped); Zawgyi and
Unicode share the same code block and render identically while being different
bytes (so encoding is detected and a mismatch refused rather than scored); and
combining-mark order varies (so NFC first).

See ``docs/burmese.md``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Burmese sentence/phrase delimiters (U+104A little section, U+104B section)
# plus the ASCII punctuation models tend to emit.
_PUNCTUATION = re.compile(r"[၊။,.!?;:\"'`()\[\]{}<>\-–—_/\\|~@#$%^&*+=]")
_WHITESPACE = re.compile(r"\s+")

# Myanmar digits U+1040-U+1049 -> ASCII, so "၂၀၂၅" and "2025" score as equal.
_MYANMAR_DIGITS = str.maketrans("၀၁၂၃၄၅၆၇၈၉", "0123456789")

# The Myanmar script blocks: main, Extended-A, Extended-B. Used to decide
# whether a space sits *between* two Burmese characters (and so is an artifact)
# or borders Latin/digits (where it is load-bearing).
_MYANMAR_CHAR = r"\u1000-\u109F\uA9E0-\uA9FF\uAA60-\uAA7F"
_SPACE_BETWEEN_MYANMAR = re.compile(f"(?<=[{_MYANMAR_CHAR}])[ \t]+(?=[{_MYANMAR_CHAR}])")
_SPACE_BEFORE_DELIM = re.compile(r"[ \t]+([၊။])")

# Probability above which myanmartools' detector is treated as calling Zawgyi.
ZAWGYI_THRESHOLD = 0.5

_detector = None


def _get_detector():
    """Load Google's Zawgyi detector lazily — it builds a Markov model on init."""
    global _detector
    if _detector is None:
        from myanmartools import ZawgyiDetector

        _detector = ZawgyiDetector()
    return _detector


def zawgyi_probability(text: str) -> float:
    """Probability that ``text`` is Zawgyi-encoded rather than Unicode.

    Returns 0.0 for text with no Myanmar characters at all.
    """
    if not has_myanmar(text):
        return 0.0
    return float(_get_detector().get_zawgyi_probability(text))


def is_zawgyi(text: str, threshold: float = ZAWGYI_THRESHOLD) -> bool:
    return zawgyi_probability(text) > threshold


def has_myanmar(text: str) -> bool:
    return any(
        "\u1000" <= ch <= "\u109f" or "\ua9e0" <= ch <= "\ua9ff" or "\uaa60" <= ch <= "\uaa7f"
        for ch in text
    )


@dataclass(frozen=True)
class NormalizeOptions:
    """Knobs for :func:`normalize`. Defaults are the fair-CER settings."""

    nfc: bool = True
    strip_whitespace: bool = True
    strip_punctuation: bool = True
    normalize_digits: bool = True
    lowercase: bool = True  # affects only Latin text mixed into Burmese


def normalize(text: str, options: NormalizeOptions | None = None) -> str:
    """Normalise Burmese text so two transcripts can be compared character by character."""
    opts = options or NormalizeOptions()

    if opts.nfc:
        text = unicodedata.normalize("NFC", text)
    if opts.normalize_digits:
        text = text.translate(_MYANMAR_DIGITS)
    if opts.strip_punctuation:
        text = _PUNCTUATION.sub("", text)
    if opts.lowercase:
        text = text.lower()
    if opts.strip_whitespace:
        text = _WHITESPACE.sub("", text)
    else:
        text = _WHITESPACE.sub(" ", text).strip()

    return text


def describe_encoding(text: str) -> str:
    """Short human-readable encoding verdict, for CLI warnings."""
    if not has_myanmar(text):
        return "no-myanmar"
    p = zawgyi_probability(text)
    return f"zawgyi (p={p:.2f})" if p > ZAWGYI_THRESHOLD else f"unicode (p={p:.2f})"


def tidy_spacing(text: str) -> str:
    """Remove inter-word spaces that some models emit inside Burmese runs.

    Burmese is written without spaces between words; spaces mark phrase
    boundaries at most. SeamlessM4T in particular decodes one space per
    sub-word unit, so ``လူသား တွေ သေဆုံး ပြီး`` comes back where a reader
    expects ``လူသားတွေသေဆုံးပြီး``. That costs nothing at scoring time — CER
    strips whitespace — but it makes the transcript itself look wrong.

    Only spaces with a Myanmar character on *both* sides are dropped, so
    spacing around Latin words, numerals and punctuation survives. A single
    space is kept after ``၊`` and ``။`` because those are the real phrase and
    sentence delimiters.
    """
    text = unicodedata.normalize("NFC", text)
    text = _SPACE_BEFORE_DELIM.sub(r"\1", text)
    # Repeat: each pass removes one space from a run, and NFC leaves no
    # zero-width joiners that would defeat the lookbehind.
    while True:
        collapsed = _SPACE_BETWEEN_MYANMAR.sub("", text)
        if collapsed == text:
            break
        text = collapsed
    text = re.sub(r"([၊။])(?=[^\s])", r"\1 ", text)
    return _WHITESPACE.sub(" ", text).strip()
