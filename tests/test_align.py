"""Forced-alignment logic.

These exercise the pure parts only — target construction and segment grouping —
with hand-built inputs, so the suite stays fast and works offline. The acoustic
model is verified separately by aligning real audio.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest

from stt.align import (
    AlignedChar,
    LoadedAligner,
    _alignment_load_kwargs,
    _alignment_load_source,
    _load_aligner_components,
    alignment_metadata,
    build_targets,
    group_segments,
    splits_a_cluster,
)
from stt.provenance import ModelBinding, ModelProvenance, digest_file

# A stand-in for MMS's Burmese vocabulary: blank, word delimiter, a few
# Myanmar characters, and the lowercase Latin subset MMS actually ships.
VOCAB = {
    "<pad>": 0,
    "|": 1,
    "က": 10,
    "မ": 11,
    "ာ": 12,
    "ဘ": 13,
    "။": 14,
    "၊": 15,
    "a": 20,
    "b": 21,
}


def _provenance(*, revision_status: str, revision: str | None) -> ModelProvenance:
    return ModelProvenance(
        backend="hf",
        requested_model="mms-1b-all",
        source_kind="huggingface",
        source_locator="facebook/mms-1b-all",
        upstream_revision=revision,
        revision_status=revision_status,
    )


def test_aligner_load_kwargs_are_local_only_and_commit_pinned():
    revision = "a" * 40
    binding = ModelBinding(_provenance(revision_status="pinned", revision=revision))
    assert _alignment_load_kwargs(binding) == {
        "local_files_only": True,
        "revision": revision,
    }

    mutable = ModelBinding(_provenance(revision_status="unknown", revision=None))
    assert _alignment_load_kwargs(mutable) == {"local_files_only": True}


def test_bound_aligner_loads_components_from_exact_local_snapshot(monkeypatch, tmp_path):
    root = tmp_path / "snapshot"
    files = {
        "config.json": b"{}",
        "model.safetensors": b"weights",
        "adapter.mya.safetensors": b"adapter",
        "vocabs/mya.txt": b"vocab",
    }
    artifacts = []
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        artifacts.append(digest_file(path, role="model-or-processor", name=name))
    provenance = replace(
        _provenance(revision_status="pinned", revision="a" * 40),
        artifacts=tuple(artifacts),
    )
    binding = ModelBinding(provenance=provenance, paths=tuple(artifacts))
    binding.validate()

    source, kwargs = _alignment_load_source(binding)
    snapshot = Path(source)
    assert snapshot.is_dir()
    assert all((snapshot / artifact.name).is_symlink() for artifact in artifacts)
    assert all(
        (snapshot / artifact.name).resolve() == Path(artifact.path).resolve()
        for artifact in artifacts
    )
    assert kwargs == {"local_files_only": True}

    calls: list[tuple[str, str, dict]] = []

    class FakeModel:
        def to(self, device):
            return self

        def eval(self):
            return self

    class AutoProcessor:
        @staticmethod
        def from_pretrained(source, **kwargs):
            calls.append(("processor", source, kwargs))
            return object()

    class AutoModelForCTC:
        @staticmethod
        def from_pretrained(source, **kwargs):
            calls.append(("model", source, kwargs))
            return FakeModel()

    class Logging:
        @staticmethod
        def get_verbosity():
            return 0

        @staticmethod
        def set_verbosity_error():
            return None

        @staticmethod
        def set_verbosity(value):
            return None

    transformers = ModuleType("transformers")
    transformers.AutoProcessor = AutoProcessor
    transformers.AutoModelForCTC = AutoModelForCTC
    transformers.logging = Logging
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    _load_aligner_components("cpu", binding)

    assert calls[0] == (
        "processor",
        str(snapshot),
        {"target_lang": "mya", "local_files_only": True},
    )
    assert calls[1] == (
        "model",
        str(snapshot),
        {"target_lang": "mya", "ignore_mismatched_sizes": True, "local_files_only": True},
    )
    assert all(call[1] != "facebook/mms-1b-all" for call in calls)


def test_aligner_rejects_binding_for_a_different_consumer(monkeypatch, tmp_path):
    artifact_path = tmp_path / "config.json"
    artifact_path.write_bytes(b"{}")
    artifact = digest_file(artifact_path, role="model-or-processor", name="config.json")
    provenance = replace(
        _provenance(revision_status="pinned", revision="a" * 40),
        backend="other",
        requested_model="other-model",
        artifacts=(artifact,),
        runtime_packages={"transformers": "4.57.6"},
        adapter_git_commit="c" * 40,
        uv_lock_sha256="d" * 64,
    ).finalized()
    binding = ModelBinding(provenance=provenance, paths=(artifact,))
    binding.validate()
    aligner = LoadedAligner(object(), object(), binding=binding)

    metadata = alignment_metadata(aligner, status="completed")

    assert metadata["trusted"] is False
    assert any("backend mismatch" in issue for issue in metadata["trust_issues"])
    assert any("model mismatch" in issue for issue in metadata["trust_issues"])
    monkeypatch.setattr(
        "stt.backends.transformers_asr._bound_hf_snapshot",
        lambda binding: pytest.fail("wrong-consumer binding reached snapshot resolution"),
    )
    with pytest.raises(RuntimeError, match="consumer mismatch"):
        _alignment_load_source(binding)


def test_alignment_metadata_is_fail_closed_without_binding():
    metadata = alignment_metadata(
        LoadedAligner(object(), object(), provenance_issues=("missing binding",)),
        status="completed",
    )
    assert metadata["backend"] == "hf"
    assert metadata["model"] == "mms-1b-all"
    assert metadata["trusted"] is False
    assert "missing binding" in metadata["trust_issues"]


def _chars(spec: str, start: float = 0.0, step: float = 0.1, score: float = 0.9):
    """Lay `spec` out on a uniform timeline, one AlignedChar per character."""
    return [
        AlignedChar(
            index=i, char=ch, start=start + i * step, end=start + (i + 1) * step, score=score
        )
        for i, ch in enumerate(spec)
    ]


# ------------------------------------------------------------------- targets


def test_targets_map_characters_and_spaces():
    ids, sources, prepared = build_targets("ကမာ ဘာ", VOCAB)
    assert ids == [10, 11, 12, 1, 13, 12]
    assert sources == [0, 1, 2, 3, 4, 5]
    assert prepared == "ကမာ ဘာ"


def test_uppercase_latin_is_lowercased_rather_than_dropped():
    """MMS's Latin subset is lowercase; without this, code-switched text is lost."""
    ids, _, _ = build_targets("AB", VOCAB)
    assert ids == [20, 21]


def test_unmappable_characters_are_dropped_but_keep_their_place():
    ids, sources, _ = build_targets("က☃မ", VOCAB)
    assert ids == [10, 11]
    # The snowman is skipped, so the second token still points at index 2.
    assert sources == [0, 2]


def test_runs_of_whitespace_collapse_to_one_delimiter():
    ids, _, _ = build_targets("က   မ", VOCAB)
    assert ids == [10, 1, 11]


def test_text_with_nothing_alignable_yields_no_targets():
    assert build_targets("☃☃", VOCAB)[0] == []


# ------------------------------------------------------------------ grouping


def test_no_characters_gives_no_segments():
    assert group_segments([], "") == []


def test_a_sentence_delimiter_ends_a_segment():
    text = "ကမာ။ဘာက။"
    segments = group_segments(_chars(text), text)
    assert [s.text for s in segments] == ["ကမာ။", "ဘာက။"]
    assert segments[0].start == pytest.approx(0.0)
    assert segments[0].end == pytest.approx(0.4)
    assert segments[1].start == pytest.approx(0.4)


def test_segments_are_marked_as_inferred_not_measured():
    text = "ကမာ။"
    assert group_segments(_chars(text), text)[0].source == "aligned"


def test_a_long_run_without_punctuation_is_still_broken_up():
    """Otherwise a speaker who never pauses produces one unreadable cue."""
    text = "ကမာဘ" * 40  # 160 characters, no delimiter
    segments = group_segments(_chars(text, step=0.01), text, max_chars=20, max_seconds=99.0)
    assert len(segments) == 8
    assert all(len(s.text) <= 20 for s in segments)


def test_a_long_pause_breaks_a_segment_even_mid_sentence():
    chars = _chars("ကမာဘကမာဘ", step=0.1)
    # Insert a 2 s silence after the fourth character.
    chars = chars[:4] + [
        AlignedChar(c.index, c.char, c.start + 2.0, c.end + 2.0, c.score) for c in chars[4:]
    ]
    segments = group_segments(chars, "ကမာဘကမာဘ", max_chars=16, gap_s=0.5)
    assert len(segments) == 2
    assert segments[0].text == "ကမာဘ"


def test_a_clause_delimiter_only_breaks_once_a_cue_is_long_enough():
    """`၊` is a comma. Breaking on every one gives cues too short to read."""
    text = "ကမ၊ာဘ။"
    segments = group_segments(_chars(text), text, max_chars=40)
    assert [s.text for s in segments] == ["ကမ၊ာဘ။"]


def test_confidence_is_the_mean_over_the_segment():
    chars = [
        AlignedChar(0, "က", 0.0, 0.1, 1.0),
        AlignedChar(1, "မ", 0.1, 0.2, 0.5),
        AlignedChar(2, "။", 0.2, 0.3, 0.0),
    ]
    assert group_segments(chars, "ကမ။")[0].confidence == pytest.approx(0.5)


def test_unalignable_characters_are_carried_into_the_segment_text():
    """The text slice comes from the source, so dropped characters survive."""
    text = "က☃မ။"
    chars = [
        AlignedChar(0, "က", 0.0, 0.1, 0.9),
        AlignedChar(2, "မ", 0.1, 0.2, 0.9),
        AlignedChar(3, "။", 0.2, 0.3, 0.9),
    ]
    assert group_segments(chars, text)[0].text == "က☃မ။"


def test_segments_never_overlap_and_run_forwards():
    text = "ကမာ။ဘာက။ကမာ။"
    segments = group_segments(_chars(text), text)
    assert all(s.end >= s.start for s in segments)
    assert all(segments[i].end <= segments[i + 1].start + 1e-9 for i in range(len(segments) - 1))


# ------------------------------------------------------- Burmese cluster safety


def test_a_cut_before_a_vowel_sign_splits_a_cluster():
    """`ာ` is a combining vowel; orphaning it opens the next cue with a mark."""
    chars = _chars("ကာမ")
    assert splits_a_cluster(chars, 0) is True  # between က and ာ
    assert splits_a_cluster(chars, 1) is False  # between ာ and မ


def test_a_cut_straight_after_a_virama_splits_a_cluster():
    chars = _chars("က္မ")
    assert splits_a_cluster(chars, 1) is True


def test_a_cut_at_the_very_end_is_always_safe():
    chars = _chars("ကမ")
    assert splits_a_cluster(chars, len(chars) - 1) is False


def test_a_forced_break_retreats_off_a_combining_mark():
    """The real failure: no punctuation, so the length limit decides the cut."""
    text = "ကာ" * 20  # every odd index is a combining vowel sign
    segments = group_segments(_chars(text, step=0.01), text, max_chars=9, max_seconds=99.0)
    for s in segments:
        assert not s.text.startswith("ာ")


def test_a_forced_break_prefers_the_longest_pause_in_the_tail():
    chars = _chars("ကမကမကမကမ", step=0.1)
    # Open a 1 s gap after index 5, inside the retreat window for a limit of 8.
    chars = chars[:6] + [
        AlignedChar(c.index, c.char, c.start + 1.0, c.end + 1.0, c.score) for c in chars[6:]
    ]
    segments = group_segments(chars, "ကမကမကမကမ", max_chars=8, max_seconds=99.0, gap_s=99.0)
    assert segments[0].text == "ကမကမကမ"


def test_a_pause_does_not_break_a_cue_mid_cluster():
    """The aligner can leave a gap between a consonant and its own vowel sign."""
    chars = _chars("ကာကာကာကာ", step=0.1)
    # A 2 s pause lands between က (index 4) and its vowel sign ာ (index 5).
    chars = chars[:5] + [
        AlignedChar(c.index, c.char, c.start + 2.0, c.end + 2.0, c.score) for c in chars[5:]
    ]
    for s in group_segments(chars, "ကာကာကာကာ", max_chars=40, gap_s=0.5):
        assert not s.text.startswith("ာ")
