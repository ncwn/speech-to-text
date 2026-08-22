"""Recover timestamps and confidence for a transcript by forced alignment.

Our most accurate model is also our most opaque: fairseq2's
``ASRInferencePipeline`` returns a bare ``List[str]`` — no timings, no scores,
so no subtitles and no way to say *which part* to distrust.

Given audio and a transcript, a CTC acoustic model can be constrained to the one
path through its output lattice that spells exactly that transcript; where that
path places each character is where it was spoken, and the probability mass on
it is how well the audio supports it. MMS-1B is the natural aligner: already a
dependency, character-level Burmese adapter, and cheap enough to be an
afterthought on a 7B run. See ``docs/findings.md#confidence``.

Alignment is over characters, not words, because Burmese is written without word
delimiters. See :mod:`stt.burmese`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from stt.provenance import (
    ModelBinding,
    ModelProvenance,
    preflight_model_binding,
    validate_binding,
)
from stt.results import Segment

if TYPE_CHECKING:
    import torch

#: MMS with its Burmese adapter. Character-level, which is what lets us align
#: Burmese text directly instead of going through a pronunciation lexicon.
MMS_REPO = "facebook/mms-1b-all"
MMS_LANG = "mya"
ALIGNMENT_BACKEND = "hf"
ALIGNMENT_MODEL = "mms-1b-all"
ALIGNMENT_LANGUAGE = "mya_Mymr"
ALIGNMENT_DTYPE = "float32"

#: Seconds of audio per forward pass. Bounds peak memory; alignment itself is
#: done over the whole file at once, so this does not affect the result.
FORWARD_WINDOW_S = 30.0

#: Burmese clause and sentence delimiters — the natural places to cut a subtitle.
SENTENCE_END = "။"
CLAUSE_END = "၊"

#: A subtitle cue people can actually read. Segments are cut at a delimiter
#: where possible and forced at these bounds when a speaker runs on.
MAX_SEGMENT_CHARS = 80
MAX_SEGMENT_S = 8.0
#: A pause this long is a segment boundary even mid-sentence.
GAP_S = 0.6


@dataclass(frozen=True)
class AlignedChar:
    """One character of the transcript, located in time."""

    index: int  # position in the source text
    char: str
    start: float
    end: float
    score: float  # 0..1 posterior for this character


class AlignmentError(RuntimeError):
    """Raised when the transcript cannot be aligned to the audio."""


def _aligner_binding_issues(binding: ModelBinding | None) -> tuple[str, ...]:
    """Ensure a binding belongs to this MMS aligner consumer."""
    if binding is None:
        return ()
    provenance = binding.provenance
    issues: list[str] = []
    if provenance.backend != ALIGNMENT_BACKEND:
        issues.append(
            f"aligner binding backend mismatch: expected {ALIGNMENT_BACKEND!r}, "
            f"got {provenance.backend!r}"
        )
    if provenance.requested_model != ALIGNMENT_MODEL:
        issues.append(
            f"aligner binding model mismatch: expected {ALIGNMENT_MODEL!r}, "
            f"got {provenance.requested_model!r}"
        )
    return tuple(issues)


@dataclass(frozen=True)
class LoadedAligner:
    """An MMS aligner plus the immutable identity of the artifacts it loaded."""

    processor: Any
    model: Any
    binding: ModelBinding | None = None
    provenance_issues: tuple[str, ...] = ()
    device: str = "cpu"

    def __iter__(self):
        """Keep the historical ``processor, model = load_aligner()`` API working."""
        yield self.processor
        yield self.model

    @property
    def provenance(self) -> ModelProvenance | None:
        return self.binding.provenance if self.binding is not None else None

    @property
    def trust_issues(self) -> tuple[str, ...]:
        issues = list(self.provenance_issues)
        provenance = self.provenance
        if provenance is None:
            issues.append("aligner provenance is unavailable")
        else:
            issues.extend(provenance.issues)
            if not provenance.complete:
                issues.append("aligner provenance is incomplete or ineligible")
        issues.extend(_aligner_binding_issues(self.binding))
        return tuple(dict.fromkeys(issues))

    @property
    def trusted(self) -> bool:
        provenance = self.provenance
        return bool(provenance is not None and provenance.complete and not self.trust_issues)

    @property
    def result_provenance(self) -> ModelProvenance | None:
        """Return provenance whose backend matches a standalone alignment result."""
        provenance = self.provenance
        if provenance is None:
            return None
        return replace(provenance, backend="align").finalized()


def _coerce_aligner(value: LoadedAligner | tuple[Any, Any], device: str) -> LoadedAligner:
    if isinstance(value, LoadedAligner):
        return value
    processor, model = value
    return LoadedAligner(
        processor=processor,
        model=model,
        device=device,
        provenance_issues=("caller-supplied aligner has no immutable provenance",),
    )


def alignment_metadata(
    aligner: LoadedAligner | tuple[Any, Any] | None,
    *,
    device: str = "cpu",
    status: str = "loaded",
    error: str | None = None,
) -> dict[str, Any]:
    """Serialize one consistent, fail-closed alignment provenance block."""
    loaded = _coerce_aligner(aligner, device) if aligner is not None else None
    provenance = loaded.provenance if loaded is not None else None
    issues = list(loaded.trust_issues) if loaded is not None else ["aligner was not loaded"]
    serialized_provenance: dict[str, Any] | None = None
    if provenance is not None:
        try:
            serialized_provenance = provenance.to_dict()
        except Exception as exc:  # noqa: BLE001 - preserve an explicit untrusted artifact
            issues.append(f"aligner provenance serialization failed: {type(exc).__name__}: {exc}")
    if status != "completed":
        issues.append(f"alignment status is {status}")
    metadata: dict[str, Any] = {
        "backend": ALIGNMENT_BACKEND,
        "model": ALIGNMENT_MODEL,
        "language": ALIGNMENT_LANGUAGE,
        "device": loaded.device if loaded is not None else device,
        "revision": provenance.upstream_revision if provenance is not None else None,
        "status": status,
        "trusted": bool(loaded is not None and loaded.trusted and status == "completed"),
        "trust_issues": list(dict.fromkeys(issues)),
        "provenance": serialized_provenance,
    }
    if error is not None:
        metadata["error"] = error
    return metadata


def _capture_environment() -> dict[str, Any]:
    from stt.measurement import capture_environment

    return capture_environment()


def _preflight_aligner(device: str) -> tuple[ModelBinding | None, tuple[str, ...]]:
    """Resolve the same pinned MMS snapshot and artifact set as the HF backend."""
    options = {
        "device": device,
        "dtype": ALIGNMENT_DTYPE,
        "language": ALIGNMENT_LANGUAGE,
        "batch_size": 1,
    }
    try:
        binding = preflight_model_binding(
            ALIGNMENT_BACKEND,
            ALIGNMENT_MODEL,
            options,
            environment=_capture_environment(),
        )
    except Exception as exc:  # noqa: BLE001 - diagnostic alignment may use a local cache
        return None, (f"aligner provenance preflight failed: {type(exc).__name__}: {exc}",)

    try:
        issues = list(validate_binding(binding))
    except Exception as exc:  # noqa: BLE001 - keep the output explicitly untrusted
        issues = [f"aligner provenance validation failed: {type(exc).__name__}: {exc}"]
    return binding, tuple(dict.fromkeys(issues))


def _load_aligner_components(device: str, binding: ModelBinding | None) -> tuple[Any, Any]:
    """Load MMS from a commit-addressed local snapshot when one is available."""
    import torch
    import transformers
    from transformers import AutoModelForCTC, AutoProcessor

    source, load_kwargs = _alignment_load_source(binding)

    processor = AutoProcessor.from_pretrained(source, target_lang=MMS_LANG, **load_kwargs)

    # MMS ships one checkpoint plus a per-language adapter. Loading it warns
    # that `lm_head` was "newly initialised because the shapes did not match" —
    # the base head covers English's 154 tokens and Burmese's covers 140. The
    # adapter then supplies the correct head, so the warning describes a weight
    # that is immediately replaced. Verified by greedy-decoding Burmese audio.
    verbosity = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()
    try:
        model = AutoModelForCTC.from_pretrained(
            source,
            target_lang=MMS_LANG,
            ignore_mismatched_sizes=True,
            **load_kwargs,
        )
    finally:
        transformers.logging.set_verbosity(verbosity)

    model = model.to(torch.device(device)).eval()
    return processor, model


def _alignment_load_kwargs(binding: ModelBinding | None) -> dict[str, Any]:
    """Build the Transformers kwargs that keep alignment loads offline and pinned."""
    provenance = binding.provenance if binding is not None else None
    load_kwargs: dict[str, Any] = {"local_files_only": True}
    if (
        provenance is not None
        and provenance.revision_status == "pinned"
        and provenance.upstream_revision
    ):
        load_kwargs["revision"] = provenance.upstream_revision
    return load_kwargs


def _alignment_load_source(binding: ModelBinding | None) -> tuple[str, dict[str, Any]]:
    """Use the validated worker snapshot for bound loads and cache-only fallback otherwise."""
    consumer_issues = _aligner_binding_issues(binding)
    if consumer_issues:
        raise RuntimeError("aligner binding consumer mismatch: " + "; ".join(consumer_issues))
    load_kwargs = _alignment_load_kwargs(binding)
    if binding is None:
        return MMS_REPO, load_kwargs

    from stt.backends.transformers_asr import _bound_hf_snapshot

    # The local snapshot is already commit-addressed. Dropping ``revision`` is
    # deliberate: Transformers must resolve every file relative to this exact
    # worker path rather than consulting another Hub cache entry.
    source = str(_bound_hf_snapshot(binding))
    load_kwargs.pop("revision", None)
    return source, load_kwargs


def load_aligner(device: str = "cpu") -> LoadedAligner:
    """Load MMS with immutable provenance, or a local-only diagnostic fallback."""
    binding, issues = _preflight_aligner(device)
    processor, model = _load_aligner_components(device, binding)

    if binding is not None:
        try:
            issues = tuple(dict.fromkeys((*issues, *validate_binding(binding))))
        except Exception as exc:  # noqa: BLE001 - trust remains fail-closed
            issues = tuple(
                dict.fromkeys((*issues, f"post-load aligner provenance validation failed: {exc}"))
            )
    return LoadedAligner(
        processor=processor,
        model=model,
        binding=binding,
        provenance_issues=issues,
        device=device,
    )


def emissions(pcm: np.ndarray, rate: int, processor: Any, model: Any) -> np.ndarray:
    """Frame-level log-probabilities for the whole file, shape ``(frames, vocab)``.

    The forward pass is windowed because activation memory grows with input
    length, but the windows are concatenated before alignment. Aligning the
    whole file in one pass avoids having to guess which slice of the transcript
    belongs to which window — the hard part of chunked forced alignment.
    """
    import torch

    from stt.audio import split_on_quiet

    device = next(model.parameters()).device
    chunks: list[np.ndarray] = []
    for start, end in split_on_quiet(pcm, rate, FORWARD_WINDOW_S):
        piece = pcm[start:end]
        if len(piece) < rate * 0.05:
            continue
        inputs = processor(piece, sampling_rate=rate, return_tensors="pt")
        with torch.inference_mode():
            logits = model(inputs.input_values.to(device)).logits
        chunks.append(torch.log_softmax(logits, dim=-1)[0].cpu().numpy())

    if not chunks:
        raise AlignmentError("audio produced no frames to align against")
    return np.concatenate(chunks, axis=0)


def build_targets(text: str, vocab: dict[str, int]) -> tuple[list[int], list[int], str]:
    """Map transcript characters onto CTC token ids.

    Returns ``(token_ids, source_indices, prepared_text)``. ``source_indices``
    records where each token came from in ``prepared_text``, so characters the
    aligner cannot represent still keep their place in the output.

    MMS's Burmese vocabulary is lowercase, and its word-delimiter token stands
    in for the space. Anything left over — measured at 0.15% on a real Burmese
    transcript, all of it uppercase Latin — is dropped: it cannot be aligned,
    but neighbouring characters still bracket it in time.
    """
    import unicodedata

    prepared = unicodedata.normalize("NFC", text)
    delimiter = vocab.get("|")

    ids: list[int] = []
    sources: list[int] = []
    for i, ch in enumerate(prepared):
        if ch.isspace():
            token = delimiter
        else:
            token = vocab.get(ch)
            if token is None:
                token = vocab.get(ch.lower())  # MMS's Latin subset is lowercase
        if token is None:
            continue
        # A repeated delimiter carries no information and only lengthens the
        # target, so collapse runs of whitespace.
        if token == delimiter and ids and ids[-1] == delimiter:
            continue
        ids.append(token)
        sources.append(i)
    return ids, sources, prepared


def _forced_align(
    log_probs: torch.Tensor, targets: torch.Tensor, blank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Constrain the CTC lattice to the one path spelling ``targets``.

    The deprecation warning torchaudio 2.8 emits here is stale. ``forced_align``
    was slated for deletion in 2.9 as part of the move to a Python-only
    TorchAudio, but the C++-backed operators were reprieved: pytorch/audio#3902
    records that ``forced_align``, ``lfilter``, ``RNNTLoss``, ``CUCTC`` and
    ``overdrive`` were preserved in 2.10 by porting them to PyTorch's stable ABI.
    The warning text shipped in 2.8 predates that reversal.

    So there is nothing to migrate to. It stays isolated in its own function
    anyway, because that is cheap and this is the only third-party call in the
    alignment path.
    """
    import warnings

    import torchaudio.functional as AF

    with warnings.catch_warnings():
        # Suppressed because it is inaccurate, not merely inconvenient.
        warnings.filterwarnings("ignore", message=".*forced_align has been deprecated.*")
        return AF.forced_align(log_probs, targets, blank=blank)


def align_chars(
    text: str,
    pcm: np.ndarray,
    rate: int,
    processor: Any,
    model: Any,
) -> list[AlignedChar]:
    """Locate every character of ``text`` in ``pcm``."""
    import torch
    import torchaudio.functional as AF

    vocab = processor.tokenizer.get_vocab()
    ids, sources, prepared = build_targets(text, vocab)
    if not ids:
        return []

    frames = emissions(pcm, rate, processor, model)
    if len(ids) > len(frames):
        raise AlignmentError(
            f"transcript needs {len(ids)} frames but the audio only provides "
            f"{len(frames)}; the text is too long for this audio"
        )

    blank = processor.tokenizer.pad_token_id or 0
    path, scores = _forced_align(
        torch.from_numpy(frames).unsqueeze(0),
        torch.tensor([ids], dtype=torch.int32),
        blank=blank,
    )
    spans = AF.merge_tokens(path[0], scores[0], blank=blank)
    if len(spans) != len(ids):
        raise AlignmentError(f"aligner returned {len(spans)} spans for {len(ids)} tokens")

    # Frames are uniform in time, so one frame is the audio duration over the
    # number of frames the encoder produced for it.
    seconds_per_frame = (len(pcm) / rate) / len(frames)

    out: list[AlignedChar] = []
    for span, source in zip(spans, sources, strict=True):
        char = prepared[source]
        if char.isspace():
            continue  # the delimiter token is not a character of the transcript
        out.append(
            AlignedChar(
                index=source,
                char=char,
                start=span.start * seconds_per_frame,
                end=span.end * seconds_per_frame,
                score=float(np.exp(span.score)),
            )
        )
    return out


def splits_a_cluster(chars: list[AlignedChar], i: int) -> bool:
    """Whether cutting after ``chars[i]`` would break a Burmese syllable.

    Burmese stacks vowel signs, medials and tone marks onto a base consonant.
    Those marks are combining characters, so a cut immediately before one
    orphans it — the reader sees a dangling ``ာ`` opening the next cue. The
    virama (U+1039) binds the consonant that follows it, so a cut straight
    after one is wrong for the same reason.
    """
    import unicodedata

    if i + 1 >= len(chars):
        return False
    if chars[i].char == "\u1039":  # virama: the next consonant belongs to this cluster
        return True
    return unicodedata.category(chars[i + 1].char) in {"Mn", "Mc"}


def _best_break(chars: list[AlignedChar], start_i: int, limit_i: int) -> int:
    """Pick the least-bad cut at or before ``limit_i`` when nothing punctuates.

    Prefers the longest pause in the tail of the cue, since a pause is where a
    speaker actually stopped. Falls back to the latest position that at least
    keeps syllables intact, and only cuts blind if there is no such position.
    """
    window = max(1, (limit_i - start_i) // 3)
    candidates = range(limit_i, max(start_i, limit_i - window) - 1, -1)

    best, best_gap = None, -1.0
    for i in candidates:
        if splits_a_cluster(chars, i):
            continue
        if best is None:
            best = i  # latest safe cut, used when every gap is negligible
        gap = chars[i + 1].start - chars[i].end if i + 1 < len(chars) else 0.0
        if gap > best_gap:
            best, best_gap = i, gap
    return best if best is not None else limit_i


def group_segments(
    chars: list[AlignedChar],
    text: str,
    max_chars: int = MAX_SEGMENT_CHARS,
    max_seconds: float = MAX_SEGMENT_S,
    gap_s: float = GAP_S,
) -> list[Segment]:
    """Gather aligned characters into readable, sentence-shaped cues.

    Breaks after ``။``, and — only when a cue is already long — after ``၊`` or
    across a pause. Characters that could not be aligned are still carried in
    the text, since they sit between characters that could.
    """
    if not chars:
        return []

    segments: list[Segment] = []
    start_i = 0  # index into `chars` of the current cue's first character

    def flush(end_i: int) -> None:
        group = chars[start_i:end_i]
        if not group:
            return
        # Slice the *source* text so unalignable characters survive.
        piece = text[group[0].index : group[-1].index + 1].strip()
        if not piece:
            return
        segments.append(
            Segment(
                text=piece,
                start=group[0].start,
                end=group[-1].end,
                confidence=float(np.mean([c.score for c in group])),
                source="aligned",
            )
        )

    i = 0
    while i < len(chars):
        ch = chars[i]
        held = i - start_i + 1
        elapsed = ch.end - chars[start_i].start
        gap_next = chars[i + 1].start - ch.end if i + 1 < len(chars) else 0.0

        at_limit = held >= max_chars or elapsed >= max_seconds
        # A pause only ends a cue if it also falls between syllables. The
        # aligner can leave a gap between a consonant and its own vowel sign
        # when it is unsure, and cutting there orphans the mark.
        at_pause = gap_next >= gap_s and held >= max_chars // 4
        boundary = (
            ch.char == SENTENCE_END
            or (ch.char == CLAUSE_END and held >= max_chars // 2)
            or (at_pause and not splits_a_cluster(chars, i))
            or at_limit
        )
        if boundary:
            # A model that emits no punctuation (omniASR emits none at all)
            # hits the length limit instead, and cutting blindly there lands
            # mid-syllable. Retreat to the best nearby seam.
            cut = i if not at_limit else _best_break(chars, start_i, i)
            flush(cut + 1)
            start_i = cut + 1
            i = cut + 1
            continue
        i += 1

    flush(len(chars))
    return segments


def align(
    text: str,
    audio: Path,
    device: str = "cpu",
    aligner: LoadedAligner | tuple[Any, Any] | None = None,
    **grouping: Any,
) -> list[Segment]:
    """Time ``text`` against ``audio`` and return readable segments.

    Pass ``aligner`` to reuse a loaded model across many files.
    """
    import soundfile as sf

    if not text.strip():
        return []

    loaded = load_aligner(device) if aligner is None else _coerce_aligner(aligner, device)
    processor, model = loaded
    pcm, rate = sf.read(str(audio), dtype="float32", always_2d=False)
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)

    chars = align_chars(text, pcm, rate, processor, model)
    import unicodedata

    return group_segments(chars, unicodedata.normalize("NFC", text), **grouping)
