# Historical measured findings

This is the only documentation file that owns local CER, timing, memory, and
derived experiment results.

## Status

The numbers below were recorded during local development on an M2 Max, but
current `main` does not track the result JSONL, immutable model revisions,
waveform identities, or raw timing artifacts needed to verify them from a clean
clone. Treat every table in this file as **historical and unverified**, not as a
current baseline or product claim.

Future measured updates belong here only after their inputs, environment,
commands, and validation are reproducible. Other documents should link here
instead of copying results.

Unless a section says otherwise, Burmese accuracy used corpus CER with
whitespace and punctuation removed. See [burmese.md](burmese.md) for the
normalization contract.

## FLEURS test observations

The recorded comparison used the same 120 Burmese FLEURS test clips, totaling
29.6 minutes. RTF is wall time divided by audio duration.

| Model | Backend/device | CER | RTF | Licence |
|---|---|---:|---:|---|
| omniASR 7B Unlimited v2 fp32 | fairseq2/CPU | 0.1017 | 1.79 | Apache-2.0 |
| omniASR 3B Unlimited v2 fp32 | fairseq2/CPU | 0.1196 | 1.54 | Apache-2.0 |
| omniASR 300M Unlimited v2 fp32 | fairseq2/CPU | 0.1298 | 1.29 | Apache-2.0 |
| SeamlessM4T v2 large | Transformers/MPS | 0.1301 | 0.16 | CC-BY-NC-4.0 |
| Dolphin small | Dolphin/CPU | 0.1722 | 0.18 | Apache-2.0 |
| MMS-1B | Transformers/MPS | 0.1796 | 0.04 | CC-BY-NC-4.0 |

The same development record showed that a 12-clip dev screen materially changed
the ranking. Small runs are useful for smoke testing, not model selection.

## Held-out observation

A hand-corrected 16.8-minute Burmese narration was used as a second, private
corpus. The copyrighted audio is not tracked, so the table cannot be audited
from this repository.

| Model | CER |
|---|---:|
| omniASR 7B fp32 | 0.0857 |
| SeamlessM4T v2 | 0.0887 |
| Dolphin small | 0.1366 |
| omniASR 300M Q4_K | 0.2245 |

A repeated proper noun dominated the gap between the two leading systems.
Correcting only that name changed the recorded scores to 0.0831 for omniASR and
0.0725 for SeamlessM4T. The durable lesson is that corpus CER can conceal a
small number of repeated, high-impact errors.

CER also removed punctuation before scoring. The historical outputs showed
substantial differences in sentence-boundary behavior, so punctuation must be
reviewed separately when choosing a transcriber.

## Ensemble and routing observations

The recorded vote used the strongest run as the alignment pivot and normalized
model-specific Burmese spacing before voting.

| Corpus | Best single | Voted |
|---|---:|---:|
| FLEURS 120 test clips | 0.1017 | 0.0930 |
| held-out narration | 0.0857 | 0.0714 |

These are the later values recorded during development; an older README copy
contained different values. Neither version has tracked supporting runs on
`main`, which is why the result is retained only here as history.

Confidence-driven routing and transcript splicing were also explored. The
recorded work found that low-confidence regions concentrated errors, but
fine-grained switching lost or duplicated characters where model segment
boundaries did not agree. Routing therefore uses coarse, adjacent blocks. Its
constants are heuristics from that workload, not portable performance
guarantees.

## Device and precision observations

Short development runs established the current runtime policies:

- `omniasr-torch` selects MPS on Apple Silicon and falls back to CPU if a Metal
  decode fails.
- GPU half precision is selected by a local dtype probe rather than a hardcoded
  chip table.
- CPU uses float32 when memory permits; half precision is a memory fallback.
- The Transformers backend keeps float32 by default.
- Dolphin supports MPS but defaults to CPU.

These are implemented behaviors and tested as such. Their original throughput
figures are not repeated in runtime comments because they were machine-specific
and lack tracked raw artifacts.

## Long-audio observations

Historical long-form runs motivated three current safeguards:

- GGUF Q4_K output sometimes entered repetition loops, so `stt.quality` detects
  repeated spans independently of CER.
- Backends with fixed input windows cut near quiet audio and retain segment
  timing for alignment and subtitles.
- Forced subtitle breaks retreat to Burmese cluster boundaries so a cue does not
  begin with a detached combining mark.

The safeguards are covered by offline tests. Their original effect sizes remain
unverified historical context and are intentionally not copied into code or
other documentation.
