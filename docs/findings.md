# Measured findings and historical context

Generated tables in this file are current only when their evidence blocks verify
from typed artifacts through `stt evidence`. Historical sections remain for the
hypotheses that motivated the harness, with an explicit status in place of any
unverified table.

All measurements are from an M2 Max (12 cores, 64 GB) unless stated otherwise.
Burmese is scored as **CER with whitespace stripped** — see
[burmese.md](burmese.md) for why WER does not apply.

**Read the measurement base before comparing two numbers.** Three different
bases appear below and they are not interchangeable:

Trusted transcription JSONL supplies accuracy and transcript-derived metrics;
`experiment-v1` supplies measured diagnostic contrasts; and `baseline-v2`
supplies common-wall timing and regression statistics.

---

## Input preparation control

The paired MMS control presents the same FLOAT WAV either through a fresh canonical
PCM16 conversion or directly through native source ingestion. Preparation is timed
outside every warm inference repeat and the two arms join by source SHA:

<!-- stt-evidence:input-control:start -->
| Condition | Preparation wall s | RTF | CI low | CI high | Text changes |
| --- | ---: | ---: | ---: | ---: | ---: |
| canonical | 0.0394 | 0.0239 | 0.0238 | 0.0244 | 1 |
| native | 0.0031 | 0.0238 | 0.0238 | 0.0245 | 1 |
<!-- stt-evidence:input-control:end -->

Native and canonical ingestion changed the transcript for the same source bytes,
so they are not semantically interchangeable. The timing ratio remains descriptive;
publication accuracy continues to use canonical prepared audio.

---

## Baseline

The accepted baseline-v2 measures the four trusted subjects over the same five
FLEURS Burmese `dev` clips. RTF is the synchronized complete-corpus wall divided
by corpus duration; intervals bootstrap five isolated worker sessions.

<!-- stt-evidence:baseline:start -->
| Subject | RTF | CI low | CI high | Peak RSS MB | Sessions | Repeats |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hf/mms-1b-all | 0.0247 | 0.0245 | 0.0252 | 625 | 5 | 15 |
| hf/seamless-m4t-v2 | 0.1623 | 0.1537 | 0.1657 | 1196 | 5 | 15 |
| dolphin/small | 0.1743 | 0.1673 | 0.1779 | 6171 | 5 | 15 |
| omniasr-torch/omniASR_LLM_Unlimited_7B_v2 | 0.4236 | 0.4138 | 0.4297 | 25433 | 5 | 15 |
<!-- stt-evidence:baseline:end -->

### Trusted accuracy refresh

The regenerated 120-clip accuracy lane is identity-verified for all four trusted
subjects. RTF is intentionally omitted here; common-wall timing comes from
baseline-v2 artifacts.

<!-- stt-evidence:accuracy-fleurs-test-120:start -->
| Model | Backend | CER | N | Scored |
| --- | --- | ---: | ---: | ---: |
| MMS-1B (verified accuracy) | `hf` | 0.1565 | 120 | 120 |
| Dolphin small (verified accuracy) | `dolphin` | 0.1480 | 120 | 120 |
| SeamlessM4T v2 (verified accuracy) | `hf` | 0.1036 | 120 | 120 |
| omniASR 7B (verified accuracy) | `omniasr-torch` | 0.0745 | 120 | 120 |
<!-- stt-evidence:accuracy-fleurs-test-120:end -->

<!-- stt-evidence:accuracy-fleurs-dev-12:start -->
| Model | Backend | CER | N | Scored |
| --- | --- | ---: | ---: | ---: |
| SeamlessM4T v2 (verified accuracy) | `hf` | 0.0594 | 12 | 12 |
| omniASR 7B (verified accuracy) | `omniasr-torch` | 0.0568 | 12 | 12 |
| MMS-1B (verified accuracy) | `hf` | 0.1448 | 12 | 12 |
| Dolphin small (verified accuracy) | `dolphin` | 0.0933 | 12 | 12 |
<!-- stt-evidence:accuracy-fleurs-dev-12:end -->

The timing table and accuracy tables deliberately use different corpora and source
types. Timing comes only from the verified baseline-v2 archive; CER comes only from
trusted transcription JSONL joined to the tracked reference identities.

Reproduce:

```bash
uv run stt bench --verify
uv run stt evidence --check
```

### Which to use

**omniASR 7B** is the most accurate, is Apache-2.0, and runs on Metal. It needs
31.2 GB on disk and about 14 GB resident on the GPU, and it emits **no
punctuation at all**. Use it when licence and raw accuracy matter and you can
post-process sentence breaks.

**SeamlessM4T v2** is the fastest at RTF 0.16 and segments output into real
sentences. On held-out audio it is *more* accurate than the 7B once a single
proper noun is set aside ([Held-out](#held-out)). The catch is the licence:
**CC-BY-NC**, evaluation only.

**Dolphin small** is the Apache-2.0 fallback if you need permissive licensing and
can accept ~30 % more error — though it never touches the GPU and costs 4.9 GB of
RAM to do it.

---

## Tails

For long audio the tail matters more than the average, because one runaway
segment contaminates everything after it. Same 120 clips:

<!-- stt-evidence:tails:start -->
| Run | Corpus CER | Clip CER p90 | CER > 0.3 % | CER > 0.3 N | N |
| --- | ---: | ---: | ---: | ---: | ---: |
| omniASR 7B | 0.0745 | 0.1520 | 0.0 | 0 | 120 |
| SeamlessM4T v2 | 0.1036 | 0.2027 | 2.5 | 3 | 120 |
| Dolphin small | 0.1480 | 0.2585 | 5.0 | 6 | 120 |
| MMS-1B | 0.1565 | 0.2665 | 4.2 | 5 | 120 |
<!-- stt-evidence:tails:end -->

The 7B has the best corpus score and tail, with no clip crossing the declared
failure threshold. The other three subjects retain measurable tail failures,
which matters when many clips are concatenated into a long workflow.

---

## Held-out

FLEURS is public and may be in these models' training data, so the ranking was
re-checked against a hand-corrected transcript of a 16.8-minute Burmese film
recap that none of them has seen ([data/reference](../data/reference/README.md)).
The FLEURS ordering held:

<!-- stt-evidence:held-out:start -->
Held-out audio and full transcript JSONL are deliberately untracked under the redistribution policy; a clean clone cannot audit a numeric table.
<!-- stt-evidence:held-out:end -->

Both leaders score *better* here than on FLEURS, so the contamination worry did
not materialise into an inflated ranking.

**But the top two are separated entirely by one proper noun.** The protagonist's
name occurs 99 times — 2.9 % of the transcript. The reference spells it `ဂျုံး`
(Joon). omniASR 7B produces `ဂျုံး` 78 times; SeamlessM4T produces `ဂျွန်`
(John) 92 times and `ဂျုံး` never. The two spellings differ in three characters,
so every mention costs three substitutions — 277 of Seamless's 810 substitutions
come from that one word.

Correct the name and the ranking inverts:

<!-- stt-evidence:held-out-name-corrected:start -->
The corrected-name held-out result depends on deliberately untracked copyrighted transcript artifacts and remains diagnostic only.
<!-- stt-evidence:held-out-name-corrected:end -->

So SeamlessM4T is about 13 % more accurate on running text, and omniASR 7B wins
overall only because it gets an out-of-vocabulary Korean name right. That is a
real advantage — on a film recap the character's name is exactly what you cannot
afford to lose — but it is not the general accuracy advantage the headline
number implies. **A corpus CER cannot be read as a verdict on a specific
recording. Check what the errors are.**

### Punctuation, which CER does not measure

Scoring normalises punctuation away on both sides. That is correct for comparing
recognition accuracy and misleading for choosing a transcriber:

<!-- stt-evidence:punctuation:start -->
Held-out punctuation counts depend on deliberately untracked copyrighted transcript artifacts and remain diagnostic only.
<!-- stt-evidence:punctuation:end -->

omniASR returns seventeen thousand characters with no sentence boundary
anywhere. Seamless segments the same audio into 191 sentences. Neither shows up
in CER. Note also that omniASR's real-time factor doubles on a single long file
(1.79 on short clips, 3.60 here) — on CPU that was just under an hour for 17
minutes of audio against four minutes for Seamless.

Character counts differ by convention: the reference is 18,767 raw characters,
~16,980 after normalisation, which is the figure error rates are computed
against.

---

## Quantisation

The 300M GGUF at Q4_K scored **0.6647** on dev clips against 0.0959 for the same
model at float32. It is bimodal rather than uniformly bad: most clips are fine,
then a few collapse into decoder repetition loops emitting three to four times
the reference length. On the 17-minute file it produced `၁၁၀၁၁၁သက်၁၁` where every
full-precision model produced the sentence.

Use 4-bit for iteration, never for output.

---

## Voting

Different systems fail on different words. An oracle picking the better of the
top two per 200-character span would score **0.0571** where the best single model
scores 0.0857 — a third of the remaining error is recoverable without a better
model, just by choosing between hypotheses already in hand.

`stt vote` implements the practical version: align every run to a pivot, then
vote position by position, weighted by measured accuracy. Chosen on FLEURS and
verified on held-out audio, changing nothing between the two:

<!-- stt-evidence:voting:start -->
No current trusted full-cohort vote inputs are tracked; derived voting remains status-only until every constituent run shares complete waveform identity.
<!-- stt-evidence:voting:end -->

Three things that turned out to matter:

**The pivot must be your best model.** Voting can only correct characters the
pivot proposed, so the first run given anchors the result. Ties break toward the
pivot, which is why adding a system can never do worse than a wash.

**A pool of weak systems achieves nothing.** SeamlessM4T + Dolphin + MMS scores
0.1301 — exactly what SeamlessM4T scores alone. The 7B is load-bearing.

**Spacing convention dominates the alignment.** SeamlessM4T emits a space per
sub-word, omniASR emits none, so on raw text the aligner spends its budget on
whitespace instead of on the characters being voted: raw 0.0764, `tidy_spacing`
0.0718, full normalisation 0.0714. The default is `tidy_spacing`, which captures
nearly all of it while keeping the ၊ and ။ delimiters that full normalisation
discards.

Cost is modest. Adding SeamlessM4T, Dolphin and MMS to a 7B run took the total
real-time factor from 3.60 to 4.03 — 12 % more compute for 7.1 % less error on
FLEURS. Adding the 3B as well reaches 0.0900 but costs RTF 6.63, which is 84 %
more compute for a further 3 %.

---

## Confidence

Every backend returns `Segment`s carrying `start`, `end` and, where it can be
had, a confidence. `stt align` recovers all three for backends that produce none
— omniASR's fairseq2 pipeline returns a bare `List[str]` — by forced alignment
against MMS-1B, whose Burmese adapter is character-level, the right granularity
for a script written without word delimiters. Out-of-vocabulary characters on a
real transcript came to **0.15 %**, all uppercase Latin, which lowercasing
removes.

The signal is real, measured on the 7B's transcript of the held-out recording
against the human reference:

<!-- stt-evidence:confidence-quartiles:start -->
No redistributable trusted run currently carries aligned confidence segments; confidence quartiles cannot be published.
<!-- stt-evidence:confidence-quartiles:end -->

Pearson **r = −0.75**, and a **6.0×** error ratio between the quartiles. Error is
concentrated, so it can be bought cheaply:

<!-- stt-evidence:confidence-routing:start -->
No redistributable trusted base/strong pair currently carries aligned confidence segments; confidence routing cannot be published.
<!-- stt-evidence:confidence-routing:end -->

Note what this does *not* justify. Making the vote itself selective would save
almost nothing: the four-model vote costs only **1.21×** the pivot alone on
FLEURS and **1.11×** on the held-out recording, because the 7B pivot is 82–90 %
of the bill and the other three are nearly free. The expensive thing is the
pivot, so the concentration above is an argument for running the *pivot*
selectively.

**A caveat on the other confidence source.** CrispASR's `no_speech_prob` measures
how likely a span is to be *silence*, not how likely the transcript is to be
right, and the omniasr-llm backend does not compute it at all — it returns
`-1.0`. That is reported as unknown rather than converted into a confidence
of 2.0.

---

## Seam tax

Routing between a cheap model and an expensive one only pays if the join is
free, and it is not. Substituting the 7B's text into Seamless's least-confident
regions, holding the escalated share of audio at ~30 % and varying only how many
separate regions that share is split into:

<!-- stt-evidence:seam-tax:start -->
No redistributable trusted aligned base/strong pair exists for recomputing splice-seam endpoints.
<!-- stt-evidence:seam-tax:end -->

Monotonic, and the span is enormous — the same 30 % of audio escalated to the
same model scores 0.162 or 0.075 depending only on fragmentation. The two models'
segment boundaries do not coincide, so each seam drops or duplicates a few
characters. At ~16,980 reference characters that is roughly **11 characters per
seam**.

Two consequences. Switching has to happen in **coarse blocks**, not per segment;
and any future work that stitches model outputs together pays this same tax and
has to be measured with it included.

**A correctness check worth keeping.** An earlier version of the splice scored
0.1821 when escalating **100 %** of the audio, which must by definition reproduce
the 7B's 0.0857. Base segments did not tile the timeline, so 7B segments landing
in the gaps were silently dropped. The simulation now refuses to report
intermediate numbers unless the 0 % and 100 % ends reproduce the two source
transcripts exactly.

---

## Routing

If confidence predicts error, the expensive model only needs to run where the
cheap one is unsure. Held-out check on the 120-clip test set, routing whole
clips — Seamless everywhere, escalating the least-confident clips to the 7B.
Whole-clip routing means no splice seams, so the tax above does not confound it.

<!-- stt-evidence:routing:start -->
No redistributable trusted aligned base/strong pair exists for recomputing the routing curve and its endpoints.
<!-- stt-evidence:routing:end -->

Thirty percent of the compute budget captures **72.5 %** of the 7B's advantage
over Seamless. Routing the same 30 % at random would capture 30 % by definition,
so the ordering is doing real work.

The RTF column here is CPU-era, matching [Baseline](#baseline). With the 7B now
on Metal the absolute compute saving is much smaller; the accuracy result is
unaffected, since routing changes which model transcribes what, not how fast it
does so.

**What did not replicate.** On the 17-minute recording, block routing appeared to
beat *both* models (CER 0.0701 against the 7B's 0.0857). That does not hold here:
per-clip routing approaches the 7B from above and never passes it. On that
recording the two models are nearly tied (0.0887 vs 0.0857), so mixing them plays
to each one's strengths, whereas on FLEURS the 7B is 22 % better outright and
mixing can only interpolate. The 0.0701 figure was also the best of 24
configurations chosen on the same file it was measured on, which is not a result.
The **mechanism** generalises; that particular number does not.

---

## Device defaults

All four backends run on Metal. Whether they *should* differs per model:

<!-- stt-evidence:device-defaults:start -->
CrispASR does not expose the selected compute device; this measurement cannot carry a trusted execution identity.
<!-- stt-evidence:device-defaults:end -->

The verified MMS device/dtype candidate smoke below is descriptive: one clip,
one isolated session, and one repeat per condition. It validates each full model
load/inference path but is not a default-changing performance gate.

<!-- stt-evidence:mms-device-dtype-smoke:start -->
| Condition | RTF | Peak RSS MB | Sessions | Repeats |
| --- | ---: | ---: | ---: | ---: |
| cpu-float32 | 0.1172 | 5120 | 1 | 1 |
| cpu-float16 | 2.4088 | 5938 | 1 | 1 |
| cpu-bfloat16 | 2.6336 | 5939 | 1 | 1 |
| mps-float32 | 0.0295 | 519 | 1 | 1 |
| mps-float16 | 0.0307 | 5938 | 1 | 1 |
| mps-bfloat16 | 0.0544 | 5939 | 1 | 1 |
<!-- stt-evidence:mms-device-dtype-smoke:end -->

### omniASR on Metal

The verified utilization diagnostic uses eight duration-matched FLEURS test clips,
MPS/float16, batch 8, and explicit cache release between duration buckets. Profiling
is descriptive by observer policy, so this table does not replace baseline timing:

<!-- stt-evidence:omniasr-metal:start -->
| Condition | RTF | GPU mean % | GPU p50 % | GPU idle % | Peak RSS MB | Sessions | Repeats |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mps-float16-batch8-bucketed | 0.3425 | 81.5 | 76.0 | 0.0 | 26669 | 1 | 1 |
<!-- stt-evidence:omniasr-metal:end -->

The CPU trace was dominated by sequential autoregressive SGEMMs despite eight
configured Torch threads. Duration bucketing keeps padding bounded, while MPS cache
release prevents transient generation buffers from accumulating across the corpus.
`--device auto` still falls back to CPU if a Metal kernel fails mid-run.

### Dolphin stays on CPU

Dolphin would not load on Metal at all until recently, failing with `Cannot
convert a MPS Tensor to float64`. The cause was exactly two tensors out of 822:
`encoder.global_cmvn.mean` and `.std`, the 80-dimensional filterbank
normalisation statistics, stored as float64. Metal implements no float64
whatsoever, so those two made the whole model unloadable while all 819 real
parameters were already float32. Casting them to float32 fixes it — normalisation
statistics have no use for sixteen significant digits — and a second bug had to
go with it: `dolphin.transcribe` places its inputs using `model.device`, a plain
string that moving the module does not update.

With both fixed, Metal produces **identical text**, and is slower:

<!-- stt-evidence:dolphin-device:start -->
The historical Dolphin CPU/MPS comparison predates the experiment archive contract; MPS also records a float64-to-float32 numerical fallback, so no trusted causal table is published.
<!-- stt-evidence:dolphin-device:end -->

It is a small model decoding 20-second windows one at a time, so kernel launch
overhead costs more than the GPU wins back. `--device mps` is still worth having
because it frees the CPU almost entirely, which matters when something else needs
those cores — but CPU stays the default.

---

## Precision

### bfloat16 is not the safe default on Apple GPUs

Synthetic GEMM can nominate dtype candidates but cannot establish full-model
support or throughput. The verified MMS matrix above drives the complete model
through CPU/MPS and all three dtypes; bfloat16 is the slowest MPS condition for
that model. `stt.hardware.fastest_dtype` remains a candidate probe, while model
experiments decide defaults.

### On CPU, float32 is the fast path

The verified MMS matrix shows both CPU half formats far behind CPU float32. The
7B baseline therefore pins CPU/float32, while its optimized Metal path uses the
separately verified MPS/float16 utilization condition.

### The best dtype belongs to the model, not just the chip

Precision is a model property, not a chip-wide rule. Seamless remains pinned to
float32. The full-model candidate smoke produced identical transcripts across all
six CPU/MPS dtype conditions:

<!-- stt-evidence:seamless-device-dtype:start -->
| Condition | RTF | Peak RSS MB | Sessions | Repeats |
| --- | ---: | ---: | ---: | ---: |
| cpu-float32 | 0.4642 | 6988 | 1 | 1 |
| cpu-float16 | 2.0727 | 6561 | 1 | 1 |
| cpu-bfloat16 | 2.1651 | 6605 | 1 | 1 |
| mps-float32 | 0.1902 | 1053 | 1 | 1 |
| mps-float16 | 0.1890 | 6590 | 1 | 1 |
| mps-bfloat16 | 0.2028 | 6560 | 1 | 1 |
<!-- stt-evidence:seamless-device-dtype:end -->

Seamless autoregressive batches can be held open by their longest decoder tail.
The production MPS path therefore duration-buckets and caps the effective batch
at two even when callers request more. The profiled utilization condition is
descriptive under observer policy:

<!-- stt-evidence:seamless-utilization:start -->
| Condition | RTF | GPU mean % | GPU p50 % | GPU idle % | Peak RSS MB | Sessions | Repeats |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mps-float32-request4-effective2 | 0.0798 | 63.3 | 62.0 | 11.2 | 1162 | 1 | 1 |
<!-- stt-evidence:seamless-utilization:end -->

omniASR's MPS/float16 path is backed by its separate utilization experiment.

---

## Threads

macOS places threads across performance and efficiency cores itself, and nothing
here overrides it. That is a measured decision:

* the GGUF backend runs at **RTF 0.182 on 4, 8 and 12 threads alike** — the work
  is on the GPU and the CPU sits at 0.05 cores;
* a torch matmul scales **1.09× from 1 thread to 12**, because Accelerate does
  its own threading through the AMX unit;
* torch already derives its default from the system — it picks 8 on an M2 Max,
  matching `hw.perflevel0.physicalcpu`, without being told.

An earlier version of this repo set thread counts from a detected performance-core
count and hardcoded `n_threads=8` for the GGUF backend. Both are gone: neither
changed any measurement, and both could only be wrong on untested hardware.

The core split *is* still detected, for reporting — a benchmark number is not
interpretable without knowing the machine. Reading macOS's level *names*
(`hw.perflevel<N>.name`) rather than assuming level 0 is what keeps that correct
across chips:

<!-- stt-evidence:threads:start -->
CrispASR does not expose the selected compute device; this measurement cannot carry a trusted execution identity.
<!-- stt-evidence:threads:end -->

---

## Telemetry

This section's numeric tables are historical hypotheses, not current evidence.
The current implementation keeps a synchronized complete-corpus wall and exact
CPU time. Optional resource observations carry monotonic offsets, work-window,
source, scope, and phase; time-weighted summaries reject observations that finish
outside the work interval. Telemetry ticks describe shape and are never treated
as independent statistical replicates.

The retired one-file snapshot originally suggested three hypotheses to retest:

* apparent machine idleness may be an integration/batching constraint rather
  than a model limit;
* Dolphin's adapter/device path needs an upstream-parity test; and
* GGUF GPU allocation is not visible through Torch and a blank value must stay
  unknown rather than becoming zero.

The old `peak RSS` column used `ru_maxrss`, a process-lifetime high-water. Because
all subjects once ran sequentially in one process, it was cross-model
contaminated. Schema v2 runs one subject per fresh worker and publishes the
in-window psutil RSS peak separately from the lifetime high-water.

### What the measurement itself costs

GPU utilisation is not available from `torch.mps`, and `powermetrics` needs
root, so it is read from `ioreg`. The primitive microbenchmarks are retired;
paired whole-run observer calibration is the authoritative perturbation test.

The profiler currently samples CPU/RSS at 50 ms and whole-GPU `ioreg` activity at
250 ms, but neither cadence is ground truth. The forced post-work tail read has
been removed. Dense observer results and the old 22% idle headline are retired;
observer-off versus observer-on repeats must quantify perturbation before a
profiled timing can support a claim.

The sampler's `ioreg` child CPU and its own in-process thread CPU are measured
and subtracted after it joins, so an in-flight observer cannot be omitted from
the correction:

<!-- stt-evidence:telemetry-cost:start -->
| Condition | RTF | Observer CPU s | Repeat wall s | USS samples |
| --- | ---: | ---: | ---: | ---: |
| fast-off | 0.0230 | 0.0000 | 5.395 | 0 |
| fast-profile | 0.0229 | 0.4914 | 5.319 | 0 |
| fast-profile-uss | 0.0233 | 0.6037 | 5.426 | 94 |
| slow-off | 1.7512 | 0.0000 | 405.863 | 0 |
| slow-profile | 1.7965 | 0.6807 | 410.986 | 0 |
| slow-profile-uss | 1.8411 | 24.0619 | 426.992 | 8527 |
<!-- stt-evidence:telemetry-cost:end -->

The calibration keeps unprofiled timings authoritative. Profiled timing and USS
remain descriptive because their paired intervals did not prove overhead below
the declared materiality boundary.

Two caveats on the number itself. `Device Utilization %` is the **whole GPU** —
there is one accelerator entry, so anything else drawing on it inflates the
reading. GPU sampling only runs when the backend resolved to `mps` or `cuda`;
CPU-only runs still collect CPU/RSS series but do not claim GPU use.

### Batching A/B

The original harness passed one CLI-sized chunk at a time, preventing a backend
from seeing the full corpus and limiting native prefetch/bucketing. The verified
MMS matrix compares batch 1/2/4/8 over separate duration-matched and mixed-duration
eight-clip corpora, with five isolated sessions, three warmups, and three repeats:

<!-- stt-evidence:batching:start -->
| Condition | RTF | Peak RSS MB | Sessions | Repeats |
| --- | ---: | ---: | ---: | ---: |
| fixed-b1 | 0.0221 | 586 | 5 | 15 |
| fixed-b2 | 0.0196 | 610 | 5 | 15 |
| fixed-b4 | 0.0190 | 582 | 5 | 15 |
| fixed-b8 | 0.0190 | 577 | 5 | 15 |
| mixed-b1 | 0.0245 | 699 | 5 | 15 |
| mixed-b2 | 0.0261 | 619 | 5 | 15 |
| mixed-b4 | 0.0270 | 609 | 5 | 15 |
| mixed-b8 | 0.0309 | 607 | 5 | 15 |
<!-- stt-evidence:batching:end -->

Every condition retained stable transcript hashes. The recomputable sweep stops at
batch 8 for fixed-length input after two plateau steps, and at batch 4 for mixed
input because larger batches regress. Duration bucketing is therefore part of the
production batching path rather than an optional benchmark trick.

---

## Subtitles

omniASR emits no `၊` or `။` whatsoever, so cue boundaries fall back to a length
limit. Cutting blindly there splits grapheme clusters, and a cue that opens with
a bare `ာ` is not readable:

```
...လူတိုင်းသိချင်ကြပါတယ်ကောင်းတ
ာလုပ်ရင်နတ်ပြည်ရောက်မယ်...
```

Burmese stacks vowel signs, medials and tone marks onto a base consonant, and the
virama binds the consonant after it. Forced breaks retreat to the longest pause
in the tail of the cue that also keeps clusters intact. On the 17-minute
recording that took cues starting with a combining mark from 1 to 0, and moved
the breaks onto real phrase boundaries.

---

## Methodology

Two rules the measurements above imposed on the ones that follow them.

### A 12-clip screen is not a benchmark

The current 12-clip table uses FLEURS **dev**, while publication accuracy uses
all 120 tracked FLEURS **test** references. The small dev set is useful for smoke
coverage and fast comparisons, but its different split and population make it an
invalid substitute for the test result.

### Vendor CER for Burmese is not comparable across papers

Vendor metrics use different corpora and often do not publish an equivalent
Burmese normalization rule. Combining-mark order, Burmese punctuation, digits,
and whitespace can materially change CER, so vendor numbers are not comparable
to this harness or to one another. What is comparable here is the ordering within
one generated table over one declared reference identity and normalization block.
