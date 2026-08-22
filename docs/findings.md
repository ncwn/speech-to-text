# Historical measured findings (unverified)

This file preserves the measurements that motivated the current code. None is a
current publication claim until its block is regenerated from trusted artifacts
through `stt evidence`; the generated status below is authoritative.

All measurements are from an M2 Max (12 cores, 64 GB) unless stated otherwise.
Burmese is scored as **CER with whitespace stripped** — see
[burmese.md](burmese.md) for why WER does not apply.

<!-- stt-evidence:publication-status:start -->
> **Unverified legacy evidence.** Numeric publication is blocked because the declared artifacts are missing trusted audio identity, complete corpus coverage, or matching provenance. Regenerate the runs before publishing measured results.
<!-- stt-evidence:publication-status:end -->

**Read the measurement base before comparing two numbers.** Three different
bases appear below and they are not interchangeable:

<!-- stt-evidence:measurement-base:start -->
> **Unverified legacy evidence.** Numeric publication is blocked because the declared artifacts are missing trusted audio identity, complete corpus coverage, or matching provenance. Regenerate the runs before publishing measured results.
<!-- stt-evidence:measurement-base:end -->

---

## Baseline

Every backend over the same 120 FLEURS Burmese `test` clips. Lower is better on
both columns.

<!-- stt-evidence:baseline:start -->
> **Unverified legacy evidence.** Numeric publication is blocked because the declared artifacts are missing trusted audio identity, complete corpus coverage, or matching provenance. Regenerate the runs before publishing measured results.
<!-- stt-evidence:baseline:end -->

**The entire table is historical.** Its RTF values sum backend-provided
`elapsed_s` fields whose work boundaries differ by runtime, so they are not valid
cross-backend performance rankings. The JSONL also predates trusted audio
identity. Schema-v2 measurements instead use a synchronized whole-corpus wall in
an isolated worker. No schema-v2 performance baseline has been accepted.

Note how little separates omniASR 300M from Seamless (0.1298 vs 0.1301) despite
a 10× parameter difference and unrelated architectures. Above roughly 300M the
returns on this task are small.

Reproduce:

```bash
uv run stt fetch-fleurs --split test --limit 120 --dest data/fleurs-test
uv run stt compare data/fleurs-test/audio -r data/fleurs-test/references.tsv
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
> **Unverified legacy evidence.** Numeric publication is blocked because the declared artifacts are missing trusted audio identity, complete corpus coverage, or matching provenance. Regenerate the runs before publishing measured results.
<!-- stt-evidence:tails:end -->

Every model puts 2.5–8 % of clips over CER 0.3. The 7B has the best tail by a
clear margin — roughly half the failure rate of anything else — and for a long
recording that matters more than the average does.

---

## Held-out

FLEURS is public and may be in these models' training data, so the ranking was
re-checked against a hand-corrected transcript of a 16.8-minute Burmese film
recap that none of them has seen ([data/reference](../data/reference/README.md)).
The FLEURS ordering held:

<!-- stt-evidence:held-out:start -->
> **Unverified legacy evidence.** Numeric publication is blocked because the declared artifacts are missing trusted audio identity, complete corpus coverage, or matching provenance. Regenerate the runs before publishing measured results.
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
> **Unverified legacy evidence.** Numeric publication is blocked because the declared artifacts are missing trusted audio identity, complete corpus coverage, or matching provenance. Regenerate the runs before publishing measured results.
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
> **Unverified legacy evidence.** Numeric publication is blocked because the declared artifacts are missing trusted audio identity, complete corpus coverage, or matching provenance. Regenerate the runs before publishing measured results.
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
> **Unverified legacy evidence.** Numeric publication is blocked because the declared artifacts are missing trusted audio identity, complete corpus coverage, or matching provenance. Regenerate the runs before publishing measured results.
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
> **Unverified legacy evidence.** Numeric publication is blocked because the declared artifacts are missing trusted audio identity, complete corpus coverage, or matching provenance. Regenerate the runs before publishing measured results.
<!-- stt-evidence:confidence-quartiles:end -->

Pearson **r = −0.75**, and a **6.0×** error ratio between the quartiles. Error is
concentrated, so it can be bought cheaply:

<!-- stt-evidence:confidence-routing:start -->
> **Unverified legacy evidence.** Numeric publication is blocked because the declared artifacts are missing trusted audio identity, complete corpus coverage, or matching provenance. Regenerate the runs before publishing measured results.
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
> **Unverified legacy evidence.** Numeric publication is blocked because the declared artifacts are missing trusted audio identity, complete corpus coverage, or matching provenance. Regenerate the runs before publishing measured results.
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
> **Unverified legacy evidence.** Numeric publication is blocked because the declared artifacts are missing trusted audio identity, complete corpus coverage, or matching provenance. Regenerate the runs before publishing measured results.
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
> **Unverified legacy evidence.** Numeric publication is blocked because the declared artifacts are missing trusted audio identity, complete corpus coverage, or matching provenance. Regenerate the runs before publishing measured results.
<!-- stt-evidence:device-defaults:end -->

### omniASR on Metal

Five FLEURS clips, same model, same audio, corpus CER 0.0280 in every row
(historical device/dtype sweep):

<!-- stt-evidence:omniasr-metal:start -->
> **Unverified legacy evidence.** Numeric publication is blocked because the declared artifacts are missing trusted audio identity, complete corpus coverage, or matching provenance. Regenerate the runs before publishing measured results.
<!-- stt-evidence:omniasr-metal:end -->

Metal is 3.5× faster than CPU's best dtype and uses less memory, for
bit-identical text, so `--device auto` selects it with an automatic fall back to
CPU if a Metal kernel fails mid-run. Float16 is what
[`fastest_dtype`](#precision) picks on this machine. The 0.61 and 0.70 figures
are from that historical sweep; the current batch-1 baseline is the comparable
regression signal.

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
> **Unverified legacy evidence.** Numeric publication is blocked because the declared artifacts are missing trusted audio identity, complete corpus coverage, or matching provenance. Regenerate the runs before publishing measured results.
<!-- stt-evidence:dolphin-device:end -->

It is a small model decoding 20-second windows one at a time, so kernel launch
overhead costs more than the GPU wins back. `--device mps` is still worth having
because it frees the CPU almost entirely, which matters when something else needs
those cores — but CPU stays the default.

---

## Precision

### bfloat16 is not the safe default on Apple GPUs

Apple's GPUs are built around float16. bfloat16 is *accepted* everywhere but is
not equally *accelerated*. A 4096² matmul on an M2 Max:

<!-- stt-evidence:precision-gflops:start -->
> **Unverified legacy evidence.** Numeric publication is blocked because the declared artifacts are missing trusted audio identity, complete corpus coverage, or matching provenance. Regenerate the runs before publishing measured results.
<!-- stt-evidence:precision-gflops:end -->

bfloat16 is 2.1× slower than float16 and slower than float32 — it is being
emulated. Metal exposes the `bfloat` type broadly, but the simdgroup matrix
intrinsics that make it fast arrived with Metal 3.1 and the M3-era GPUs, and
whether *any* shipped Apple GPU has true hardware bfloat16 matrix units is
disputed. Since that answer changes per generation, `stt.hardware.fastest_dtype`
times both formats on the actual device and caches the result under
`.cache/stt/hardware.json`, keyed by chip and torch version.

### On CPU, float32 is the fast path

PyTorch has no native half-precision CPU kernels and emulates them. Measured on
the 7B: bfloat16 on CPU runs at RTF 8.85 against float32's 2.14 — a **4.1×**
penalty for identical text.

The 7B is 28 GB of float32 weights and needs roughly 34 GB resident once
activations and read buffers are counted, so `--dtype auto` falls back to
bfloat16 on CPU only when the machine cannot hold float32. That threshold is
computed from the card's checkpoint size against detected RAM, not from a fixed
constant, so the same model takes the fast path on a 64 GB machine and the
memory-safe one on a 16 GB machine.

### The best dtype belongs to the model, not just the chip

SeamlessM4T v2 on the same GPU goes the other way — float32 is both faster and
more accurate, so half precision buys only memory:

<!-- stt-evidence:seamless-dtype:start -->
> **Unverified legacy evidence.** Numeric publication is blocked because the declared artifacts are missing trusted audio identity, complete corpus coverage, or matching provenance. Regenerate the runs before publishing measured results.
<!-- stt-evidence:seamless-dtype:end -->

omniASR's LLM decoder is matmul-bound and gains from float16; Seamless is not and
does not. So the probe drives `omniasr-torch` only, and the `hf` backend keeps
float32.

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
> **Unverified legacy evidence.** Numeric publication is blocked because the declared artifacts are missing trusted audio identity, complete corpus coverage, or matching provenance. Regenerate the runs before publishing measured results.
<!-- stt-evidence:threads:end -->

---

## Telemetry

This section's numeric tables are historical hypotheses, not current evidence.
The current implementation keeps a synchronized complete-corpus wall and exact
CPU time. Optional resource observations carry monotonic offsets, work-window,
source, scope, and phase; time-weighted summaries reject observations that finish
outside the work interval. Telemetry ticks describe shape and are never treated
as independent statistical replicates.

The older one-file snapshot below is retained as historical context:

<!-- stt-evidence:telemetry-snapshot:start -->
> **Unverified legacy evidence.** Numeric publication is blocked because the declared artifacts are missing trusted audio identity, complete corpus coverage, or matching provenance. Regenerate the runs before publishing measured results.
<!-- stt-evidence:telemetry-snapshot:end -->

These observations originally suggested three hypotheses to retest:

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
root, so it is read from `ioreg`. On this host, `getrusage` costs about 0.40 us,
current RSS through `psutil` about 1.5 us, and one `ioreg` read about 17--19 ms
wall time / 15--18 ms child CPU. The primitive counters are sound; the sampling
cadence and aggregation are the important part.

The profiler currently samples CPU/RSS at 50 ms and whole-GPU `ioreg` activity at
250 ms, but neither cadence is ground truth. The forced post-work tail read has
been removed. Dense observer results and the old 22% idle headline are retired;
observer-off versus observer-on repeats must quantify perturbation before a
profiled timing can support a claim.

The sampler's `ioreg` child CPU and its own in-process thread CPU are measured
and subtracted after it joins, so an in-flight observer cannot be omitted from
the correction:

<!-- stt-evidence:telemetry-cost:start -->
> **Unverified legacy evidence.** Numeric publication is blocked because the declared artifacts are missing trusted audio identity, complete corpus coverage, or matching provenance. Regenerate the runs before publishing measured results.
<!-- stt-evidence:telemetry-cost:end -->

The historical cost table above is not yet regenerated under the timestamped
observer. The implementation records sampler child CPU and profiler thread CPU,
but the next iteration still needs paired observer-off/on worker sessions.

Two caveats on the number itself. `Device Utilization %` is the **whole GPU** —
there is one accelerator entry, so anything else drawing on it inflates the
reading. GPU sampling only runs when the backend resolved to `mps` or `cuda`;
CPU-only runs still collect CPU/RSS series but do not claim GPU use.

### Batching A/B

The original harness passed one CLI-sized chunk at a time, preventing a backend
from seeing the full corpus and limiting native prefetch/bucketing. Schema v2 now
passes the complete corpus once per repeat. The old Seamless A/B below remains a
hypothesis for the required full batching matrix:

<!-- stt-evidence:batching:start -->
> **Unverified legacy evidence.** Numeric publication is blocked because the declared artifacts are missing trusted audio identity, complete corpus coverage, or matching provenance. Regenerate the runs before publishing measured results.
<!-- stt-evidence:batching:end -->

No speedup is currently published from this table. Batching remains explicit,
and transcript hashes must match before throughput is compared. The next matrix
must use isolated repeated workers, exact model provenance, common-wall RTF, and
complete corpus coverage.

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

An earlier version of the baseline used 12 FLEURS **dev** clips:

<!-- stt-evidence:methodology-screen:start -->
> **Unverified legacy evidence.** Numeric publication is blocked because the declared artifacts are missing trusted audio identity, complete corpus coverage, or matching provenance. Regenerate the runs before publishing measured results.
<!-- stt-evidence:methodology-screen:end -->

Every model scored roughly twice as well on the small sample, and the Seamless vs
omniASR gap — 38 % on 12 clips — vanished entirely on 120. Small samples did not
merely add noise, they inverted the ranking. Twelve clips is a smoke test for
"does this backend work at all", nothing more.

### Vendor CER for Burmese is not comparable across papers

Meta reports **CER 4.4** for `omniASR_LLM_Unlimited_7B_v2` on Burmese. Measured
here on 120 FLEURS test clips, that same checkpoint scores **0.1017** — 2.3×
higher. The scaling curve behind it is smooth and well behaved:

<!-- stt-evidence:methodology-card:start -->
> **Unverified legacy evidence.** Numeric publication is blocked because the declared artifacts are missing trusted audio identity, complete corpus coverage, or matching provenance. Regenerate the runs before publishing measured results.
<!-- stt-evidence:methodology-card:end -->

so this is not a broken checkpoint or a bad decode — the model scales exactly as
it should, at about one CER point per 10× parameters. The gap is almost certainly
text normalisation: Burmese CER moves a long way depending on how you treat
combining-mark order, the ၊ and ။ delimiters, and whitespace, and Meta does not
publish its rule.

The same applies to the 54.9 % WER on the `whisper-large-v3-myanmar` card and
myMediWhisper's 23.44 %: three corpora, three normalisation rules, one of them
clinical dialogue. **Vendor numbers for Burmese are not comparable across papers
— including Meta's, and including this document.** What is comparable is the
ordering *within* one table, which is why every number here comes from the same
harness on the same clips.
