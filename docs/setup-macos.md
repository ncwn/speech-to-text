# Setup on macOS (Apple Silicon)

## Requirements

- macOS 14 or newer. `fairseq2n` wheels are tagged `macosx_14_0_arm64`; on
  anything older pip falls back to building from source and fails.
- Apple Silicon. There are no `fairseq2n` wheels for Intel Macs.
- Python 3.12. `omnilingual-asr` caps at 3.12 and `fairseq2n` publishes
  cp310–cp312 only. `uv` installs the right interpreter for you.
- [uv](https://docs.astral.sh/uv/), ffmpeg, libsndfile.

## Install

```bash
brew install ffmpeg libsndfile
./scripts/bootstrap.sh
```

`bootstrap.sh` verifies the prerequisites, creates a Python 3.12 virtualenv, and
installs both runtimes.

To install selectively:

```bash
uv sync                                    # core CLI + evaluation only
uv sync --extra gguf                       # + Metal runtime  (small)
uv sync --extra omniasr                    # + PyTorch stack  (~3 GB of wheels)
uv sync --extra gguf --extra omniasr       # both
```

Verify with `uv run stt backends`.

## Things that will bite you

**libsndfile must come from Homebrew.** The Python `soundfile` wheel bundles its
own copy, but `fairseq2n` loads the system library by name via `dlopen`. Without
it you get `OSError: fairseq2 requires libsndfile` on import — even though
`import soundfile` works fine.

**Do not upgrade torch on its own.** `fairseq2n` links against libtorch's C++
ABI, which has no stability guarantee between releases, so `fairseq2n 0.6`
requires exactly `torch==2.8.0`. Upgrading produces missing-symbol errors at
import. The pin is in `pyproject.toml`; leave it.

**kenlm builds from source.** It is a dependency of `omnilingual-asr` and ships
no wheels, so the install compiles a C++ extension. Xcode Command Line Tools are
enough (`xcode-select --install`); cmake is pulled in automatically as a build
requirement.

**Dependency resolution failures** that backtrack through old `fairseq2`
versions almost always mean no matching `fairseq2n` wheel was found for your
Python version or macOS version. Check `python -VV` reports 3.12 and that
`pip debug --verbose` lists a `macosx_14_0_arm64` platform tag — an x86_64
Python running under Rosetta is the other usual culprit.

## Disk

Model weights are cached outside the repo:

```
~/.cache/fairseq2/assets/     PyTorch checkpoints — up to 31 GB each
~/.cache/crispasr/            GGUF checkpoints — around 1 GB each
```

Downloading every PyTorch card would need well over 60 GB. Check free space
before pulling the 7B, and delete cards you are done with rather than letting
them accumulate.

## GPU status

`torch.backends.mps.is_available()` returns True, but that only says Metal
exists — fairseq2 has no validated MPS path, so `omniasr-torch` defaults to CPU
and will not silently pick MPS. `--device mps` is available to experiment with;
expect unimplemented-operator errors or wrong output rather than a clean
speedup.

For actual GPU acceleration use `omniasr-gguf`, which reaches roughly RTF 0.2
on an M2 Max with the 300M model — about five times faster than real time.

## Troubleshooting

**`ggml_metal_*` log spam.** ggml logs kernel compilation to fd 1/2 from native
code. It is suppressed by default; pass `-v` to see it, which is what you want
when a Metal load fails.

**First GGUF run is slow.** Metal kernels are compiled and cached to
`~/Library/Caches/ggml-metal/`. Subsequent runs skip it.

**`omniasr-torch` first run takes a long time.** It is downloading — 6.5 GB for
the 300M card, 31.2 GB for the 7B. fairseq2 prints a progress bar to stderr; if
you have redirected it, watch the cache directory instead:

```bash
du -sh ~/.cache/fairseq2/assets/
```

**fairseq2 does not resume interrupted downloads.** It writes to
`*.download.tmp` and only renames on completion, so a failed download never
leaves a corrupt checkpoint — but every retry restarts from byte zero. On a slow
or flaky link the 3B (17.5 GB) and 7B (31.2 GB) cards may never finish:

```
AssetDownloadError: The server sent 3,712,927,948 bytes which is less than the
expected size of 17,522,712,611 bytes.
```

`dl.fbaipublicfiles.com` *does* honour HTTP Range requests, so fetch the large
cards with `curl -C -` straight into the cache instead. fairseq2 keys the cache
directory on the first 24 hex characters of `sha1(uri)`:

```bash
FILE=omniASR-LLM-Unlimited-7B-v2.pt
URL=https://dl.fbaipublicfiles.com/mms/$FILE
DIR=$(python -c "import hashlib,sys;print(hashlib.sha1(sys.argv[1].encode()).hexdigest()[:24])" "$URL")

mkdir -p ~/.cache/fairseq2/assets/$DIR
curl -L --fail -C - --retry 100 --retry-all-errors --retry-delay 5 \
     --speed-limit 10240 --speed-time 60 \
     -o ~/.cache/fairseq2/assets/$DIR/$FILE "$URL"
```

Re-run the identical command after any interruption and it picks up where it
stopped. Verify the size is byte-exact before using it — a truncated checkpoint
loads and emits silent garbage rather than raising.

## Dolphin weights

`dolphin.load_model(size, directory, device)` writes `config.yaml` and
`train.yaml` beside the checkpoint and skips files that already exist. Pointing
two sizes at one directory leaves the first model's config next to the second
model's weights, and the load dies with `size mismatch for
decoder.decoders.5.norm3.bias`. This backend gives each size its own directory
under `~/.cache/dolphin/<size>/`; delete any older flat cache.

## Apple Silicon: which precision, which cores

Two hardware-dependent choices matter, and both are measured rather than
assumed — the repo probes the machine once and caches the answer under
`~/.cache/stt/hardware.json`.

### bfloat16 is not the safe default on Apple GPUs

Apple's GPUs are built around float16. bfloat16 is *accepted* everywhere but is
not equally *accelerated*. A 4096² matmul on an M2 Max:

| dtype | GFLOP/s |
|---|---:|
| float16 | **12,306** |
| float32 | 11,253 |
| bfloat16 | 5,797 |

bfloat16 is 2.1× slower than float16 and slower than float32 — it is being
emulated. Metal exposes the `bfloat` type broadly, but the simdgroup matrix
intrinsics that make it fast arrived with Metal 3.1 and the M3-era GPUs, and
whether *any* shipped Apple GPU has true hardware bfloat16 matrix units is
disputed. Since that answer changes per generation, `stt.hardware.fastest_dtype`
times both formats on the actual device instead of consulting a table.

End to end on the omniASR 7B, five FLEURS clips, all producing identical text
and identical corpus CER (0.0280):

| config | RTF | GPU |
|---|---:|---:|
| Metal float16 | **0.61** | 17.1 GB |
| Metal bfloat16 | 0.70 | 17.1 GB |
| CPU float32 | 2.14 | — |
| CPU bfloat16 | 8.85 | — |

### …but the best dtype belongs to the model, not just the chip

SeamlessM4T v2 on the same GPU goes the other way — float32 is both faster and
more accurate, so half precision buys only memory:

| dtype | RTF | CER |
|---|---:|---:|
| float32 | **0.16** | **0.0420** |
| float16 | 0.26 | 0.0455 |
| bfloat16 | 0.26 | 0.0420 |

omniASR's LLM decoder is matmul-bound and gains from float16; Seamless is not
and does not. So the probe drives `omniasr-torch` only, and the `hf` backend
keeps float32.

### Performance cores versus efficiency cores

Efficiency cores make a parallel step finish later, because it finishes with its
slowest thread. The split is not a constant, so the repo reads macOS's own
naming (`hw.perflevel<N>.name`) and counts everything that is not called
*Efficiency*:

| chip | levels | threads used |
|---|---|---:|
| M2 Max | Performance 8 + Efficiency 4 | 8 of 12 |
| M5 Pro | Super 6 + Performance 12 | 18 of 18 |

Reading the names rather than taking the fastest level is what makes this
correct on an M5 Pro, which has **no efficiency cores at all** — taking only
level 0 there would idle two thirds of the CPU. This sets torch's thread pool
and the GGUF backend's `n_threads`, which was previously hardcoded to 8: right
for an M2 Max by coincidence, wrong for a base M4 (4 performance cores) and for
an M5 Pro.

Note that thread count is a small lever on this stack today. Apple's Accelerate
backend does its own threading through the AMX unit, so a torch matmul scales
only 1.09× from 1 thread to 12 — and with Metal working, the heavy models are
not on the CPU at all.
