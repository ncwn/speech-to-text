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
installs the runtimes.

To install selectively:

```bash
uv sync                                    # core CLI + evaluation only
uv sync --extra gguf                       # + Metal/GGUF runtime  (small)
uv sync --extra omniasr                    # + PyTorch/fairseq2    (~3 GB of wheels)
uv sync --extra hf                         # + transformers        (Seamless, MMS, Whisper)
uv sync --extra dolphin                    # + Dolphin             (pulls funasr + modelscope)
uv sync --extra gguf --extra omniasr       # any combination
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

Weights are cached in `.cache/` inside the checkout, so a clone stays
self-contained:

```
.cache/fairseq2/     omniASR PyTorch checkpoints — up to 31 GB each
.cache/crispasr/     omniASR GGUF checkpoints — around 1 GB each
.cache/dolphin/<size>/   Dolphin checkpoints — 570 MB / 1.5 GB
.cache/huggingface/  transformers models (Seamless, MMS, Whisper)
.cache/stt/          the hardware probe — 4 KB
```

`.cache/` is gitignored. Set `STT_CACHE_DIR` to put the weights somewhere else
— `STT_CACHE_DIR=~/.cache` restores the conventional per-user location. Run
`uv run stt doctor` to see where everything currently is.

Downloading every PyTorch card would need well over 60 GB. Check free space
before pulling the 7B, and delete cards you are done with rather than letting
them accumulate.

**One exception: fairseq2.** It reads `~/.cache/fairseq2` directly and exposes
no environment variable, so its checkpoints are only repo-local if that path is
a symlink into `.cache/fairseq2`. `stt doctor` reports which arrangement is in
place and `stt doctor --migrate` offers to relink it.

**The Hugging Face cache is normally shared** with every other Python project on
the machine. `--migrate` therefore moves only the model repos this project
names, and leaves anything else where it is.

## GPU status

All four backends run on Metal. Three default to it; Dolphin defaults to CPU,
where it is measurably faster. Each default was measured rather than assumed —
see [Device defaults](findings.md#device-defaults).

`--device` overrides it per run (`auto`, `cpu`, `mps`, `cuda`), and
`omniasr-torch` falls back to CPU automatically if a Metal kernel fails
mid-run.

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
du -sh "$(uv run python -c 'from stt.paths import cache_root; print(cache_root())')/fairseq2"
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
under `.cache/dolphin/<size>/`; delete any older flat cache.

## Precision and cores

Two hardware-dependent choices matter — which 16-bit format this GPU actually
accelerates, and how many threads to use — and both are measured rather than
assumed. The repo probes the machine once and caches the answer under
`.cache/stt/hardware.json`.

`stt hardware` prints what it found. The measurements and what they changed are
in [Precision](findings.md#precision) and [Threads](findings.md#threads).

Nothing about a chip is written down in this repo: the float16-versus-bfloat16
choice is timed on the device, whether a card can afford float32 on CPU is
computed from its checkpoint size against detected RAM, and core counts and
names come from `sysctl`. On an untested chip it should need no code change to
do the right thing.
