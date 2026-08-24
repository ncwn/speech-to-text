# Setup on macOS (Apple Silicon)

## Requirements

- macOS 14 or newer. `fairseq2n` wheels are tagged `macosx_14_0_arm64`; on
  anything older the bootstrap check stops before an unsupported build.
- Apple Silicon. There are no `fairseq2n` wheels for Intel Macs.
- Python 3.12. `omnilingual-asr` caps at 3.12 and `fairseq2n` publishes
  cp310–cp312 only. `uv` installs the right interpreter for you.
- [uv](https://docs.astral.sh/uv/), ffmpeg, libsndfile, and Xcode Command Line
  Tools (needed to build `kenlm`).

## Install

```bash
brew install ffmpeg libsndfile
./scripts/bootstrap.sh
```

`bootstrap.sh` verifies the prerequisites, creates a Python 3.12 virtualenv, and
installs the GGUF and omniASR runtimes. Use `--gguf-only` or `--torch-only` to
install one of those two runtimes.

To install selectively:

```bash
uv sync                                    # core CLI + evaluation only
uv sync --extra gguf                       # + Metal runtime  (small)
uv sync --extra omniasr                    # + PyTorch stack  (~3 GB of wheels)
uv sync --extra hf                         # + transformers on Metal
uv sync --extra dolphin                    # + Dolphin
uv sync --extra gguf --extra omniasr --extra hf --extra dolphin  # all
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
versions usually mean no matching `fairseq2n` wheel was found. Check
`uv run python -VV`, `uname -m`, and `sw_vers -productVersion`; an x86_64
Python running under Rosetta is the other usual culprit. The bootstrap script
performs the architecture and macOS checks before syncing.

## Disk

Model weights are cached outside the repo:

```
~/.cache/fairseq2/assets/     PyTorch checkpoints — up to 31 GB each
~/.cache/crispasr/            GGUF checkpoints — around 1 GB each
~/.cache/dolphin/<size>/      Dolphin checkpoints
~/.cache/huggingface/hub/     Hugging Face model snapshots
```

Downloading every PyTorch card would need well over 60 GB. Check free space
before pulling the 7B, and delete cards you are done with rather than letting
them accumulate.

## GPU status

On Apple Silicon, `omniasr-torch` and `hf` use MPS automatically when it is
available. The omniASR adapter retries on CPU if a fairseq2 Metal kernel fails;
use `--device cpu` to force CPU or `--device mps` to request Metal.
`omniasr-gguf` lets CrispASR select Metal, while `dolphin` defaults to CPU.

For the runtime defaults and their historical rationale, see
[Models and runtimes](models.md) and [Historical findings](findings.md).

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

**The pinned fairseq2 downloader does not resume interrupted downloads.** A
retry restarts the current checkpoint. Prefer a stable connection for the 3B
and 7B cards, and do not place partial files into the asset cache manually;
fairseq2's hashed cache layout is an internal detail.

## Dolphin weights

`dolphin.load_model(size, directory, device)` writes `config.yaml` and
`train.yaml` beside the checkpoint and skips files that already exist. Pointing
two sizes at one directory leaves the first model's config next to the second
model's weights, and the load dies with `size mismatch for
decoder.decoders.5.norm3.bias`. This backend gives each size its own directory
under `~/.cache/dolphin/<size>/`; delete any older flat cache.

## Runtime defaults and hardware reporting

`stt hardware` reads the chip, core layout, and RAM from the machine. On MPS,
the float16-versus-bfloat16 timing used by `omniasr-torch` is cached at
`~/.cache/stt/hardware.json`; core layout, RAM, and float32 headroom are read or
computed at runtime. Use `stt hardware --refresh` to rerun the dtype probe.

Thread counts are left to macOS and to each runtime. The `--threads` option is
available for GGUF experiments, but the default is the runtime's own choice.
