# Setup on macOS (Apple Silicon)

These instructions were checked against `scripts/bootstrap.sh`, `pyproject.toml`,
the locked environment, and the installed runtime metadata on 2026-08-24.

## Choose a runtime set

The default and `--torch-only` bootstrap paths install `omnilingual-asr 0.2.0`
and therefore require:

- Apple Silicon and macOS 14 or newer; the locked `fairseq2n 0.6` wheel is
  tagged `macosx_14_0_arm64` and has no Intel macOS wheel.
- Python 3.12. `uv` installs it; this project requires `>=3.12,<3.13`.
- Xcode Command Line Tools, because `kenlm` builds a C++ extension.
- Homebrew `libsndfile`, which fairseq2 loads as a system library.
- `uv` and ffmpeg.

The `--gguf-only` path does not install fairseq2, so the script does not require
macOS 14, Xcode Command Line Tools, Homebrew, or `libsndfile`. It still requires
Apple Silicon, `uv`, and ffmpeg. The locked CrispASR wheel is tagged for arm64
macOS 11 or newer.

## Default setup, step by step

From the repository root:

1. Confirm Apple Silicon and macOS 14 or newer.

   ```bash
   uname -m
   sw_vers -productVersion
   ```

2. Install Homebrew if `brew --version` fails, using the instructions at
   <https://brew.sh/>.

3. Install the system dependencies.

   ```bash
   xcode-select -p || xcode-select --install
   # If the installer opened, finish it before continuing.
   xcode-select -p
   brew install ffmpeg libsndfile
   ```

4. Install `uv` if `uv --version` fails, using the instructions at
   <https://docs.astral.sh/uv/getting-started/installation/>.

5. Install the default runtime set.

   ```bash
   ./scripts/bootstrap.sh
   ```

6. Verify the environment without downloading weights.

   ```bash
   uv run python -VV
   uv run stt backends
   uv run stt models
   ```

The bootstrap creates or updates `.venv` with
`uv sync --frozen --extra gguf --extra omniasr`. It installs the project and
runtime dependencies only. It does not download model weights and does not
remove, rewrite, or validate model caches.

For the smaller GGUF-only setup:

```bash
brew install ffmpeg
./scripts/bootstrap.sh --gguf-only
uv run stt backends
```

## Selective dependency sets

`uv sync` makes the environment match the extras named in that invocation.
Extras are not an additive install history: always name the complete set you
want to keep. Each command below also installs the project's default `dev`
dependency group.

```bash
uv sync --frozen                                      # no backend extra
uv sync --frozen --extra gguf                         # core + GGUF
uv sync --frozen --extra omniasr                      # core + fairseq2 omniASR
uv sync --frozen --extra hf                           # core + Transformers
uv sync --frozen --extra dolphin                      # core + Dolphin
uv sync --frozen --extra omniasr --extra hf           # fairseq2 + alignment
uv sync --frozen --extra gguf --extra omniasr --extra hf --extra dolphin
```

After changing the set, run `uv run stt backends` again.

## Download one model

Model listing is read-only. Download only the selected backend/model pair:

```bash
uv run stt models
uv run stt models --backend omniasr-gguf
uv run stt models --backend omniasr-gguf --download llm-unlimited-300m-v2
```

The download command uses the same upstream cache as transcription. A later
`stt transcribe -b omniasr-gguf -m llm-unlimited-300m-v2 ...` reuses it.
The GGUF, Dolphin, and Transformers adapters use immutable upstream revisions.
Running `stt compare` without `--only` considers cached defaults only and keeps
the Transformers load offline; name `--only hf` to permit its normal first-use
download behavior.

## Dependency constraints

[`omnilingual-asr==0.2.0`](https://pypi.org/project/omnilingual-asr/) declares
`fairseq2[arrow]>=0.5.2,<=0.6.0`; the lock selects `fairseq2==0.6` and
`fairseq2n==0.6`. The native package requires `torch==2.8.0`, so this repository
pins both `torch` and `torchaudio` to 2.8.0. Do not upgrade either package
independently: `fairseq2n` links against the libtorch C++ ABI.

The latest [fairseq2 release](https://github.com/facebookresearch/fairseq2/releases/tag/v0.8.1)
checked for this audit was 0.8.1, whose
[`fairseq2n` metadata](https://pypi.org/project/fairseq2n/) requires
`torch==2.9.1`. That newer pair is not compatible with `omnilingual-asr
0.2.0`'s fairseq2 upper bound, so it is not used here.

## Model caches

Bootstrap and `uv sync` leave these caches in place:

| Runtime | Default used here | Supported location controls |
|---|---|---|
| fairseq2 | `~/.cache/fairseq2/assets/` | `FAIRSEQ2_CACHE_DIR`; otherwise `XDG_CACHE_HOME/fairseq2/assets` |
| CrispASR | `~/.cache/crispasr/` | `CRISPASR_CACHE_DIR` |
| Hugging Face Hub | `~/.cache/huggingface/hub/` | `HF_HUB_CACHE`; otherwise `HF_HOME/hub` or `XDG_CACHE_HOME/huggingface/hub` |
| Dolphin | `~/.cache/dolphin/<size>/` | No repository override; each size has its own directory |
| dtype probe | `~/.cache/stt/hardware.json` | No repository override |
| ggml Metal kernels | `~/Library/Caches/ggml-metal/` | Managed by ggml |

Set cache variables before the first command in a shell; changing one points the
runtime at a different cache and does not migrate existing files.

Never delete an entire user cache tree to recover one model. That can destroy
unrelated application data and every working checkpoint. Preserve the existing
cache and test a fresh isolated location instead:

```bash
FAIRSEQ2_CACHE_DIR="$PWD/.cache/recovery/fairseq2" \
  uv run stt models --backend omniasr-torch \
  --download omniASR_LLM_Unlimited_300M_v2

CRISPASR_CACHE_DIR="$PWD/.cache/recovery/crispasr" \
  uv run stt models --backend omniasr-gguf \
  --download llm-unlimited-300m-v2

HF_HUB_CACHE="$PWD/.cache/recovery/huggingface" \
  uv run stt models --backend hf --download whisper-my-small
```

The repository ignores `.cache/`. After confirming a replacement, quarantine
only the suspect model directory by renaming it; do not mix files from two
downloads. fairseq2 uses content-hashed paths, so use a fresh
`FAIRSEQ2_CACHE_DIR` rather than editing its internal layout. Dolphin has no
cache override in this adapter: stop active runs and rename only the affected
`~/.cache/dolphin/<size>/` directory before downloading that size again. The
adapter leaves a mismatched cache untouched and refuses to load it.

The GGUF adapter also leaves a mismatched or partial model untouched. Quarantine
only the affected `.gguf` and its same-name `.gguf.src` sidecar before retrying
the selective download; the model's SHA-256 digest, not the advisory sidecar,
determines whether the cache is usable.

Inspect space without modifying anything:

```bash
du -sh ~/.cache/fairseq2/assets ~/.cache/crispasr \
  ~/.cache/huggingface/hub ~/.cache/dolphin 2>/dev/null
```

## Device behavior

The repository's `omniasr-torch` adapter asks PyTorch for MPS when available.
If decode raises on MPS, the adapter reloads the model on CPU and retries the
file. This is repository behavior, not a claim that fairseq2 officially
supports Metal or that every fairseq2 operation has an MPS kernel. Use
`--device cpu` to avoid the attempt or `--device mps` to request it explicitly.

The Transformers adapter selects MPS when available but does not perform the
same automatic CPU retry. CrispASR selects its native ggml device. Dolphin
defaults to CPU; `--device mps` requests the adapter's explicit MPS path.

## Troubleshooting

**`fairseq2 requires libsndfile`.** Install Homebrew `libsndfile`; the copy
bundled inside the Python `soundfile` wheel does not satisfy fairseq2's system
library lookup.

**`kenlm` fails to build.** Confirm `xcode-select -p` succeeds, then rerun the
same frozen sync. No separate CMake install is required by this project.

**Dependency resolution backtracks through fairseq2 releases.** Confirm
`uv run python -VV` reports Python 3.12, `uname -m` reports `arm64`, and macOS is
14 or newer for the fairseq2 path.

**Native GGUF logs are hidden.** Pass `-v` to transcription when diagnosing a
Metal load; the adapter suppresses native ggml output otherwise.

**A first run appears idle.** Download the selected model explicitly with
`stt models --backend BACKEND --download MODEL`, keep stderr visible, and watch
only that runtime's cache with `du -sh`. Do not insert partial files manually.

**Dolphin reports a tensor size mismatch.** Each size must keep its own cache
directory. Quarantine the affected size directory and download that size again;
do not reuse a flat directory containing another size's configuration.
