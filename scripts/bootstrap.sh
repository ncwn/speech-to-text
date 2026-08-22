#!/usr/bin/env bash
#
# Set up the environment on macOS (Apple Silicon).
# Verifies prerequisites, then installs the ASR runtimes.
#
# Usage: ./scripts/bootstrap.sh [--minimal | --gguf-only | --torch-only]

set -euo pipefail

cd "$(dirname "$0")/.."

# Every runtime by default. Installing a subset leaves `stt backends` showing
# the rest as missing, which is a worse first experience than a longer install.
EXTRAS=(--extra gguf --extra omniasr --extra hf --extra dolphin)
case "${1:-}" in
    --minimal)    EXTRAS=() ;;
    --gguf-only)  EXTRAS=(--extra gguf) ;;
    --torch-only) EXTRAS=(--extra omniasr) ;;
    "")           ;;
    *)            echo "Unknown option: $1" >&2; exit 2 ;;
esac

fail()  { printf '\033[31merror\033[0m  %s\n' "$*" >&2; exit 1; }
warn()  { printf '\033[33mwarn\033[0m   %s\n' "$*" >&2; }
ok()    { printf '\033[32mok\033[0m     %s\n' "$*"; }

# --- prerequisites --------------------------------------------------------

[[ "$(uname -s)" == "Darwin" ]] || fail "This script targets macOS."
[[ "$(uname -m)" == "arm64" ]]  || fail "Apple Silicon required — fairseq2n ships no Intel Mac wheels."

macos_major=$(sw_vers -productVersion | cut -d. -f1)
if (( macos_major < 14 )); then
    fail "macOS 14+ required; fairseq2n wheels are tagged macosx_14_0_arm64 (found $(sw_vers -productVersion))."
fi
ok "macOS $(sw_vers -productVersion) on $(uname -m)"

command -v uv >/dev/null || fail "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
ok "uv $(uv --version | awk '{print $2}')"

command -v ffmpeg >/dev/null || fail "ffmpeg not found. Install: brew install ffmpeg"
ok "ffmpeg present"

# fairseq2n dlopens the system libsndfile by name. The Python soundfile wheel
# bundles its own copy, which does NOT satisfy this.
if ! (brew list --formula 2>/dev/null | grep -qx libsndfile); then
    fail "libsndfile not found. Install: brew install libsndfile"
fi
ok "libsndfile present"

# kenlm has no wheels and compiles a C++ extension during install.
xcode-select -p >/dev/null 2>&1 || fail "Xcode Command Line Tools missing. Install: xcode-select --install"
ok "C++ toolchain present"

# Weights land in the repo-local cache by default, so check that volume rather
# than $HOME. See src/stt/paths.py.
CACHE_DIR="${STT_CACHE_DIR:-$PWD/.cache}"
mkdir -p "$CACHE_DIR"
avail=$(df -g "$CACHE_DIR" | awk 'NR==2 {print $4}')
if (( avail < 40 )); then
    warn "${avail} GB free on the volume holding ${CACHE_DIR}. The 7B checkpoint alone is 31.2 GB."
else
    ok "${avail} GB free for model caches at ${CACHE_DIR}"
fi

if [[ ! -f .env && -f .env.example ]]; then
    cp .env.example .env
    ok "created .env from .env.example"
fi

# --- install --------------------------------------------------------------

echo
echo "Installing: uv sync ${EXTRAS[*]:-(core only)}"
uv python install 3.12
uv sync ${EXTRAS[@]+"${EXTRAS[@]}"}

# Tracked hooks, so a fresh clone gets the guards without remembering to.
if [[ -d .githooks ]] && git rev-parse --git-dir >/dev/null 2>&1; then
    git config core.hooksPath .githooks
    ok "pre-commit hook enabled (make check)"
fi

echo
uv run stt doctor

cat <<'EOF'

Next steps:

  uv run stt fetch-fleurs --limit 20        # Burmese audio + references
  uv run stt transcribe data/fleurs/audio -b omniasr-gguf
  uv run stt eval outputs/omniasr-gguf-*.jsonl -r data/fleurs/references.tsv

Weights download into ./.cache on first use. `uv run stt doctor` shows where
they are; set STT_CACHE_DIR to put them somewhere else.

  make check          # ruff + tests, ~6 s (also runs on every commit)
  make bench          # regression gate against baselines/bench.json, minutes

EOF
