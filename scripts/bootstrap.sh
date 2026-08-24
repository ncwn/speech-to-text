#!/usr/bin/env bash
#
# Set up runtime dependencies on macOS (Apple Silicon).
# Installs both omniASR runtimes by default; does not download model weights.
#
# Usage: ./scripts/bootstrap.sh [--gguf-only | --torch-only]

set -euo pipefail

cd "$(dirname "$0")/.."

EXTRAS=(--extra gguf --extra omniasr)
INSTALL_OMNIASR=1
case "${1:-}" in
    --gguf-only)  EXTRAS=(--extra gguf); INSTALL_OMNIASR=0 ;;
    --torch-only) EXTRAS=(--extra omniasr) ;;
    "")           ;;
    *)            echo "Unknown option: $1" >&2; exit 2 ;;
esac

fail()  { printf '\033[31merror\033[0m  %s\n' "$*" >&2; exit 1; }
ok()    { printf '\033[32mok\033[0m     %s\n' "$*"; }

# --- prerequisites --------------------------------------------------------

[[ "$(uname -s)" == "Darwin" ]] || fail "This script targets macOS."
[[ "$(uname -m)" == "arm64" ]]  || fail "Apple Silicon required; the macOS runtimes ship no Intel wheels."

macos_major=$(sw_vers -productVersion | cut -d. -f1)
if (( INSTALL_OMNIASR && macos_major < 14 )); then
    fail "macOS 14+ required; fairseq2n wheels are tagged macosx_14_0_arm64 (found $(sw_vers -productVersion))."
fi
ok "macOS $(sw_vers -productVersion) on $(uname -m)"

command -v uv >/dev/null || fail "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
ok "uv $(uv --version | awk '{print $2}')"

if ! command -v ffmpeg >/dev/null; then
    command -v brew >/dev/null || fail "ffmpeg and Homebrew not found. Install Homebrew: https://brew.sh"
    fail "ffmpeg not found. Install: brew install ffmpeg"
fi
ok "ffmpeg present"

if (( INSTALL_OMNIASR )); then
    command -v brew >/dev/null || fail "Homebrew not found. Install: https://brew.sh"

    # fairseq2n dlopens the system libsndfile by name. The Python soundfile wheel
    # bundles its own copy, which does NOT satisfy this.
    if ! brew list --versions libsndfile >/dev/null 2>&1; then
        fail "libsndfile not found. Install: brew install libsndfile"
    fi
    ok "libsndfile present"

    # kenlm has no wheels and compiles a C++ extension during install.
    xcode-select -p >/dev/null 2>&1 || fail "Xcode Command Line Tools missing. Install: xcode-select --install"
    ok "C++ toolchain present"
fi

# --- install --------------------------------------------------------------

echo
echo "Installing: uv sync --frozen ${EXTRAS[*]}"
uv python install 3.12
uv sync --frozen "${EXTRAS[@]}"

echo
uv run stt backends

cat <<'EOF'

Runtime dependencies installed. No model weights were downloaded or model caches changed.

Next steps:

  uv run stt models                          # list models, sizes, and exact IDs
  uv run stt fetch-fleurs --limit 20        # Burmese audio + references
  uv run stt transcribe data/fleurs/audio -b BACKEND -m MODEL_ID
  uv run stt eval outputs/BACKEND.jsonl -r data/fleurs/references.tsv

EOF
