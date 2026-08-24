#!/usr/bin/env bash

# Bounded product-selection gate for the Torch Omnilingual CTC cards.
# CTC cards are short-audio models, so this intentionally uses FLEURS rather
# than the long Eternity file. Outputs are ignored working artifacts.

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
audio_dir="${root}/data/fleurs-test/audio"
reference="${root}/data/fleurs-test/references.tsv"
output_dir="${root}/outputs/ctc-product-gate"
limit="${CTC_GATE_LIMIT:-0}"
device="${CTC_GATE_DEVICE:-mps}"
dtype="${CTC_GATE_DTYPE:-float16}"

if [[ ! -d "${audio_dir}" || ! -f "${reference}" ]]; then
    printf 'CTC gate needs data/fleurs-test/audio and data/fleurs-test/references.tsv\n' >&2
    exit 2
fi

if ! git -C "${root}" diff --quiet || ! git -C "${root}" diff --cached --quiet; then
    printf 'CTC gate requires a clean worktree for trusted provenance\n' >&2
    exit 2
fi

mkdir -p "${output_dir}/logs"

reference_for_eval="${reference}"
if [[ "${limit}" -gt 0 ]]; then
    ids_file="${output_dir}/selected-audio-ids.txt"
    reference_for_eval="${output_dir}/references-${limit}.tsv"
    find "${audio_dir}" -type f -name '*.wav' -print \
        | sort \
        | head -n "${limit}" \
        | sed 's#.*/##; s#\.wav$##' >"${ids_file}"
    awk 'NR == FNR { selected[$1] = 1; next } FNR == 1 || selected[$1]' \
        "${ids_file}" "${reference}" >"${reference_for_eval}"
fi

models=(
    omniASR_CTC_300M_v2
    omniASR_CTC_1B_v2
)
batches=(1 8)

for model in "${models[@]}"; do
    for batch_size in "${batches[@]}"; do
        stem="omniasr-torch--${model}-mps-${dtype}-batch${batch_size}"
        result="${output_dir}/${stem}.jsonl"
        log="${output_dir}/logs/${stem}.log"

        printf '\n== %s batch=%s ==\n' "${model}" "${batch_size}"
        uv run stt transcribe "${audio_dir}" \
            --backend omniasr-torch \
            --model "${model}" \
            --language mya_Mymr \
            --limit "${limit}" \
            --device "${device}" \
            --dtype "${dtype}" \
            --batch-size "${batch_size}" \
            --output "${result}" \
            --no-show >"${log}" 2>&1

        uv run stt eval "${result}" \
            --reference "${reference_for_eval}" \
            --allow-partial | tee "${output_dir}/${stem}.score.txt"
    done
done

printf '\nCTC product gate artifacts: %s\n' "${output_dir}"
