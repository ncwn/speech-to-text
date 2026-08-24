#!/usr/bin/env bash

set -euo pipefail

branch="${1:-$(git branch --show-current)}"
if [[ -z "${branch}" ]]; then
    exit 0
fi
if [[ "${branch}" == "main" ]]; then
    printf 'Direct work on main is prohibited; create a typed branch first.\n' >&2
    exit 1
fi
case "${branch}" in
    feat/*|fix/*|perf/*|refactor/*|docs/*|test/*|research/*|chore/*|ci/*|build/*|revert/*|dependabot/*|renovate/*)
        ;;
    *)
        printf 'Invalid branch name: %s\n' "${branch}" >&2
        printf 'Use type/lowercase-kebab-case, for example feat/live-transcription.\n' >&2
        exit 1
        ;;
esac
suffix="${branch#*/}"
if [[ ! "${suffix}" =~ ^[a-z0-9]+(-[a-z0-9]+)*$ ]]; then
    printf 'Invalid branch suffix: %s\n' "${suffix}" >&2
    printf 'Use lowercase kebab-case after the branch type.\n' >&2
    exit 1
fi
