# Repository Workflow

- Never commit or push directly to `main`.
- Start each implementation from current `main` on a typed branch such as
  `feat/...`, `fix/...`, `perf/...`, `docs/...`, `test/...`, or `chore/...`.
- Open a pull request targeting `main`; do not merge until required checks pass
  and review conversations are resolved.
- Use Conventional Commit messages and keep each commit independently green.
- Use merge commits for PRs. Do not squash or rebase evidence-bearing commits;
  benchmark provenance records the adapter commit SHA.
- Keep model weights, local caches, downloaded corpora, and diagnostic outputs
  out of Git.
- Run `make check` before every commit. Runtime/evidence changes require the
  additional validation described in `docs/GIT_WORKFLOW.md`.
