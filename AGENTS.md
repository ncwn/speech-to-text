# Repository Workflow

[`docs/GIT_WORKFLOW.md`](docs/GIT_WORKFLOW.md) is the authoritative source for
branch, commit, pull-request, merge, and validation policy.

- Start from current `main` on a typed branch; never commit or push directly to
  `main`.
- Run `make check` before committing and the universal checks in the workflow
  before opening a pull request.
- Keep model weights, caches, downloaded data, and generated outputs out of Git.
