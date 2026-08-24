# Contributing

Follow [`docs/GIT_WORKFLOW.md`](docs/GIT_WORKFLOW.md) for the repository's
branch, commit, pull-request, merge, and validation policy.

## Start

```bash
git switch main
git pull --ff-only
git switch -c feat/short-description
make hooks
```

Use Conventional Commits, for example:

```text
feat(backend): add runtime capability probe
fix(seamless): preserve boundary context
docs(workflow): clarify pull-request checks
```

Before opening a pull request, run the universal checks in
[`docs/GIT_WORKFLOW.md`](docs/GIT_WORKFLOW.md) and record any applicable
real-model commands and results in the pull request.
