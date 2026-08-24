# Contributing

All changes use a short-lived branch and a pull request. Direct commits to
`main` are prohibited.

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
perf(batch): reduce decoder idle time
docs(evidence): explain provenance gate
test(ctc): cover empty-output failure
```

Before opening a PR, run `make check` and complete the repository-specific
validation matrix in [`docs/GIT_WORKFLOW.md`](docs/GIT_WORKFLOW.md).

PRs are merged with a merge commit after checks and review. Evidence-bearing
commits must retain their original SHA, so squash and rebase merges are not
allowed.
