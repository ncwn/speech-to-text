# Git And Pull Request Workflow

## Protected Branch

`main` is always releasable. It accepts changes only through pull requests.
Force pushes, branch deletion, and direct pushes are disabled. Required checks
must pass and review conversations must be resolved before merge.

For a solo maintainer, a PR plus passing checks and the completed PR checklist
is the review record. Require an independent approval when another maintainer
is available.

## Branches

Create branches from current `main` using lowercase kebab-case:

- `feat/<short-name>`
- `fix/<short-name>`
- `perf/<short-name>`
- `refactor/<short-name>`
- `docs/<short-name>`
- `test/<short-name>`
- `research/<short-name>`
- `chore/<short-name>`
- `ci/<short-name>`
- `build/<short-name>`
- `revert/<short-name>`

One branch owns one coherent outcome. Open a draft PR early for work that spans
multiple commits.

## Commits

Use `type(scope): imperative summary`. Commits should be reviewable and pass the
relevant local checks. Do not commit WIP markers, secrets, weights, caches,
downloaded data, or generated outputs.

Benchmark and evidence artifacts bind the adapter commit SHA. Use GitHub's
**Create a merge commit** option. Do not squash or rebase evidence-bearing PRs,
because rewriting the commit invalidates that identity.

## Pull Request Lifecycle

1. Update `main` with `git pull --ff-only`.
2. Create a typed branch.
3. Make small Conventional Commits.
4. Add focused tests and update durable documentation.
5. Run local validation.
6. Push and open a draft or ready PR against `main`.
7. Resolve review comments and rerun affected checks.
8. Merge with a merge commit only after required checks pass.
9. Delete the remote branch and sync local `main`.

## Validation Matrix

Every PR:

```bash
git diff --check
make check
```

Evidence or measured documentation:

```bash
uv run stt evidence --check
```

Backend/runtime changes also run focused real-weight tests and the relevant
experiment or parity lane. Baseline updates require the clean-tree protocol in
`Makefile`; never refresh a baseline merely to make a check pass.

## Emergency Changes

Do not bypass branch protection for urgency. Create a `fix/...` branch and PR.
If GitHub is unavailable and an emergency commit is unavoidable, record the
exception and immediately follow it with a PR that documents validation and
restores the protected workflow.

## GitHub Protection

The `main` ruleset requires:

- pull request before merge
- passing `branch-policy` and `offline-tests` checks
- resolved review conversations
- merge commits allowed; squash/rebase disabled
- force pushes and deletion disabled
- administrators included
