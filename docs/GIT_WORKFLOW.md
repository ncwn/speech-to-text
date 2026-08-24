# Git And Pull Request Workflow

This is the authoritative source for repository branch, commit, pull-request,
merge, and validation policy. GitHub branch protection is configured outside the
repository; keep it aligned with the `branch-policy` and `offline-tests` jobs in
[`CI`](../.github/workflows/ci.yml).

## Protected Branch

`main` accepts changes only through pull requests. Required checks must pass and
review conversations must be resolved before merge. Use merge commits; squash
and rebase merges are not part of this workflow.

## Branches

Start from the current `main` and use one short-lived branch for one coherent
outcome. Branch syntax and accepted prefixes are enforced by
[`scripts/check_branch_name.sh`](../scripts/check_branch_name.sh); do not copy
that list into other documentation.

Open a draft pull request early when work spans multiple commits.

## Commits

Use Conventional Commits in the form `type(scope): imperative summary`. Keep
commits reviewable and independently green. Do not commit WIP markers, secrets,
model weights, caches, downloaded data, or generated outputs; repository ignore
patterns are maintained in [`.gitignore`](../.gitignore).

## Pull Request Lifecycle

1. Update `main` with `git pull --ff-only`.
2. Create a typed branch and make focused changes, tests, and documentation.
3. Run the universal validation below.
4. Push and open a pull request against `main`.
5. Record applicable real-model commands and results for runtime/model changes.
6. Resolve review conversations, then merge with a merge commit after required
   checks pass and sync local `main`.

## Validation

Run these checks before opening every pull request:

```bash
git diff --check
make check
```

For runtime or model changes, the pull request must also record each applicable
real-model command and result, including the model, device, input set, and output
location. This is additional to the universal checks; the repository does not
assume one fixed measurement command.
