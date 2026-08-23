# Guards for this repo. `check` is what the pre-commit hook runs; it must stay
# fast enough that nobody is tempted to disable it. `bench` loads real models
# and takes minutes, which is exactly why it is not in the hook.

.DEFAULT_GOAL := check
.PHONY: check lint test bench bench-update hooks all

## check: lint, format and the offline test suite (~10 s)
check: lint test

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest -q

## bench: isolated smoke comparison against baseline-v2, plus the tests that
## need real weights.
bench:
	uv run stt bench
	uv run pytest -m weights -q

## bench-update: rewrite baseline-v2 only from a clean tree and the trusted
## five-session protocol. Transcript changes require explicit approval.
bench-update:
	uv run stt bench --update --workers 5 --warmups 3 --repeats 3 --no-profile \
		--accept-transcript-changes \
		--approval-note "explicit baseline-v2 refresh approval"

## hooks: point git at the tracked hooks directory
hooks:
	git config core.hooksPath .githooks
	@echo "pre-commit hook enabled (make check on every commit)"

all: check bench
