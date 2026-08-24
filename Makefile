# Guards for this repo. `check` is the fast pre-commit and CI lane; real model
# benchmarks remain explicit targets.

.DEFAULT_GOAL := check
.PHONY: check lint test bench bench-update hooks all

## check: lint, format and the offline test suite
check: lint test

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest -q

## bench: isolated smoke comparison plus tests that need real weights
bench:
	uv run stt bench
	uv run pytest -m weights -q

## bench-update: clean-tree, trusted baseline-v2 refresh protocol
bench-update:
	uv run stt bench --update --workers 5 --warmups 3 --repeats 3 --no-profile \
		--accept-transcript-changes \
		--approval-note "explicit baseline-v2 refresh approval"

hooks:
	git config core.hooksPath .githooks
	@echo "pre-commit hook enabled"

all: check bench
