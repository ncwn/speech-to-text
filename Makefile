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

## bench: isolated smoke comparison against the historical baseline, plus the
## tests that need real weights. The old schema is transcript-only; timings are
## not comparable until a trusted v2 baseline exists.
bench:
	uv run stt bench
	uv run pytest -m weights -q

## bench-update: deliberately blocked until immutable model provenance and the
## remaining measurement acceptance gates are implemented.
bench-update:
	@echo "bench update blocked: complete the measurement-plan acceptance gates first"
	@false

## hooks: point git at the tracked hooks directory
hooks:
	git config core.hooksPath .githooks
	@echo "pre-commit hook enabled (make check on every commit)"

all: check bench
