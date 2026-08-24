# Fast offline guards. Real model benchmarks remain explicit targets.

.DEFAULT_GOAL := check
.PHONY: check lint test bench bench-update hooks all

check: lint test

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest -q

bench:
	uv run stt bench
	uv run pytest -m weights -q

bench-update:
	uv run stt bench --update --workers 5 --warmups 3 --repeats 3 --no-profile \
		--accept-transcript-changes \
		--approval-note "explicit baseline refresh approval"

hooks:
	git config core.hooksPath .githooks
	@echo "pre-commit hook enabled"

all: check bench
