# Fast offline guards.

.DEFAULT_GOAL := check
.PHONY: check lint test hooks

check: lint test

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest -q

hooks:
	git config core.hooksPath .githooks
	@echo "pre-commit hook enabled"
