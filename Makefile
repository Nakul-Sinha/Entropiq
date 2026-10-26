.PHONY: setup lint fmt typecheck test up down

setup:
	uv sync
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy packages/core

test:
	uv run pytest

up:
	@echo "docker compose stack arrives in P7.4"

down:
	@echo "docker compose stack arrives in P7.4"
