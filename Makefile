.PHONY: install lint format test train serve

install:
	uv sync --all-extras

lint:
	uv run ruff check .
	uv run mypy src tests

format:
	uv run ruff format .

test:
	uv run pytest

train:
	uv run python -m energy_price_mlops.training.train

serve:
	uv run uvicorn energy_price_mlops.serving.app:app --reload --host 0.0.0.0 --port 8000

