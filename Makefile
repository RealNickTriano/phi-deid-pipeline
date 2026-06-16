.PHONY: install model test eval api clean

install:
	uv sync

model:
	uv run python -m spacy download en_core_web_sm

test:
	uv run pytest -q

eval:
	uv run python scripts/run_eval.py --n 100 --show

api:
	uv run uvicorn phi_deid.api:app --reload

clean:
	rm -rf .pytest_cache **/__pycache__ src/*.egg-info
