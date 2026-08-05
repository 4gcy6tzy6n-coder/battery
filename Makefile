.PHONY: install test lint phase1

install:
	python -m pip install -e '.[dev]'

test:
	python -m pytest -q

lint:
	ruff check src tests scripts

phase1: test lint
	python scripts/download_nasa.py
	python scripts/inventory_nasa.py
