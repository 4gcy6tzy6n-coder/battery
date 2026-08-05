.PHONY: install test lint phase1 phase2a-smoke

install:
	python -m pip install -e '.[dev]'

test:
	python -m pytest -q

lint:
	ruff check src tests scripts

phase1: test lint
	python scripts/download_nasa.py
	python scripts/inventory_nasa.py

phase2a-smoke:
	python scripts/build_dataset.py --config configs/data/nasa_randomized.yaml --archive data/raw/battery_alt_dataset.zip --output data/processed/nasa_randomized-smoke --limit-batteries 6 --limit-cycles 3
