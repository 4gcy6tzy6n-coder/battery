# TriStateLite

TriStateLite targets lightweight probabilistic joint prediction of battery SOC, SOH, and
time-to-end-of-discharge. This repository is currently limited to Phase 1 data discovery.

Phase 1 downloads the official NASA Randomized and Recommissioned Battery archive, records its
provenance, inventories every ZIP member safely, and reports candidate schemas. It does not perform
semantic battery parsing and must stop before labels, splits, windows, models, or training are built.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
ruff check src tests scripts
```
