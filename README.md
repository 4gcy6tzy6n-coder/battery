# TriStateLite

TriStateLite targets lightweight probabilistic joint prediction of battery SOC, SOH, and
time-to-end-of-discharge. Phase 2A builds audited, leakage-safe data artifacts and stops before
model, loss, training, and evaluation implementation.

Phase 1 downloads and inventories the official NASA Randomized and Recommissioned Battery archive.
The executed source audit then fixes the Phase 2A semantics: maximal contiguous `mode == -1`
discharges, causal 1 Hz sampling, positive discharge current, and physical-battery isolation.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/download_nasa.py
python scripts/inventory_nasa.py
make phase2a-smoke
```

For code quality, run:

```bash
ruff check src tests scripts
```

The smoke target builds six batteries and three cycles per battery into
`data/processed/nasa_randomized-smoke`. It writes partitioned Parquet samples, cycle summaries,
split/scaler provenance, rejection reasons, a leakage audit, and a build report. See
`docs/data_dictionary.md` and `docs/leakage_contract.md` for the enforceable contracts.
