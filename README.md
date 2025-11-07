# Ticker prediction pipeline

Small project to fetch price data, feature-engineer it, train a small PyTorch model, and save the model. The repo includes a simple Airflow DAG example and tests.

## What it does
- `ticker_data_crawler.py` — fetches raw price data (Close - Open) for tickers and writes a raw CSV.
- `feature_engineering.py` — reusable feature-engineering utilities: lag features, binarize target, write engineered CSV and helper to convert to tensors.
- `ticker_pipeline.py` — pipeline runner (config-driven). It reads `config.yaml`, runs crawler -> feature engineering -> trainer.
## What it does
- `ticker_data_crawler.py` — fetches raw price data (Close - Open) for tickers and writes a raw CSV.
- `feature_engineering.py` — reusable feature-engineering utilities: lag features, binarize target, write engineered CSV and helper to convert to tensors.
- `ticker_trainer.py` — Trainer class that trains a small PyTorch network on the engineered CSV.
- `ticker_pipeline.py` — pipeline runner (config-driven). It reads `config.yaml`, runs crawler -> feature engineering -> trainer.

## Quick start (recommended: use a virtualenv)

On macOS install CPU-only PyTorch and deps (adjust if you have a GPU):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# CPU wheel example (adjust to your platform/OS)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install pandas scikit-learn pyyaml yfinance pytest
```

## Configuration
Edit `config.yaml` (a default exists) or create your own YAML with keys:

```yaml
source_tickers: [SCHB, QQQ]
other_etfs: []
interval: "1h"
raw_csv: raw_ticker_changes.csv
engineered_csv: engineered_ticker.csv
model_out: ticker_model.pth
target: "target"
trainer:
  batch_size: 64
  epochs: 200
  lr: 0.001
  train_ratio: 0.8
feature_selection:
  enabled: true
  max_features: 5
```

## Run the pipeline

```bash
python -m ticker_pipeline
```

Or call the runner from Python with a custom config file:

```bash
python -c "from ticker_pipeline import run_pipeline_from_config; run_pipeline_from_config('path/to/config.yaml')"
```

## Tests
The repository includes pytest tests that patch network calls and training so they run quickly.

Install pytest and run:

```bash
pytest -q
```

## Notes
- Defaults use CPU PyTorch; pick the appropriate install for your platform.
- `feature_engineering.py` performs lagging/binarization and writes engineered CSVs consumed by the trainer.

## Files of interest
- `config.yaml` — default config
- `ticker_pipeline.py` — pipeline runner (config-driven; filename retained for compatibility)
- `feature_engineering.py` — feature logic
- `ticker_trainer.py` — Trainer and model
- `ticker_data_crawler.py` — data fetcher
- `test_*.py` — pytest tests
