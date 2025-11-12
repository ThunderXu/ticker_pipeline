import os
from typing import Optional

import yaml
import os
from typing import Optional
import pandas as pd
import yaml

from ticker_data_crawler import StockDataCrawler
from feature_engineering import FeatureEngineer
import feature_selection as fs
from ticker_trainer import Trainer


def load_config(path: Optional[str] = None) -> dict:
    """Load pipeline configuration from YAML. If path is None, load `config.yaml` next to this file."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def run_pipeline(stocks, etfs, interval: str = "1h"):
    """Backward-compatible wrapper: runs crawler -> feature engineering -> trainer using module-level arguments.

    This keeps the original behavior for callers that call run_pipeline directly.
    """
    RAW_CSV = "raw_ticker_changes.csv"
    ENGINEERED_CSV = "engineered_ticker.csv"

    # 1) fetch raw changes
    crawler = StockDataCrawler(stocks=stocks, etfs=etfs, interval=interval)
    success = crawler.fetch_data()
    crawler.save_to_csv()  # always writes a file (may be empty)

    if not success:
        print("No raw data fetched — aborting pipeline.")
        return

    if not os.path.exists(RAW_CSV):
        print("Raw CSV missing after fetch — aborting.")
        return

    # 2) feature engineer
    fe = FeatureEngineer(target_col="target", lags=(1, 2), binarize_target=True)
    engineered_df = fe.process_file(RAW_CSV, out_csv_path=ENGINEERED_CSV)
    if engineered_df.empty:
        print("Engineered data is empty — aborting.")
        return
    print(f"Feature engineering produced {len(engineered_df)} rows -> {ENGINEERED_CSV}")

    # 3) train
    trainer = Trainer(engineered_csv=ENGINEERED_CSV, batch_size=64, epochs=200, lr=1e-3, train_ratio=0.8)
    model = trainer.train()
    trainer.save_model("ticker_model.pth")


def run_pipeline_from_config(config_path: Optional[str] = None):
    """Run the full pipeline using a YAML configuration file.

    config keys used:
      source_tickers: list of tickers
      other_etfs: list of tickers
      interval: interval string
      raw_csv: path to write raw CSV
      engineered_csv: path to write engineered CSV
      model_out: path to save model
      trainer: {batch_size, epochs, lr, train_ratio}
    """
    cfg = load_config(config_path)

    # read source tickers from config
    source_tickers = cfg.get("source_tickers", [])
    other_etfs = cfg.get("other_etfs", [])
    interval = cfg.get("interval", "1h")
    raw_csv = cfg.get("raw_csv", "raw_ticker_changes.csv")
    eng_csv = cfg.get("engineered_csv", "engineered_ticker.csv")
    model_out = cfg.get("model_out", "ticker_model.pth")
    target = cfg.get("target", "target")
    trainer_cfg = cfg.get("trainer", {})
    skip_crawler = cfg.get("skip_crawler", False)
    skip_feature_engineering = cfg.get("skip_feature_engineering", False)
    skip_feature_selection = cfg.get("skip_feature_selection", False)
    skip_training = cfg.get("skip_training", False)

    if not skip_crawler:
        # 1) fetch raw changes
        crawler = StockDataCrawler(stocks=source_tickers, etfs=other_etfs, interval=interval, target=target)
        # let tests override behavior by patching methods; ensure crawler writes to the configured raw_csv
        crawler.OUTPUT_CSV = raw_csv
        success = crawler.fetch_data()
        crawler.save_to_csv()

        if not success:
            print("No raw data fetched — aborting pipeline.")
            return

        if not os.path.exists(raw_csv):
            print("Raw CSV missing after fetch — aborting.")
            return
    if not skip_feature_engineering:
        # 2) feature engineer
        fe = FeatureEngineer(target_col=target, lags=(1,2), binarize_target=True)
        engineered_df = fe.process_file(raw_csv, out_csv_path=eng_csv)
        if engineered_df.empty:
            print("Engineered data is empty — aborting.")
            return
        print(f"Feature engineering produced {len(engineered_df)} rows -> {eng_csv}")

    if not skip_feature_selection:
        # Optional: feature selection (configurable)
        selection_cfg = cfg.get("feature_selection", {})
        if selection_cfg.get("enabled", True):
            max_feats = selection_cfg.get("max_features", 5)
            # pass through some eval params (epochs, batch_size, lr)
            eval_params = {
                "epochs": selection_cfg.get("epochs", 20),
                "batch_size": selection_cfg.get("batch_size", 64),
                "lr": selection_cfg.get("lr", 1e-3),
            }
            print(f"Running feature selection (max_features={max_feats})...")
            try:
                selected, best_acc, history = fs.run_feature_selection(
                    eng_csv, max_features=max_feats, target_col=target, **eval_params
                )
            except Exception as e:
                print(f"Feature selection failed: {e}. Continuing with full feature set.")
                selected = []
                best_acc = 0.0

            print(f"Feature selection result: selected={selected}, best_acc={best_acc:.4f}")
            if selected:
                engineered_df = pd.read_csv(eng_csv, index_col=0)
                # reduce engineered_df to selected tickers columns + target and overwrite eng_csv
                reduced = fs._select_columns_for_tickers(engineered_df, selected, target_col=target)
                reduced.to_csv(eng_csv)
                print(f"Engineered CSV reduced to selected features and saved to {eng_csv} (rows={len(reduced)})")

    if not skip_training:
        # 3) train
        bs = trainer_cfg.get("batch_size", 64)
        epochs = trainer_cfg.get("epochs", 200)
        lr = trainer_cfg.get("lr", 1e-3)
        train_ratio = trainer_cfg.get("train_ratio", 0.8)

        trainer = Trainer(engineered_csv=eng_csv, batch_size=bs, epochs=epochs, lr=lr, train_ratio=train_ratio, target_col=target)
        model = trainer.train()
        print("saving model to", model_out)
        trainer.save_model(model_out)


if __name__ == "__main__":
    # run with default config.yaml next to this file
    run_pipeline_from_config()
