import os
import pandas as pd
import torch.nn as nn
import yaml
import ticker_pipeline as pipeline
from unittest.mock import patch


def _write_sample_raw_csv(path):
    # construct a small raw CSV that FeatureEngineer expects (AAPL, target)
    idx = pd.date_range("2023-01-01", periods=6, freq="h")
    df = pd.DataFrame({
        "AAPL": [1.0, 2.0, 1.5, -0.5, 0.0, 0.2],
        "target": [0.5, -0.2, 0.3, 0.0, 0.1, -0.4],
    }, index=idx)
    df.to_csv(path, index=True)


def test_run_pipeline_happy_path(tmp_path):
    # prepare temp paths and write a small YAML config
    raw_csv = str(tmp_path / "raw.csv")
    eng_csv = str(tmp_path / "engineered.csv")
    model_out = str(tmp_path / "ticker_model.pth")

    cfg = {
        "source_tickers": ["AAPL"],
        "other_etfs": [],
        "interval": "1h",
        "raw_csv": raw_csv,
        "engineered_csv": eng_csv,
        "model_out": model_out,
        "target": "target",
        "trainer": {"batch_size": 8, "epochs": 1, "lr": 0.01, "train_ratio": 0.8},
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as fh:
        yaml.safe_dump(cfg, fh)

    # patch crawler.fetch_data to return True and save_to_csv to write known raw CSV
    def fake_fetch(self):
        return True

    def fake_save(self):
        _write_sample_raw_csv(raw_csv)

    # patch Trainer.train to avoid real training and save_model to write dummy file
    def fake_train(self):
        # return a tiny dummy model
        return nn.Linear(1, 1)

    def fake_save_model(self, out_path):
        # create an empty file to simulate saved model
        open(out_path, "wb").close()

    # use unittest.mock.patch to temporarily replace attributes
    with patch.object(pipeline.StockDataCrawler, "fetch_data", new=fake_fetch), \
         patch.object(pipeline.StockDataCrawler, "save_to_csv", new=fake_save), \
         patch.object(pipeline.Trainer, "train", new=fake_train), \
         patch.object(pipeline.Trainer, "save_model", new=fake_save_model), \
         patch("feature_selection.run_feature_selection", return_value=(['AAPL'], 0.99, [])):

        # run pipeline using the YAML config
        pipeline.run_pipeline_from_config(str(cfg_path))

    assert os.path.exists(raw_csv), "Raw CSV should be created by fake_save"
    raw_df = pd.read_csv(raw_csv, index_col=0)
    assert not raw_df.empty
    # assertions: engineered CSV created and model file saved by fake_save_model
    assert os.path.exists(eng_csv), "Engineered CSV should be created"
    eng_df = pd.read_csv(eng_csv, index_col=0)
    assert not eng_df.empty
    assert os.path.exists(model_out)
