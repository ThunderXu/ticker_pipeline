import os
import pandas as pd
import torch.nn as nn
import yaml
from unittest.mock import patch
import pytest
import ticker_pipeline as pipeline


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
         patch.object(pipeline.FeatureSelector, "run_feature_selection", return_value=(['AAPL'], 0.99, [])):

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


@pytest.mark.parametrize(
    "skip_crawler, skip_fe, skip_fs, skip_train, expect",
    [
        # none skipped -> all main steps called
        (False, False, False, False, {"fetch": True, "save": True, "fe": True, "read_csv": True, "fs": True, "train": True, "save_model": True}),
        # everything skipped -> none called
        (True, True, True, True, {"fetch": False, "save": False, "fe": False, "read_csv": False, "fs": False, "train": False, "save_model": False}),
        # skip crawler only -> fe/fs/train should still run
        (True, False, False, False, {"fetch": False, "save": False, "fe": True, "read_csv": True, "fs": True, "train": True, "save_model": True}),
        # skip everything before training
        (True, True, True, False, {"fetch": False, "save": False, "fe": False, "read_csv": False, "fs": False, "train": True, "save_model": True}),
    ],
)
def test_pipeline_respects_skip_flags(tmp_path, skip_crawler, skip_fe, skip_fs, skip_train, expect):
    # build config returned by load_config
    cfg = {
        "source_tickers": ["AAPL", "SCHB"],
        "other_etfs": [],
        "interval": "1h",
        "raw_csv": str(tmp_path / "raw.csv"),
        "engineered_csv": str(tmp_path / "engineered.csv"),
        "model_out": str(tmp_path / "ticker_model.pth"),
        "target": "target",
        "trainer": {"batch_size": 8, "epochs": 1, "lr": 0.01, "train_ratio": 0.8},
        "feature_selection": {"enabled": True, "max_features": 2, "epochs": 1},
        "skip_crawler": skip_crawler,
        "skip_feature_engineering": skip_fe,
        "skip_feature_selection": skip_fs,
        "skip_training": skip_train,
    }

    # Setup mocks for components
    with patch.object(pipeline, "load_config", return_value=cfg), \
         patch.object(pipeline.StockDataCrawler, "fetch_data", autospec=True) as mock_fetch, \
         patch.object(pipeline.StockDataCrawler, "save_to_csv", autospec=True) as mock_save_csv, \
         patch.object(pipeline.FeatureEngineer, "process_file", autospec=True) as mock_fe_process, \
         patch.object(pipeline.pd, "read_csv", autospec=True) as mock_read_csv, \
         patch.object(pipeline.FeatureSelector, "run_feature_selection", autospec=True) as mock_fs_run, \
         patch.object(pipeline.Trainer, "train", autospec=True) as mock_train, \
         patch.object(pipeline.Trainer, "save_model", autospec=True) as mock_save_model:

        # configure return values for mocks that will be "called"
        mock_fetch.return_value = True
        # feature engineer returns a small DataFrame when called
        mock_fe_process.return_value = pd.DataFrame({
            "AAPL": [0.1, 0.2],
            "SCHB": [0.2, -0.1],
            "target": [1, 0],
        })
        # feature selection returns a chosen list when called
        mock_fs_run.return_value = (["AAPL"], 0.9, [])
        # trainer.train returns a dummy object
        mock_train.return_value = object()
        mock_save_csv.side_effect = lambda self: _write_sample_raw_csv(cfg["raw_csv"])

        # run pipeline (it will use our patched load_config)
        pipeline.run_pipeline_from_config(config_path=None)

    # Assert expected call behavior
    assert mock_fetch.called == expect["fetch"]
    assert mock_save_csv.called == expect["save"]
    assert mock_fe_process.called == expect["fe"]
    assert mock_fs_run.called == expect["fs"]
    assert mock_train.called == expect["train"]
    assert mock_save_model.called == expect["save_model"]
    assert mock_read_csv.called == expect["read_csv"]
