import pandas as pd
import numpy as np
from feature_engineering import FeatureEngineer, df_to_tensors
import pytest

def make_sample_raw_csv(path):
    # create simple raw changes CSV with AAPL and target columns
    idx = pd.date_range("2023-01-01", periods=6, freq="h")
    df = pd.DataFrame({
        "AAPL": [1.0, 2.0, 1.5, -0.5, 0.0, 0.2],
        "target": [0.5, -0.2, 0.3, 0.0, 0.1, -0.4],
    }, index=idx)
    df.to_csv(path, index=True)
    return df

def test_feature_engineer_process_file_and_tensors(tmp_path):
    raw_csv = tmp_path / "raw.csv"
    engineered_csv = tmp_path / "engineered.csv"
    make_sample_raw_csv(raw_csv)

    fe = FeatureEngineer(target_col="target", lags=(1,2), binarize_target=True)
    eng_df = fe.process_file(str(raw_csv), out_csv_path=str(engineered_csv))

    # should produce lagged columns for AAPL
    assert "AAPL_prev1" in eng_df.columns
    assert "AAPL_prev2" in eng_df.columns
    assert "target" in eng_df.columns

    # target should be binarized (0/1)
    assert eng_df["target"].isin([0,1]).all()

    # no NaNs, rows less than raw (due to lag drops)
    assert not eng_df.isnull().any().any()
    assert len(eng_df) <= 6

    # df_to_tensors returns X,y shapes consistent with engineered df
    X, y, features = df_to_tensors(eng_df, target_col="target")
    assert X.shape[0] == y.shape[0] == len(eng_df)
    assert X.shape[1] == len(features)