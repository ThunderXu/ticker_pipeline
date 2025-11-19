import os
import pandas as pd
import numpy as np
import torch

import predictor
from feature_engineering import FeatureEngineer


def make_ohlc_df(num_rows=6):
    # create a simple increasing OHLC so that Close-Open positive
    idx = pd.date_range(end=pd.Timestamp.now(), periods=num_rows, freq='H')
    open_ = np.linspace(100.0, 105.0, num_rows)
    close = open_ + 0.5  # always positive change
    df = pd.DataFrame({'Open': open_, 'High': close + 0.1, 'Low': open_ - 0.1, 'Close': close}, index=idx)
    return df


def test_predict_once_alignment(tmp_path, monkeypatch):
    # prepare engineered csv using three tickers and lags (1,2) => 3 + 6 = 9 features
    tickers = ['A', 'B', 'C']
    target = 'T'

    # build a raw historical df for engineering
    # index aligned timestamps
    idx = pd.date_range(end=pd.Timestamp.now(), periods=10, freq='H')
    data = {}
    for t in tickers + [target]:
        data[t] = np.linspace(1.0, 2.0, len(idx))
    raw_hist = pd.DataFrame(data, index=idx)

    fe = FeatureEngineer(target_col=target, lags=(1,2), binarize_target=False)
    eng = fe.transform_df(raw_hist)

    eng_csv = tmp_path / 'eng.csv'
    eng.to_csv(eng_csv)

    # monkeypatch yf.download to return an OHLC df for any ticker. Create one
    # DataFrame and return the same object for each call so indexes align.
    df_ohlc = make_ohlc_df(num_rows=6)

    def fake_download(ticker, start, end, interval, progress, auto_adjust):
        return df_ohlc

    monkeypatch.setattr(predictor.yf, 'download', fake_download)

    # fit scaler and feature cols from engineered csv
    scaler, feature_cols = predictor.fit_scaler_from_engineered(str(eng_csv), target_col=target)

    # create a dummy model that returns logits producing prob>0.5
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return torch.full((x.size(0), 1), 2.0, device=x.device)

    model = DummyModel()

    cfg = {
        'source_tickers': tickers,
        'other_etfs': [],
        'target': target,
        'interval': '1h',
    }

    res = predictor.predict_once(cfg, model, scaler, feature_cols)
    assert res.get('ok') is True
    assert 'pred_prob' in res
    assert 0.0 <= res['pred_prob'] <= 1.0


def test_predict_missing_feature(tmp_path, monkeypatch):
    # engineered csv has only tickers A and B, but live data includes A,B,C -> predictor should require exact features
    tickers_train = ['A', 'B']
    tickers_live = ['A', 'B', 'C']
    target = 'T'

    idx = pd.date_range(end=pd.Timestamp.now(), periods=10, freq='H')
    data = {}
    for t in tickers_train + [target]:
        data[t] = np.linspace(1.0, 2.0, len(idx))
    raw_hist = pd.DataFrame(data, index=idx)
    fe = FeatureEngineer(target_col=target, lags=(1,2), binarize_target=False)
    eng = fe.transform_df(raw_hist)
    eng_csv = tmp_path / 'eng2.csv'
    eng.to_csv(eng_csv)

    # fake download returns OHLC for A,B,C; return same df for each call so indexes align
    df_ohlc = make_ohlc_df(num_rows=6)

    def fake_download(ticker, start, end, interval, progress, auto_adjust):
        return df_ohlc

    monkeypatch.setattr(predictor.yf, 'download', fake_download)

    scaler, feature_cols = predictor.fit_scaler_from_engineered(str(eng_csv), target_col=target)

    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return torch.full((x.size(0), 1), 2.0, device=x.device)

    model = DummyModel()

    cfg = {
        'source_tickers': tickers_live,
        'other_etfs': [],
        'target': target,
        'interval': '1h',
    }

    res = predictor.predict_once(cfg, model, scaler, feature_cols)
    # since live sample will contain C_prev* features but feature_cols expect only A/B features
    # The predictor should still succeed if the feature_cols are a subset of sample columns; here
    # feature_cols come from engineered CSV (A/B plus lags) and should be present in sample -> ok
    assert res.get('ok') is True