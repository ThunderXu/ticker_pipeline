"""Realtime predictor that uses the trained model and config.yaml to predict
and validate ticker direction during market open hours.

Behavior:
- Loads configuration from `config.yaml` (or a path passed to `main`).
- Loads the trained model weights from `model_out` and infers feature dimension
  from `engineered_csv` saved by the training pipeline.
- Fits a StandardScaler on the historical engineered CSV (so live features
  are standardized consistently with training data).
- On each interval (configurable), fetches recent price data via yfinance,
  computes raw change features, runs the same `FeatureEngineer.transform_df`
  to produce lagged features, then predicts the next interval change (binary).
- Sleeps until the next interval ends and then fetches the realized label to
  validate the prediction. Results are appended to `predictions.csv`.

This script is intentionally conservative: it uses the existing project
components (FeatureEngineer, SimpleNN) and stores outputs locally.
"""

import time
import yaml
import os
import argparse
import datetime
from zoneinfo import ZoneInfo
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import torch

from feature_engineering import FeatureEngineer, df_to_tensors
from ticker_trainer import SimpleNN


DEFAULT_PREDICTIONS_CSV = "predictions.csv"


def load_config(path: Optional[str] = None) -> dict:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def interval_to_seconds(interval: str) -> int:
    # support common yfinance intervals
    mapping = {"1m": 60, "2m": 120, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "1d": 86400}
    return mapping.get(interval, 3600)


def is_us_market_open(now_utc: Optional[datetime.datetime] = None) -> bool:
    # US market hours: 09:30 - 16:00 US/Eastern (regular trading hours)
    now_utc = now_utc or datetime.datetime.now(datetime.timezone.utc)
    eastern = now_utc.astimezone(ZoneInfo("US/Eastern"))
    # Weekdays Mon-Fri
    if eastern.weekday() >= 5:
        return False
    open_time = eastern.replace(hour=9, minute=30, second=0, microsecond=0)
    close_time = eastern.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_time <= eastern <= close_time


def fit_scaler_from_engineered(engineered_csv: str, target_col: str = "target") -> Tuple[StandardScaler, List[str]]:
    """Fit a StandardScaler on the engineered CSV and return it plus the
    ordered list of feature column names (excludes the target column).
    """
    df = pd.read_csv(engineered_csv, index_col=0)
    if target_col not in df.columns:
        raise ValueError(f"Engineered CSV missing target column '{target_col}'")
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].values.astype("float32")
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler, feature_cols


def load_model_weights(model_path: str, input_dim: int, device: Optional[torch.device] = None) -> torch.nn.Module:
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = SimpleNN(input_dim).to(device)
    state = torch.load(model_path, map_location=device)
    # If saved as state_dict
    if isinstance(state, dict):
        try:
            model.load_state_dict(state)
        except Exception:
            # saved entire model earlier; try direct load
            model = state
    else:
        model = state
    model.eval()
    return model


def fetch_recent_changes(tickers: List[str], target: str, interval: str, lookback_days: int = 3) -> pd.DataFrame:
    # Download recent OHLC and compute Close - Open for each ticker and name target column 'target'
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=lookback_days)
    all_series = []
    for t in list(tickers) + [target]:
        try:
            df = yf.download(t, start=start, end=end, interval=interval, progress=False, auto_adjust=False)
        except Exception as e:
            print(f"Warning: yf.download failed for {t}: {e}")
            continue
        if df.empty:
            continue
        # compute change series and name it explicitly rather than calling .rename()
        changes = df["Close"] - df["Open"]
        changes.name = "target" if t == target else t
        all_series.append(changes)

    if not all_series:
        return pd.DataFrame()

    out = pd.concat(all_series, axis=1)
    out.dropna(inplace=True)
    # Ensure the configured target column is present and named 'target' so the
    # downstream FeatureEngineer (which expects 'target') can find it. If the
    # raw fetch missed the target ticker, return an empty DataFrame so the
    # caller can handle the missing-data case.
    if target in out.columns and "target" not in out.columns:
        out = out.rename(columns={target: "target"})

    if "target" not in out.columns:
        # target series not available
        print(f"Warning: target column '{target}' not present in fetched data")
        return pd.DataFrame()

    return out


def predict_once(
    cfg: dict,
    model: torch.nn.Module,
    scaler: StandardScaler,
    feature_cols: List[str],
    device: Optional[torch.device] = None,
) -> dict:
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    source_tickers = cfg.get("source_tickers", [])
    other_etfs = cfg.get("other_etfs", [])
    tickers = source_tickers + other_etfs
    target = cfg.get("target", "target")
    interval = cfg.get("interval", "1h")

    raw = fetch_recent_changes(tickers, target, interval, lookback_days=3)
    if raw.empty:
        return {"ok": False, "reason": "no_data"}

    fe = FeatureEngineer(target_col="target", lags=(1, 2), binarize_target=False)
    eng = fe.transform_df(raw)
    if eng.empty:
        return {"ok": False, "reason": "engineered_empty"}

    # use the last row as the most recent features; prediction is about the target value of that row
    sample = eng.tail(1)
    # Ensure sample contains the exact feature columns used during training
    missing = [c for c in feature_cols if c not in sample.columns]
    if missing:
        print(f"Warning: live sample missing expected features: {missing}")
        return {"ok": False, "reason": "missing_features", "missing": missing}
    X = sample[feature_cols].values.astype("float32")
    X_scaled = scaler.transform(X)
    xt = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(xt)
        prob = torch.sigmoid(logits).cpu().numpy().ravel()[0]
        pred_label = int(prob > 0.5)

    return {
        "ok": True,
        "pred_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "pred_prob": float(prob),
        "pred_label": pred_label,
        "features": sample.drop(columns=["target"]).to_dict(orient="records")[0],
    }


def get_actual_label_for_last_interval(tickers: List[str], target: str, interval: str) -> Optional[int]:
    # Fetch a small lookback and compute the last interval's target change sign
    df = fetch_recent_changes(tickers, target, interval, lookback_days=1)
    if df.empty:
        return None
    # fetch_recent_changes normalizes the target column name to 'target'. Prefer
    # that column; fall back to the original ticker name if present.
    if "target" in df.columns:
        last = df["target"].iloc[-1]
    elif target in df.columns:
        last = df[target].iloc[-1]
    else:
        return None
    return int(last > 0)


def save_prediction_row(row: dict, out_csv: str = DEFAULT_PREDICTIONS_CSV):
    """Append a prediction row or update an existing row with the same
    timestamp.

    Behavior:
    - If the CSV does not exist, create it and write the row.
    - If a row exists with the same timestamp, update its empty fields
      (e.g., fill actual_label and outcome) instead of appending a duplicate.
    - Otherwise append the new row.
    """
    df_new = pd.DataFrame([row])
    if not os.path.exists(out_csv):
        df_new.to_csv(out_csv, index=False)
        return

    # Read existing CSV and try to find a matching timestamp to update.
    try:
        df = pd.read_csv(out_csv)
    except Exception:
        # fall back to append if reading fails
        df_new.to_csv(out_csv, mode="a", index=False, header=False)
        return

    if "timestamp" in df.columns and "timestamp" in df_new.columns:
        ts = df_new.at[0, "timestamp"]
        matches = df["timestamp"] == ts
        if matches.any():
            # update the first matching row's empty fields
            idx = matches.idxmax()
            for col in df_new.columns:
                val = df_new.at[0, col]
                # treat NaN/empty string as missing
                existing = df.at[idx, col] if col in df.columns else None
                if col not in df.columns:
                    df[col] = ""
                if (pd.isna(existing) or existing == "" or existing is None) and (val is not None and val != ""):
                    df.at[idx, col] = val
            df.to_csv(out_csv, index=False)
            return

    # no existing row to update -> append
    df_new.to_csv(out_csv, mode="a", index=False, header=False)


def main_once(cfg_path: Optional[str] = None):
    cfg = load_config(cfg_path)
    eng_csv = cfg.get("engineered_csv", "engineered_ticker.csv")
    model_out = cfg.get("model_out", "ticker_model.pth")
    interval = cfg.get("interval", "1h")
    predict_seconds = cfg.get("predict_interval_seconds") or interval_to_seconds(interval)

    scaler, feature_cols = fit_scaler_from_engineered(eng_csv, target_col=cfg.get("target", "target"))

    # infer input_dim
    df_eng = pd.read_csv(eng_csv, index_col=0)
    input_dim = df_eng.drop(columns=[cfg.get("target", "target")]).shape[1]
    model = load_model_weights(model_out, input_dim)

    result = predict_once(cfg, model, scaler, feature_cols)
    if not result.get("ok"):
        print("Prediction aborted:", result.get("reason"))
        return

    print(f"Predicted prob={result['pred_prob']:.4f} label={result['pred_label']}")
    row = {
        "timestamp": result["pred_at"],
        "pred_prob": result["pred_prob"],
        "pred_label": result["pred_label"],
        "actual_label": None,
        "outcome": None,
    }
    save_prediction_row(row)

    # Wait one prediction interval and fetch actual
    wait = predict_seconds
    print(f"Waiting {wait} seconds to fetch actual label...")
    time.sleep(wait)

    actual = get_actual_label_for_last_interval(cfg.get("source_tickers", []) + cfg.get("other_etfs", []), cfg.get("target", "target"), interval)
    if actual is None:
        print("Could not fetch actual label after waiting")
        return

    row["actual_label"] = actual
    row["outcome"] = int(row["pred_label"] == actual)
    save_prediction_row(row)
    print(f"Actual label={actual} -> outcome={row['outcome']}")


def run_loop(cfg_path: Optional[str] = None, once: bool = False, dry_run: bool = False):
    cfg = load_config(cfg_path)
    interval = cfg.get("interval", "1h")
    predict_seconds = cfg.get("predict_interval_seconds") or interval_to_seconds(interval)

    eng_csv = cfg.get("engineered_csv", "engineered_ticker.csv")
    model_out = cfg.get("model_out", "ticker_model.pth")

    scaler, feature_cols = fit_scaler_from_engineered(eng_csv, target_col=cfg.get("target", "target"))
    df_eng = pd.read_csv(eng_csv, index_col=0)
    input_dim = df_eng.drop(columns=[cfg.get("target", "target")]).shape[1]
    model = load_model_weights(model_out, input_dim)

    print("Starting predictor loop. Press Ctrl-C to stop.")
    try:
        while True:
            now = datetime.datetime.now(datetime.timezone.utc)
            if is_us_market_open(now):
                print("Market open: running prediction cycle")
                res = predict_once(cfg, model, scaler, feature_cols)
                if not res.get("ok"):
                    print("Prediction cycle failed:", res.get("reason"))
                else:
                    # save immediate prediction row with no actual yet
                    row = {
                        "timestamp": res["pred_at"],
                        "pred_prob": res["pred_prob"],
                        "pred_label": res["pred_label"],
                        "actual_label": None,
                        "outcome": None,
                    }
                    save_prediction_row(row)

                    if once or dry_run:
                        # in dry-run/once, fetch actual immediately from historical data
                        actual = get_actual_label_for_last_interval(cfg.get("source_tickers", []) + cfg.get("other_etfs", []), cfg.get("target", "target"), interval)
                        if actual is not None:
                            row["actual_label"] = actual
                            row["outcome"] = int(row["pred_label"] == actual)
                            save_prediction_row(row)
                            print("Dry-run actual fetched ->", row["outcome"])
                    else:
                        # sleep until next interval end and then fetch actual label
                        time.sleep(predict_seconds)
                        actual = get_actual_label_for_last_interval(cfg.get("source_tickers", []) + cfg.get("other_etfs", []), cfg.get("target", "target"), interval)
                        if actual is not None:
                            row["actual_label"] = actual
                            row["outcome"] = int(row["pred_label"] == actual)
                            save_prediction_row(row)
                            print(f"Pred outcome: {row['outcome']} (pred={row['pred_label']} actual={actual})")
                if once:
                    break
                # small pause before next cycle
                time.sleep(1)
            else:
                # sleep until next market check (30s)
                print("Market closed. Sleeping 30s...")
                time.sleep(30)
    except KeyboardInterrupt:
        print("Stopped by user")


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to config.yaml", default=None)
    parser.add_argument("--once", action="store_true", help="Run one prediction cycle and exit")
    parser.add_argument("--dry-run", action="store_true", help="Use historical data to validate immediately (no waiting)")
    args = parser.parse_args()
    run_loop(args.config, once=args.once, dry_run=args.dry_run)


if __name__ == "__main__":
    cli()
